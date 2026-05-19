import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import logging

from src.data.data_list import ImageList
from src.utils.utils import *

logger = logging.getLogger(__name__)

class NegProxyLinear(nn.Module):
    """
    Per-class linear negative classifier on frozen CLIP features, with an
    explicit set of learnable class prototypes used for contrastive reg.

    forward() → (neg_logits, proto_sim)
      neg_logits : (B, C)  — higher = more "negative" / less likely this class
      proto_sim  : (B, C)  — cosine similarity to each class prototype
    """
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feat_dim, num_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -1.0)   # sigmoid(-1) ≈ 0.27

        self.prototypes = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.orthogonal_(self.prototypes)         # maximise initial separation

    def forward(self, x: torch.Tensor):
        neg_logits = self.linear(x)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        proto_sim  = x @ proto_norm.T               # (B, C), x already L2-normed
        return neg_logits, proto_sim


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def neg_proxy_loss(
    neg_logits:  torch.Tensor,   # (B, C)
    proto_sim:   torch.Tensor,   # (B, C)
    labels:      torch.Tensor,   # (B,)
    margin:      float = 2.0,
    alpha:       float = 0.3,
    noise_feats: torch.Tensor | None = None,   # augmented feats for consistency
    mlp:         nn.Module | None = None,
    lambda_dc:   float = 0.1,
) -> torch.Tensor:
    """
    Three-term loss:

    1. Margin ranking  — true-class logit should be the lowest (most "positive"):
         loss = mean_j≠k  max(0, margin - (logit_j - logit_k))

    2. Prototype cross-entropy  — source prototype similarity should be
         discriminative; provides a class-structured anchor for CLIP space.

    3. Domain consistency (optional)  — KL( p_aug || p_clean ) encourages
         predictions to be stable under Gaussian feature perturbation.
    """
    B, C = neg_logits.shape

    # 1. Margin ranking loss
    pos_logits = neg_logits.gather(1, labels.unsqueeze(1))          # (B, 1)
    margin_loss = F.relu(margin - (neg_logits - pos_logits)).mean()

    # 2. Prototype contrastive loss  (cross-entropy over cosine similarities)
    proto_loss = F.cross_entropy(proto_sim, labels)

    loss = margin_loss + alpha * proto_loss

    # 3. Domain consistency regularisation
    if noise_feats is not None and mlp is not None:
        neg_aug, _ = mlp(noise_feats)
        p_clean = torch.sigmoid(neg_logits).detach()
        p_aug   = torch.sigmoid(neg_aug)
        dc_loss  = F.kl_div((p_aug + 1e-8).log(), p_clean, reduction='batchmean')
        loss = loss + lambda_dc * dc_loss

    return loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def data_load(cfg, clip_preprocess):
    train_bs  = cfg.TEST.BATCH_SIZE
    txt_src   = open(cfg.s_dset_path).readlines()
    dsize     = len(txt_src)
    tr_size   = int(0.9 * dsize)

    if cfg.SOURCE.TRTE == "val":
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt    = txt_src

    make_loader = lambda txt, bs, shuffle: DataLoader(
        ImageList(txt, transform=clip_preprocess),
        batch_size=bs, shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS, drop_last=False,
    )
    return {
        "source_tr": make_loader(tr_txt, train_bs,     True),
        "source_te": make_loader(te_txt, train_bs * 2, False),
    }


# ---------------------------------------------------------------------------
# Evaluation  (ranking-based — actually measures the negative proxy quality)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_neg_proxy(loader, clip_model, mlp, cfg):
    """
    Rank-1 accuracy: fraction of samples where the true class has the
    *lowest* negative logit (i.e. is ranked most "positive").
    Mean rank reported alongside for finer resolution.
    """
    mlp.eval()
    rank1_correct = mean_rank_sum = n = 0

    for inputs, labels in loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        feats = clip_model.encode_image(inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        neg_logits, _ = mlp(feats)
        ranks = neg_logits.argsort(dim=1).argsort(dim=1)           # rank of each class
        true_ranks = ranks.gather(1, labels.unsqueeze(1)).squeeze(1)

        rank1_correct  += (true_ranks == 0).sum().item()
        mean_rank_sum  += true_ranks.float().sum().item()
        n              += inputs.size(0)

    mlp.train()
    return rank1_correct / n * 100, mean_rank_sum / n



def train_neg_proxy(cfg):
    # 1. Frozen CLIP backbone
    import clip
    arch        = getattr(getattr(cfg, 'DIFF_ALIGN', None), 'ARCH', None) or 'ViT-B/32'
    clip_model  = clip.load(arch)[0].float().cuda().eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    logger.info(f"CLIP backbone: {arch}  (frozen)")

    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    feat_dim  = clip_model.visual.output_dim
    class_num = cfg.class_num
    logger.info(f"classes={class_num}, feat_dim={feat_dim}")

    # 2. Data
    dset_loaders = data_load(cfg, clip_transform)

    # 3. Model + optimiser
    mlp       = NegProxyLinear(feat_dim, class_num).cuda()
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 4. Training
    max_epochs      = 100
    eval_interval   = 10
    noise_std       = 0.05   # Gaussian std for domain consistency perturbation
    best_val_rank1  = -1.0
    best_state      = None

    for epoch in range(max_epochs):
        mlp.train()
        epoch_loss = 0.0

        for inputs, labels in dset_loaders["source_tr"]:
            inputs, labels = inputs.cuda(), labels.cuda()

            with torch.no_grad():
                feats = clip_model.encode_image(inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                # Augmented features: Gaussian noise → re-normalise
                noise_feats = feats + noise_std * torch.randn_like(feats)
                noise_feats = noise_feats / noise_feats.norm(dim=-1, keepdim=True)

            neg_logits, proto_sim = mlp(feats)
            loss = neg_proxy_loss(
                neg_logits, proto_sim, labels,
                noise_feats=noise_feats, mlp=mlp,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            rank1, mean_rank = eval_neg_proxy(dset_loaders["source_te"], clip_model, mlp, cfg)
            logger.info(
                f"Epoch {epoch+1:3d}/{max_epochs}  "
                f"train_loss={epoch_loss/len(dset_loaders['source_tr']):.4f}  "
                f"rank1_acc={rank1:.1f}%  mean_rank={mean_rank:.2f}"
            )
            if rank1 > best_val_rank1:
                best_val_rank1 = rank1
                best_state     = {k: v.clone() for k, v in mlp.state_dict().items()}
                logger.info(f"  -> new best  rank1={best_val_rank1:.1f}%")

    # 5. Save
    mlp.load_state_dict(best_state)
    os.makedirs(cfg.output_dir_src, exist_ok=True)
    save_path = osp.join(cfg.output_dir_src, "source_C_aux_neg_head.pt")
    torch.save(best_state, save_path)
    logger.info(f"Saved → {save_path}")

    return mlp
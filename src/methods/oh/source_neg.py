import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging

from src.data.data_list import ImageList
from src.utils.utils import *

logger = logging.getLogger(__name__)

# ======================================================================
# Approach B: CLIP Feature-Space Negative Proxy
#
# Trains a lightweight MLP on frozen CLIP image features to predict
# per-class "negativeness" scores via multi-label BCE.
#
# For a source sample of class k:
#   target[k] = 0     (this IS class k, not "not k")
#   target[j!=k] = 1  (this is NOT class j)
#
# At test time on target domain, the learned decision boundaries
# transfer through CLIP's domain-invariant feature space.
# ======================================================================


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def data_load(cfg, clip_preprocess):
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_src = open(cfg.s_dset_path).readlines()

    if cfg.SOURCE.TRTE == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        tr_txt, te_txt = torch.utils.data.random_split(
            txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        _, te_txt = torch.utils.data.random_split(
            txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=clip_preprocess)
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"], batch_size=train_bs, shuffle=True,
        num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=clip_preprocess)
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"], batch_size=train_bs * 2, shuffle=False,
        num_workers=cfg.NUM_WORKERS, drop_last=False)

    return dset_loaders


class NegProxyLinear(nn.Module):
    """
    Per-class linear negative classifiers on frozen CLIP features.
    N independent binary classifiers: for class k, predicts whether
    the image is NOT class k (=1) or IS class k (=0).

    Single linear layer (no hidden) so that per-class gradients
    are independent — the 64 "positive" classes don't drown out
    the 1 "negative" class through a shared hidden representation.

    Bias initialized negative → sigmoid starts < 0.5, which matches
    the prior that most images are NOT most classes.
    """
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(feat_dim, num_classes)
        # Negative bias initialization: sigmoid(-1.0) ≈ 0.27 < 0.5
        nn.init.constant_(self.linear.bias, -1.0)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


def eval_neg_proxy(loader, clip_model, mlp, cfg):
    """Evaluate neg proxy: BCE loss + per-class metrics."""
    mlp.eval()
    total_loss = 0.0
    total_true_neg = 0.0   # sigmoid < 0.5 on true class (correct)
    total_other_pos = 0.0  # sigmoid > 0.5 on at least one other class
    n_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            feats = clip_model.encode_image(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            neg_logits = mlp(feats)

            target = torch.full_like(neg_logits, 0.95)
            target.scatter_(1, labels.unsqueeze(1), 0.05)
            loss = F.binary_cross_entropy_with_logits(neg_logits, target)
            total_loss += loss.item() * inputs.size(0)

            neg_probs = torch.sigmoid(neg_logits)
            # True class correctly identified as "not negative"?
            true_probs = neg_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            total_true_neg += (true_probs < 0.5).float().sum().item()
            n_samples += inputs.size(0)

    mlp.train()
    return total_loss / n_samples, total_true_neg / n_samples * 100


def train_neg_proxy(cfg):
    logger.info("=== Approach B: Training CLIP Negative Proxy ===")

    # --- 1. Load CLIP model (frozen feature extractor) ---
    import clip
    arch = getattr(getattr(cfg, 'DIFF_ALIGN', None), 'ARCH', None) or 'ViT-B/32'
    logger.info(f"Loading CLIP model: {arch}")
    clip_model = clip.load(arch)[0]
    clip_model.float().cuda()
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Define CLIP transform inline (avoid relying on clip.load() preprocess)
    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    # Infer feature dimension from CLIP
    feat_dim = clip_model.visual.output_dim

    # --- 2. Data loading ---
    dset_loaders = data_load(cfg, clip_transform)
    class_num = cfg.class_num
    logger.info(f"Source classes: {class_num}, feature dim: {feat_dim}")

    # --- 3. Build negative proxy (single linear layer: per-class gradients independent) ---
    mlp = NegProxyLinear(feat_dim, class_num).cuda()
    logger.info(f"NegProxy: Linear({feat_dim} -> {class_num}), bias_init=-1.0")

    # --- 4. Optimizer ---
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # --- 5. Training ---
    max_epochs = 50
    interval = 5
    best_val_loss = float('inf')
    best_state = None

    logger.info(f"Training for {max_epochs} epochs...")
    for epoch in range(max_epochs):
        mlp.train()
        epoch_loss = 0.0
        n_batches = 0

        for inputs, labels in dset_loaders["source_tr"]:
            inputs, labels = inputs.cuda(), labels.cuda()

            # Extract frozen CLIP features
            with torch.no_grad():
                feats = clip_model.encode_image(inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)

            # Forward through MLP
            neg_logits = mlp(feats)

            # Multi-label target: 0 for true class, 1 for others
            target = torch.full_like(neg_logits, 0.95)
            target.scatter_(1, labels.unsqueeze(1), 0.05)

            loss = F.binary_cross_entropy_with_logits(neg_logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % interval == 0 or epoch == 0:
            val_loss, true_neg_acc = eval_neg_proxy(
                dset_loaders["source_te"], clip_model, mlp, cfg)
            log_str = (f"Epoch {epoch+1}/{max_epochs}: "
                       f"train_loss={avg_loss:.4f}, "
                       f"val_loss={val_loss:.4f}, "
                       f"true_neg_acc={true_neg_acc:.1f}%")
            logger.info(log_str)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
                logger.info(f"  -> new best (val_loss={best_val_loss:.4f})")

    # --- 6. Save ---
    mlp.load_state_dict(best_state)
    output_dir = cfg.output_dir_src
    os.makedirs(output_dir, exist_ok=True)
    save_path = osp.join(output_dir, "source_C_aux_neg_head.pt")
    torch.save(best_state, save_path)
    logger.info(f"Negative proxy saved to {save_path}")

    return mlp

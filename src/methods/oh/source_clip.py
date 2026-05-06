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

from src.models import network
from src.data.data_list import ImageList
from src.utils.utils import *
from src.utils.prompt_tuning_diff_new_optim import promote_learning_diff_promote_learning
import clip

logger = logging.getLogger(__name__)

# ==========================================================================================
# Helper Functions (Referenced from previous scripts)
# ==========================================================================================

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(cfg): 
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_src = open(cfg.s_dset_path).readlines()
    txt_val = open(cfg.test_dset_path).readlines() # Using test_dset_path for validation

    if cfg.SOURCE.TRTE == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        tr_txt = txt_src
        te_txt = txt_val

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs*2, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)

    return dset_loaders

def cal_acc_clip(loader, mypromt_model, cfg):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1]
            
            logits, _ = mypromt_model(inputs)
            
            if start_test:
                all_logits = logits.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_logits = torch.cat((all_logits, logits.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # Positive Accuracy (on first C logits)
    logits_pos = all_logits[:, :cfg.class_num]
    _, predict_pos = torch.max(logits_pos, 1)
    acc_pos = torch.sum(torch.squeeze(predict_pos).float() == all_label).item() / float(all_label.size()[0])

    # Negative Accuracy (on last C logits)
    logits_neg = all_logits[:, cfg.class_num:]
    _, predict_neg = torch.min(logits_neg, 1)
    acc_neg = torch.sum(torch.squeeze(predict_neg).float() == all_label).item() / float(all_label.size()[0])
   
    return acc_pos*100, acc_neg*100

# ==========================================================================================
# Main Training Function
# ==========================================================================================
def train_clip_prompt_dual(cfg):
    logger.info("Starting CLIP Prompt Tuning (NEG-ONLY, Enhanced Version)...")
    output_dir = cfg.output_dir_src

    # ===== Load class names =====
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    classnames = [name.replace("_", " ") for name in List_rd]
    cfg.classname = classnames

    # ===== Data =====
    dset_loaders = data_load(cfg)

    # ===== Load CLIP =====
    clip_model, _, _ = clip.load(cfg.DIFF_ALIGN.ARCH, device="cuda")
    clip_model.float().eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    if cfg.class_num_hidden != cfg.class_num:
        logger.warning("class_num_hidden != class_num ; automatically matched.")
        cfg.class_num_hidden = cfg.class_num

    # ===== Prompt Learner =====
    mypromt = promote_learning_diff_promote_learning(cfg, clip_model).cuda()
    init_negative_with_mean(mypromt)
    # ===== Load NEG prompt weights =====
    # state_neg = torch.load(
    #     osp.join(output_dir, "source_prompt_learner_dual_kl_neg.pt"),
    #     map_location="cuda"
    # )
    # mypromt.negative_prompt_learner.load_state_dict(state_neg)
    # optimizer_neg = op_copy(optim.SGD([
    # {'params': mypromt.negative_prompt_learner.ctx, 'lr': cfg.OPTIM.LR * 5}
    # ]))
    # ===== Freeze POS prompt =====
    for p in mypromt.positive_prompt_learner.parameters():
        p.requires_grad = False

    # ===================================================
    #       Independent NEG optimizer (critical)
    # ===================================================
    optimizer_neg = optim.SGD(
        [{'params': mypromt.negative_prompt_learner.ctx, 'lr': cfg.OPTIM.LR * 5}],
        momentum=0.9, weight_decay=1e-3, nesterov=True
    )
    optimizer_neg = op_copy(optimizer_neg)

    # ===== Training configs =====
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0
    best_neg_acc = -1
    best_neg_prompt = None

    mypromt.train()

    # ===== Hyperparameters =====
    temp_neg = 1.0
    p_low_val = 1.0 / (cfg.class_num * 1000)
    margin_value =1.0

    W_KL = 1.0
    W_MARGIN =0.5
    W_ENT = 0.1
    W_CE = 0

    # ===== Main training loop =====
    while iter_num < max_iter:

        # Data iterator
        try:
            inputs_source, labels_source = next(source_iter)
        except (NameError, StopIteration):
            source_iter = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(source_iter)

        if inputs_source.size(0) <= 1:
            continue

        iter_num += 1

        inputs_source = inputs_source.cuda()
        labels_source = labels_source.cuda()
        batch = inputs_source.size(0)

        # ===================================================
        #                    Forward pass
        # ===================================================
        logits, _ = mypromt(inputs_source)    # (B, 2C)
        logits_neg = logits[:, cfg.class_num:]  # (B, C)

        # (1) Target distribution for KL
        log_p_neg = F.log_softmax(logits_neg / temp_neg, dim=1)
        probs_neg = F.softmax(logits_neg / temp_neg, dim=1)

        # Create target distribution: GT small, others uniform
        target_dist = logits_neg.new_full((batch, cfg.class_num), (1 - p_low_val) / (cfg.class_num - 1))
        target_dist.scatter_(1, labels_source.unsqueeze(1), p_low_val)

        loss_kl = F.kl_div(log_p_neg, target_dist, reduction="batchmean")

        # (2) Margin ranking loss (GT must be lower than others by margin)
        with torch.no_grad():
            mask = torch.ones_like(logits_neg).bool()
            mask.scatter_(1, labels_source.unsqueeze(1), False)
        max_non_gt, _ = logits_neg.masked_fill(~mask, -1e9).max(dim=1)
        gt_logit = logits_neg.gather(1, labels_source.unsqueeze(1)).squeeze(1)
        loss_margin = torch.relu(margin_value + gt_logit - max_non_gt).mean()

        # (3) Entropy regularization (prevent collapse)
        entropy_all = -torch.sum(probs_neg * torch.log(probs_neg + 1e-12), dim=1).mean()
        loss_ent = -entropy_all  # maximize entropy

        # (4) Optional CE on logits_neg
        if W_CE > 0:
            loss_ce = F.cross_entropy(logits_neg, labels_source)
        else:
            loss_ce = torch.tensor(0.0).cuda()

        # ===== Final NEG loss =====
        loss_neg = (
            W_KL * loss_kl
            + W_MARGIN * loss_margin
            + W_ENT * loss_ent
            + W_CE * loss_ce
        )

        optimizer_neg.zero_grad()
        loss_neg.backward()
        optimizer_neg.step()

        # ===================================================
        #                Logging & Evaluation
        # ===================================================
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            mypromt.eval()

            pos_acc, neg_acc = cal_acc_clip(dset_loaders["source_te"], mypromt, cfg)

            logger.info(
                f"[Iter {iter_num}/{max_iter}] "
                f"LossKL: {loss_kl.item():.4f}, "
                f"LossMargin: {loss_margin.item():.4f}, "
                f"LossEnt: {loss_ent.item():.4f}, "
                f"LossCE: {loss_ce.item():.4f}, "
                f"TotalNeg: {loss_neg.item():.4f}, "
                f"PosAcc: {pos_acc:.2f}%, NegAcc: {neg_acc:.2f}%"
            )

            # save NEG prompt
            if neg_acc >= best_neg_acc:
                best_neg_acc = neg_acc
                best_neg_prompt = mypromt.negative_prompt_learner.state_dict()
                logger.info(f"Best NEG updated at iter {iter_num}, NegAcc={neg_acc:.2f}%")

            mypromt.train()

    # ===== Save best NEG prompt =====
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Saving best NEG prompt (NegAcc={best_neg_acc:.2f}%)")
    torch.save(best_neg_prompt,
               osp.join(output_dir, "source_prompt_learner_dual_kl_neg.pt"))
    
@torch.no_grad()
def init_negative_with_mean(mypromt):
    """
    Initialize negative prompts as mean of all other positive prompts.
    This creates per-class ctx of shape (n_cls, n_ctx, ctx_dim) and replaces
    mypromt.negative_prompt_learner.ctx (and ctx_init_state) accordingly.
    """

    # device / dtype
    device = mypromt.clip_model.visual.conv1.weight.device
    dtype = mypromt.positive_prompt_learner.dtype

    # 1) 得到 positive text embeddings: shape (C, D)
    pos_prompts = mypromt.positive_prompt_learner()               # (C, L, D)
    pos_tok = mypromt.positive_prompt_learner.tokenized_prompts
    pos_emb = mypromt.text_encoder(pos_prompts, pos_tok)          # (C, D)
    pos_emb = pos_emb.to(device)                                  # ensure device
    C, D = pos_emb.shape
    n_ctx = mypromt.negative_prompt_learner.n_ctx  # number of ctx tokens (int)

    # 2) 构造每类的 negative prototype：mean of others -> (C, D)
    neg_protos = []
    for i in range(C):
        if C == 1:
            # corner case: only one class -> use negative of itself (rare)
            proto = -pos_emb[i:i+1]
        else:
            idx = [j for j in range(C) if j != i]
            proto = pos_emb[idx].mean(dim=0, keepdim=True)  # (1, D)
        neg_protos.append(proto)
    neg_protos = torch.cat(neg_protos, dim=0)               # (C, D)
    neg_protos = neg_protos / (neg_protos.norm(dim=-1, keepdim=True) + 1e-12)

    # 3) 构造 new_ctx：shape (n_cls, n_ctx, D)
    # 我们在每个 token 上复制 proto 向量并加一点小噪声以便可训练
    new_ctx = torch.zeros((C, n_ctx, D), device=device, dtype=dtype)
    for i in range(C):
        for t in range(n_ctx):
            # small scale to avoid覆盖原语义并留有可训练空间
            new_ctx[i, t] = neg_protos[i] * 0.01 + torch.randn(D, device=device, dtype=dtype) * 0.001

    # 4) set parameter and init_state consistently
    # PromptLearner expects ctx to be either (n_ctx, D) or (n_cls, n_ctx, D).
    # We will set it to (n_cls, n_ctx, D) so forward uses it directly.
    # However original PromptLearner used ctx shape (n_ctx, D) — replacing with (n_cls, n_ctx, D) is fine.
    neg_pl = mypromt.negative_prompt_learner

    # If existing ctx is a Parameter, delete/replace it safely:
    try:
        # assign new nn.Parameter
        neg_pl.ctx = nn.Parameter(new_ctx)
    except Exception as e:
        # fallback: set attribute then wrap as Parameter
        neg_pl.__dict__['ctx'] = nn.Parameter(new_ctx)

    # update ctx_init_state (used by reset)
    neg_pl.ctx_init_state = new_ctx.detach().clone()

    # Ensure dtype/device consistent
    neg_pl.ctx = neg_pl.ctx.to(device=device, dtype=dtype)

    print(f"[NEG INIT] negative ctx initialized to per-class means: new ctx shape = {neg_pl.ctx.shape}")



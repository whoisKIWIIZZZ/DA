import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import clip
import logging
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.data_list import ImageList_idx
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def image_test_clip():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])


def data_load(cfg):
    txt_test = open(cfg.test_dset_path).readlines()
    dsets = {}
    dset_loaders = {}
    dsets["test"] = ImageList_idx(txt_test, transform=image_test_clip())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=cfg.TEST.BATCH_SIZE * 3,
                                      shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders


def evaluate_zeroshot_neg(cfg):
    """Test zero-shot CLIP with 'a photo of not {cls}' as negative prompts."""
    logger.info("=== Zero-shot Negative Prompt Test ===")

    # Load CLIP
    arch = getattr(getattr(cfg, 'ProDe', None), 'ARCH', None) or 'ViT-B/32'
    logger.info(f"Loading CLIP: {arch}")
    clip_model = clip.load(arch)[0]
    clip_model.float().cuda()
    clip_model.eval()

    # Load class names
    classnames = []
    with open(cfg.name_file) as f:
        for line in f:
            classnames.extend([i for i in line.split()])
    classnames = [name.replace("_", " ") for name in classnames]
    C = len(classnames)
    logger.info(f"Dataset: {cfg.SETTING.DATASET}, classes: {C}")

    # Build positive and negative text prompts
    pos_prompts = [f"a photo of a {name}." for name in classnames]
    neg_prompts = [f"a photo of not {name}." for name in classnames]

    pos_tokens = torch.cat([clip.tokenize(p) for p in pos_prompts]).cuda()
    neg_tokens = torch.cat([clip.tokenize(p) for p in neg_prompts]).cuda()

    # Encode all text features once
    with torch.no_grad():
        pos_text_feats = clip_model.encode_text(pos_tokens)
        pos_text_feats = pos_text_feats / pos_text_feats.norm(dim=-1, keepdim=True)
        neg_text_feats = clip_model.encode_text(neg_tokens)
        neg_text_feats = neg_text_feats / neg_text_feats.norm(dim=-1, keepdim=True)

        logit_scale = clip_model.logit_scale.exp()

    # Data
    dset_loaders = data_load(cfg)
    loader = dset_loaders["test"]

    all_pos_logits = []
    all_neg_logits = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            img_feats = clip_model.encode_image(inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            pos_logits = logit_scale * img_feats @ pos_text_feats.t()
            neg_logits = logit_scale * img_feats @ neg_text_feats.t()

            all_pos_logits.append(pos_logits.cpu())
            all_neg_logits.append(neg_logits.cpu())
            all_labels.append(labels)

    pos_logits = torch.cat(all_pos_logits, dim=0)  # (N, C)
    neg_logits = torch.cat(all_neg_logits, dim=0)  # (N, C)
    labels = torch.cat(all_labels, dim=0)           # (N,)

    # ---- Positive accuracy (standard CLIP zero-shot) ----
    _, pos_pred = torch.max(pos_logits, dim=1)
    pos_acc = (pos_pred == labels).float().mean().item() * 100

    # ---- Negative-based prediction: lowest neg_logit = true class ----
    # "a photo of not {cls}" should be least matching for the TRUE class
    _, neg_pred = torch.min(neg_logits, dim=1)
    neg_acc = (neg_pred == labels).float().mean().item() * 100

    # ---- Combined: pos_logits - neg_logits (higher = more positive evidence) ----
    combined = pos_logits - neg_logits
    _, comb_pred = torch.max(combined, dim=1)
    comb_acc = (comb_pred == labels).float().mean().item() * 100

    # ---- Per-class analysis: does neg_logits[true_class] rank lowest? ----
    neg_ranks = neg_logits.argsort(dim=1).argsort(dim=1)  # 0 = lowest neg_logit
    true_neg_ranks = neg_ranks.gather(1, labels.unsqueeze(1)).float()
    mean_neg_rank = true_neg_ranks.mean().item()
    neg_rank1 = (true_neg_ranks == 0).float().mean().item() * 100  # true class is lowest neg

    # ---- Where does CLIP fail? ----
    pos_correct = (pos_pred == labels)
    pos_wrong = ~pos_correct

    # Among positive-wrong samples, can negative signal correct it?
    if pos_wrong.sum() > 0:
        neg_correct = (neg_pred == labels)
        comb_correct = (comb_pred == labels)
        neg_fixes_pos = (pos_wrong & neg_correct).float().sum().item()
        neg_fix_rate = neg_fixes_pos / pos_wrong.float().sum().item() * 100
        comb_fixes_pos = (pos_wrong & comb_correct).float().sum().item()
        comb_fix_rate = comb_fixes_pos / pos_wrong.float().sum().item() * 100
    else:
        neg_fix_rate = comb_fix_rate = 0.0

    # ---- Logging ----
    logger.info("=" * 60)
    logger.info(f"Positive (standard CLIP)  Accuracy = {pos_acc:.2f}%")
    logger.info(f"Negative (lowest=pred)    Accuracy = {neg_acc:.2f}%")
    logger.info(f"Combined (pos - neg)      Accuracy = {comb_acc:.2f}%")
    logger.info(f"Negative rank-1 rate (true cls lowest neg) = {neg_rank1:.2f}%")
    logger.info(f"Mean rank of true class in neg_logits = {mean_neg_rank:.2f} (0=best)")
    logger.info(f"Negative fixes {neg_fix_rate:.1f}% of CLIP's mistakes")
    logger.info(f"Combined  fixes {comb_fix_rate:.1f}% of CLIP's mistakes")

    # Per-class breakdown
    matrix = confusion_matrix(labels, pos_pred)
    per_cls_acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    logger.info("Per-class positive accuracy:")
    for i, name in enumerate(classnames):
        logger.info(f"  {i:3d} {name:20s}: {per_cls_acc[i]:5.1f}%")
    logger.info("=" * 60)

    # ---- Return for potential use in ProDe ----
    return {
        'pos_acc': pos_acc,
        'neg_acc': neg_acc,
        'comb_acc': comb_acc,
        'neg_rank1': neg_rank1,
        'mean_neg_rank': mean_neg_rank,
        'neg_fix_rate': neg_fix_rate,
        'comb_fix_rate': comb_fix_rate,
    }


def evaluate_semantic_prior(cfg):
    """Test text-guided semantic prior: neg_prior = softmax(pos) @ class_sim."""
    logger.info("=== Semantic Prior Evaluation ===")

    arch = getattr(getattr(cfg, 'ProDe', None), 'ARCH', None) or 'ViT-B/32'
    logger.info(f"Loading CLIP: {arch}")
    clip_model = clip.load(arch)[0]
    clip_model.float().cuda()
    clip_model.eval()

    # Load class names
    classnames = []
    with open(cfg.name_file) as f:
        for line in f:
            classnames.extend([i for i in line.split()])
    classnames = [name.replace("_", " ") for name in classnames]
    C = len(classnames)
    logger.info(f"Dataset: {cfg.SETTING.DATASET}, classes: {C}")

    # Positive text prompts
    pos_prompts = [f"a photo of a {name}." for name in classnames]
    pos_tokens = torch.cat([clip.tokenize(p) for p in pos_prompts]).cuda()

    with torch.no_grad():
        text_feats = clip_model.encode_text(pos_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # Semantic similarity matrix
        class_sim = text_feats @ text_feats.T
        class_sim.fill_diagonal_(0)
        class_sim = class_sim / (class_sim.sum(dim=1, keepdim=True) + 1e-6)

        logit_scale = clip_model.logit_scale.exp()

    logger.info("class_sim stats: diag_removed, row_normed, "
                f"max={class_sim.max().item():.3f}, mean={class_sim.mean().item():.4f}")

    # Data
    dset_loaders = data_load(cfg)
    loader = dset_loaders["test"]

    all_pos_logits = []
    all_neg_prior = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            img_feats = clip_model.encode_image(inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            pos_logits = logit_scale * img_feats @ text_feats.t()

            # Second-order semantic propagation
            p = torch.softmax(pos_logits, dim=1)
            neg_prior = p @ class_sim

            all_pos_logits.append(pos_logits.cpu())
            all_neg_prior.append(neg_prior.cpu())
            all_labels.append(labels)

    pos_logits = torch.cat(all_pos_logits, dim=0)
    neg_prior = torch.cat(all_neg_prior, dim=0)
    labels = torch.cat(all_labels, dim=0)
    p = torch.softmax(pos_logits, dim=1)

    # Baseline
    _, pos_pred = torch.max(pos_logits, dim=1)
    pos_acc = (pos_pred == labels).float().mean().item() * 100

    # Sweep alpha: corrected = softmax - alpha * neg_prior
    # Also sweep in logit space: corrected_logits = pos_logits - alpha * logit_scale * neg_prior
    logger.info("=" * 60)
    logger.info(f"Baseline CLIP zero-shot: {pos_acc:.2f}%")
    logger.info("")

    # Prob-space correction: argmax(p - alpha * neg_prior)
    best_prob, best_alpha_prob = pos_acc, 0.0
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        corrected = p - alpha * neg_prior
        _, pred = torch.max(corrected, dim=1)
        acc = (pred == labels).float().mean().item() * 100
        delta = acc - pos_acc
        logger.info(f"  prob-space  alpha={alpha:5.1f}: {acc:.2f}% (delta={delta:+.2f}%)")
        if acc > best_prob:
            best_prob = acc
            best_alpha_prob = alpha

    logger.info(f"  -> best prob-space: alpha={best_alpha_prob}, acc={best_prob:.2f}%")

    # Logit-space correction: argmax(pos_logits - alpha * logit_scale * neg_prior)
    logger.info("")
    best_logit, best_alpha_logit = pos_acc, 0.0
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        corrected = pos_logits - alpha * logit_scale * neg_prior
        _, pred = torch.max(corrected, dim=1)
        acc = (pred == labels).float().mean().item() * 100
        delta = acc - pos_acc
        logger.info(f"  logit-space alpha={alpha:5.1f}: {acc:.2f}% (delta={delta:+.2f}%)")
        if acc > best_logit:
            best_logit = acc
            best_alpha_logit = alpha

    logger.info(f"  -> best logit-space: alpha={best_alpha_logit}, acc={best_logit:.2f}%")

    # Where does CLIP fail and does residual help?
    pos_correct = (pos_pred == labels)
    pos_wrong = ~pos_correct
    if pos_wrong.sum() > 0:
        corrected = p - best_alpha_prob * neg_prior
        _, best_pred = torch.max(corrected, dim=1)
        fixes = (pos_wrong & (best_pred == labels)).float().sum().item()
        fix_rate = fixes / pos_wrong.float().sum().item() * 100
        logger.info(f"\nBest prob model fixes {fix_rate:.1f}% of CLIP's {pos_wrong.sum().item()} mistakes")

        # Per-class: which classes benefit most?
        matrix = confusion_matrix(labels, pos_pred)
        per_cls_acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        matrix_corr = confusion_matrix(labels, best_pred)
        per_cls_acc_corr = matrix_corr.diagonal() / matrix_corr.sum(axis=1) * 100
        logger.info("\nPer-class delta (corrected - baseline):")
        for i, name in enumerate(classnames):
            d = per_cls_acc_corr[i] - per_cls_acc[i]
            if abs(d) > 1.0:
                logger.info(f"  {i:3d} {name:20s}: {per_cls_acc[i]:5.1f}% -> {per_cls_acc_corr[i]:5.1f}% ({d:+.1f}%)")

    logger.info("=" * 60)

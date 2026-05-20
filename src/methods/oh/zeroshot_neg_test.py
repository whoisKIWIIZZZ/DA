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


# 多种表达"not [CLS]"语义的自然语言prompt模板
NEG_TEMPLATES = {
    "not":              "a photo of not {name}.",           # 直接"not"修饰
    "without":          "a photo without a {name}.",        # 介词"without"
    "no":               "a photo with no {name}.",          # 限定词"no"
    "is_not":           "a photo that is not a {name}.",    # 定语从句否定
    "other_than":       "a photo of something other than a {name}.",  # "other than"
    "anything_but":     "a photo of anything but a {name}.",          # "anything but"
    "sent_neg":         "this is not a photo of a {name}.", # 句子级否定
}


def evaluate_zeroshot_neg(cfg):
    """测试多种自然语言否定prompt的zero-shot CLIP表现."""
    logger.info("=== Zero-shot Negative Prompt Test (多模板) ===")

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

    # 正prompt和所有负prompt模板的文本特征
    pos_prompts = [f"a photo of a {name}." for name in classnames]
    pos_tokens = torch.cat([clip.tokenize(p) for p in pos_prompts]).cuda()

    with torch.no_grad():
        pos_text_feats = clip_model.encode_text(pos_tokens)
        pos_text_feats = pos_text_feats / pos_text_feats.norm(dim=-1, keepdim=True)

        # 编码所有负模板
        neg_text_feats = {}
        for key, template in NEG_TEMPLATES.items():
            prompts = [template.format(name=name) for name in classnames]
            tokens = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            feats = clip_model.encode_text(tokens)
            neg_text_feats[key] = feats / feats.norm(dim=-1, keepdim=True)

        logit_scale = clip_model.logit_scale.exp().item()

    # 提取所有图像特征（只做一次）
    logger.info("Extracting image features...")
    dset_loaders = data_load(cfg)
    loader = dset_loaders["test"]

    all_img_feats = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            img_feats = clip_model.encode_image(inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(img_feats)
            all_labels.append(labels.cpu())

    img_feats = torch.cat(all_img_feats, dim=0)  # (N, D), on GPU
    labels = torch.cat(all_labels, dim=0)          # (N,), on CPU
    N = img_feats.shape[0]
    logger.info(f"Test samples: {N}")

    # 正prompt预测（基线）—— 结果移回CPU
    pos_logits = (logit_scale * img_feats @ pos_text_feats.t()).cpu()
    _, pos_pred = torch.max(pos_logits, dim=1)
    pos_acc = (pos_pred == labels).float().mean().item() * 100
    pos_wrong = ~(pos_pred == labels)

    # 逐模板评估
    results = {}
    for key, neg_feats in neg_text_feats.items():
        neg_logits = (logit_scale * img_feats @ neg_feats.t()).cpu()

        # argmin 预测
        _, neg_pred = torch.min(neg_logits, dim=1)
        neg_acc = (neg_pred == labels).float().mean().item() * 100

        # 组合预测
        combined = pos_logits - neg_logits
        _, comb_pred = torch.max(combined, dim=1)
        comb_acc = (comb_pred == labels).float().mean().item() * 100

        # rank 分析
        neg_ranks = neg_logits.argsort(dim=1).argsort(dim=1)
        true_neg_ranks = neg_ranks.gather(1, labels.unsqueeze(1)).float()
        mean_neg_rank = true_neg_ranks.mean().item()
        neg_rank1 = (true_neg_ranks == 0).float().mean().item() * 100

        # 修复率
        if pos_wrong.sum() > 0:
            neg_fixes = (pos_wrong & (neg_pred == labels)).float().sum().item()
            neg_fix_rate = neg_fixes / pos_wrong.float().sum().item() * 100
            comb_fixes = (pos_wrong & (comb_pred == labels)).float().sum().item()
            comb_fix_rate = comb_fixes / pos_wrong.float().sum().item() * 100
        else:
            neg_fix_rate = comb_fix_rate = 0.0

        results[key] = {
            'template': NEG_TEMPLATES[key],
            'neg_acc': neg_acc, 'comb_acc': comb_acc,
            'neg_rank1': neg_rank1, 'mean_neg_rank': mean_neg_rank,
            'neg_fix_rate': neg_fix_rate, 'comb_fix_rate': comb_fix_rate,
        }

    # ---- 输出对比表 ----
    logger.info("=" * 90)
    logger.info(f"Baseline CLIP zero-shot (positive): {pos_acc:.2f}%  (N={N}, C={C})")
    logger.info("")
    header = (f"{'Template':<16s} {'Prompt':<44s} "
              f"{'NegAcc':>7s} {'CombAcc':>7s} {'Rank1':>7s} {'NFix%':>7s} {'CFix%':>7s}")
    logger.info(header)
    logger.info("-" * 90)
    for key, r in results.items():
        logger.info(f"{key:<16s} {r['template']:<44s} "
                    f"{r['neg_acc']:7.2f}% {r['comb_acc']:7.2f}% "
                    f"{r['neg_rank1']:7.2f}% {r['neg_fix_rate']:7.1f}% {r['comb_fix_rate']:7.1f}%")
    logger.info("=" * 90)

    # 找到最好的模板，打印per-class详情
    best_key = max(results, key=lambda k: results[k]['comb_acc'])
    best = results[best_key]
    logger.info(f"\nBest template: [{best_key}] \"{best['template']}\"")
    logger.info(f"  NegAcc={best['neg_acc']:.2f}%  CombAcc={best['comb_acc']:.2f}%  "
                f"Rank1={best['neg_rank1']:.2f}%  NFix={best['neg_fix_rate']:.1f}%")

    # Per-class breakdown (baseline)
    matrix = confusion_matrix(labels, pos_pred)
    per_cls_acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    logger.info("\nPer-class positive accuracy:")
    for i, name in enumerate(classnames):
        logger.info(f"  {i:3d} {name:20s}: {per_cls_acc[i]:5.1f}%")

    return {'pos_acc': pos_acc, 'best_template': best_key, 'results': results}


def data_load_src(cfg):
    """加载 source + target 数据"""
    dset_loaders = data_load(cfg)
    txt_src = open(cfg.s_dset_path).readlines()
    dsets_src = ImageList_idx(txt_src, transform=image_test_clip())
    dset_loaders["source"] = DataLoader(dsets_src, batch_size=cfg.TEST.BATCH_SIZE * 3,
                                         shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders


def evaluate_learnable_neg(cfg):
    """
    PositiveCoOp 风格：负向嵌入跳过文本编码器，直接在 CLIP 特征空间学习自由向量。
    在 source 训练，在 target 测试，检验跨域迁移性（domain invariance）。
    """
    logger.info("=== Learnable Negative Embedding (S→T domain invariance test) ===")

    arch = getattr(getattr(cfg, 'ProDe', None), 'ARCH', None) or 'ViT-B/32'
    logger.info(f"Loading CLIP: {arch}")
    clip_model = clip.load(arch)[0]
    clip_model.float().cuda()
    clip_model.eval()

    # 类别名
    classnames = []
    with open(cfg.name_file) as f:
        for line in f:
            classnames.extend([i for i in line.split()])
    classnames = [name.replace("_", " ") for name in classnames]
    C = len(classnames)
    D = clip_model.visual.output_dim
    logger.info(f"Dataset: {cfg.SETTING.DATASET}, S={cfg.domain[cfg.SETTING.S]}, "
                f"T={cfg.domain[cfg.SETTING.T]}, classes={C}, dim={D}")

    # 正prompt文本特征（固定，不训练）
    pos_prompts = [f"a photo of a {name}." for name in classnames]
    pos_tokens = torch.cat([clip.tokenize(p) for p in pos_prompts]).cuda()
    with torch.no_grad():
        pos_text = clip_model.encode_text(pos_tokens)
        pos_text = pos_text / pos_text.norm(dim=-1, keepdim=True)
        logit_scale = clip_model.logit_scale.exp().item()

    # 负向可学习嵌入（C, D）—— 不走文本编码器
    neg_embed = nn.Parameter(torch.randn(C, D, device=pos_text.device, dtype=pos_text.dtype) * 0.02)

    # 文本编码的负prompt（用于对比）
    neg_text_feats = {}
    for key in ["not", "without", "other_than"]:
        template = NEG_TEMPLATES[key]
        prompts = [template.format(name=name) for name in classnames]
        tokens = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            feats = clip_model.encode_text(tokens)
            neg_text_feats[key] = feats / feats.norm(dim=-1, keepdim=True)

    # 数据
    dset_loaders = data_load_src(cfg)
    src_loader = dset_loaders["source"]
    tgt_loader = dset_loaders["test"]
    logger.info(f"S={cfg.domain[cfg.SETTING.S]}: {len(src_loader.dataset)}, "
                f"T={cfg.domain[cfg.SETTING.T]}: {len(tgt_loader.dataset)}")

    # ===== 在 Source 上训练 neg_embed =====
    optimizer = torch.optim.Adam([neg_embed], lr=1e-3, weight_decay=1e-4)
    num_epochs = 20
    logger.info(f"Training on S for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for data in src_loader:
            inputs, labels = data[0].cuda(), data[1].cuda()
            with torch.no_grad():
                img_feats = clip_model.encode_image(inputs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            neg_logits = logit_scale * img_feats @ neg_embed.t()
            loss = nn.CrossEntropyLoss()(-neg_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.min(neg_logits, dim=1)
            correct += (pred == labels).float().sum().item()
            total += labels.size(0)

        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1:3d}: loss={total_loss/len(src_loader):.4f}, "
                        f"src_neg_acc={correct/total*100:.2f}%")

    # ===== 提取 Target 图像特征 =====
    logger.info(f"\nEvaluating on T ({cfg.domain[cfg.SETTING.T]})...")
    all_img_feats = []
    all_labels = []
    with torch.no_grad():
        for data in tgt_loader:
            inputs, labels = data[0].cuda(), data[1]
            img_feats = clip_model.encode_image(inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(img_feats.cpu())
            all_labels.append(labels)

    img_feats = torch.cat(all_img_feats, dim=0)   # CPU
    labels = torch.cat(all_labels, dim=0)
    N = img_feats.shape[0]

    # Baseline
    pos_logits = logit_scale * img_feats @ pos_text.cpu().t()
    _, pos_pred = torch.max(pos_logits, dim=1)
    pos_acc = (pos_pred == labels).float().mean().item() * 100
    pos_wrong = ~(pos_pred == labels)

    # 文本编码 neg（原方案）
    text_results = {}
    for key, neg_feats in neg_text_feats.items():
        neg_feats_cpu = neg_feats.cpu()
        nlogits = logit_scale * img_feats @ neg_feats_cpu.t()
        _, npred = torch.min(nlogits, dim=1)
        nacc = (npred == labels).float().mean().item() * 100
        combined = pos_logits - nlogits
        _, cpred = torch.max(combined, dim=1)
        cacc = (cpred == labels).float().mean().item() * 100
        text_results[key] = {'neg_acc': nacc, 'comb_acc': cacc}

    # 可学习 neg（S→T）
    neg_embed_cpu = neg_embed.detach().cpu()
    neg_logits = logit_scale * img_feats @ neg_embed_cpu.t()
    _, neg_pred = torch.min(neg_logits, dim=1)
    neg_acc = (neg_pred == labels).float().mean().item() * 100

    combined = pos_logits - neg_logits
    _, comb_pred = torch.max(combined, dim=1)
    comb_acc = (comb_pred == labels).float().mean().item() * 100

    # Rank 分析
    neg_ranks = neg_logits.argsort(dim=1).argsort(dim=1)
    true_neg_ranks = neg_ranks.gather(1, labels.unsqueeze(1)).float()
    rank1 = (true_neg_ranks == 0).float().mean().item() * 100
    mean_rank = true_neg_ranks.mean().item()

    # Fix rate
    if pos_wrong.sum() > 0:
        neg_fixes = (pos_wrong & (neg_pred == labels)).float().sum().item()
        neg_fix_rate = neg_fixes / pos_wrong.float().sum().item() * 100
        comb_fixes = (pos_wrong & (comb_pred == labels)).float().sum().item()
        comb_fix_rate = comb_fixes / pos_wrong.float().sum().item() * 100
    else:
        neg_fix_rate = comb_fix_rate = 0.0

    # ===== 保存模型 =====
    save_path = osp.join(cfg.output_dir_src, f"learned_neg_{cfg.domain[cfg.SETTING.S]}.pt")
    os.makedirs(cfg.output_dir_src, exist_ok=True)
    torch.save({
        'neg_embed': neg_embed.detach().cpu(),
        'pos_text': pos_text.cpu(),
        'classnames': classnames,
        'dataset': cfg.SETTING.DATASET,
        'arch': arch,
        'C': C,
        'D': D,
        'logit_scale': logit_scale,
    }, save_path)
    logger.info(f"Saved: {save_path}")

    # ===== 输出对比 =====
    logger.info("=" * 90)
    logger.info(f"Baseline CLIP zero-shot (positive): {pos_acc:.2f}%  (N={N}, C={C})")
    logger.info(f"S={cfg.domain[cfg.SETTING.S]} → T={cfg.domain[cfg.SETTING.T]}")
    logger.info("")
    header = (f"{'Method':<35s} {'NegAcc':>8s} {'CombAcc':>8s} "
              f"{'Rank1':>8s} {'NFix%':>8s} {'CFix%':>8s}")
    logger.info(header)
    logger.info("-" * 90)
    for key in ["not", "without", "other_than"]:
        r = text_results[key]
        logger.info(f"  text-encoded ({key:12s})     {r['neg_acc']:7.2f}% {r['comb_acc']:7.2f}%")
    logger.info(f"  learned free embed (S→T)    {neg_acc:7.2f}% {comb_acc:7.2f}% "
                f"{rank1:7.2f}% {neg_fix_rate:7.1f}% {comb_fix_rate:7.1f}%")
    logger.info(f"  (mean rank of true class: {mean_rank:.2f})")
    logger.info("=" * 90)

    return {
        'pos_acc': pos_acc,
        'text_results': text_results,
        'learned_neg_acc': neg_acc, 'learned_comb_acc': comb_acc,
        'neg_rank1': rank1, 'mean_neg_rank': mean_rank,
        'neg_fix_rate': neg_fix_rate, 'comb_fix_rate': comb_fix_rate,
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

        logit_scale = clip_model.logit_scale.exp().item()

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

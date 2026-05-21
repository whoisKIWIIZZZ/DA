import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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



def sweep_alpha_neg(pos_logits, neg_logits, labels, alphas=None):
    """对 neg_logits 做 alpha sweep: combined = pos_logits - alpha * neg_logits"""
    if alphas is None:
        alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    baseline = (pos_logits.argmax(1) == labels).float().mean().item() * 100
    best_alpha = 0.0
    best_acc = baseline
    for alpha in alphas:
        combined = pos_logits - alpha * neg_logits
        acc = (combined.argmax(1) == labels).float().mean().item() * 100
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    return best_alpha, best_acc


def evaluate_learnable_neg(cfg):
    """
    PositiveCoOp 风格：负向嵌入跳过文本编码器，直接在 CLIP 特征空间学习自由向量。
    在 source 训练，在 target 测试，检验跨域迁移性（domain invariance）。
    """
    do_train = getattr(getattr(cfg, 'LEARNABLE_NEG', None), 'TRAIN', True)
    logger.info(f"=== Learnable Negative Embedding (train={do_train}) ===")

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

    s_name = cfg.domain[cfg.SETTING.S]
    save_path = osp.join(cfg.output_dir_src, f"learned_neg_{s_name}.pt")

    if do_train:
        # ===== 加载 Source 数据 =====
        txt_src = open(cfg.s_dset_path).readlines()
        dsets_src = ImageList_idx(txt_src, transform=image_test_clip())
        src_loader = DataLoader(dsets_src, batch_size=cfg.TEST.BATCH_SIZE * 3,
                                shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
        logger.info(f"S={s_name}: {len(src_loader.dataset)} samples")

        # ===== 训练 neg_embed（反混淆：pos 错 → neg 标记 pos 的错类） =====
        lambda_anti = 1.0  # 反混淆 loss 权重
        anti_margin = 5.0  # hinge margin：neg_logits[错类] 比 neg_logits[真类] 高多少即可
        optimizer = torch.optim.Adam([neg_embed], lr=1e-3, weight_decay=1e-4)
        num_epochs = 20
        logger.info(f"Training on S for {num_epochs} epochs (λ_anti={lambda_anti}, margin={anti_margin})...")
        logger.info(f"  标准 CE:  neg_logits[真类] ↓")
        logger.info(f"  反混淆 hinge:  pos 错时 neg_logits[错类] > neg_logits[真类] + {anti_margin}")

        for epoch in range(num_epochs):
            total_loss = 0
            total_ce = 0
            total_anti = 0
            neg_correct = 0
            pos_correct = 0
            total = 0
            anti_samples = 0
            for data in src_loader:
                inputs, labels = data[0].cuda(), data[1].cuda()
                with torch.no_grad():
                    img_feats = clip_model.encode_image(inputs)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    pos_logits = logit_scale * img_feats @ pos_text.t()
                    pos_pred = pos_logits.argmax(1)

                neg_embed_norm = neg_embed / neg_embed.norm(dim=-1, keepdim=True)
                neg_logits = logit_scale * img_feats @ neg_embed_norm.t()

                # 标准 CE：真类的 neg_logits 应最小
                loss_ce = nn.CrossEntropyLoss()(-neg_logits, labels)

                # 反混淆 hinge：pos 预测错的类 j，neg 应该给高分
                #   neg_logits[j] > neg_logits[真类] + margin 即可，避免发散
                pos_wrong_mask = (pos_pred != labels)
                if pos_wrong_mask.any():
                    neg_wrong = neg_logits[pos_wrong_mask].gather(
                        1, pos_pred[pos_wrong_mask].unsqueeze(1)).squeeze(1)
                    neg_true = neg_logits[pos_wrong_mask].gather(
                        1, labels[pos_wrong_mask].unsqueeze(1)).squeeze(1)
                    loss_anti = F.relu(anti_margin - (neg_wrong - neg_true)).mean()
                    anti_samples += pos_wrong_mask.sum().item()
                else:
                    loss_anti = torch.tensor(0.0, device=neg_logits.device)

                loss = loss_ce + lambda_anti * loss_anti

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_ce += loss_ce.item()
                total_anti += loss_anti.item() if isinstance(loss_anti, torch.Tensor) and loss_anti.item() != 0 else 0.0
                neg_correct += (neg_logits.argmin(1) == labels).float().sum().item()
                pos_correct += (pos_pred == labels).float().sum().item()
                total += labels.size(0)

            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1:3d}: loss={total_loss/len(src_loader):.4f} "
                            f"(CE={total_ce/len(src_loader):.4f} anti={total_anti/max(len(src_loader),1):.4f}) "
                            f"neg_acc={neg_correct/total*100:.1f}% pos_acc={pos_correct/total*100:.1f}%"
                            f"  anti_samples={anti_samples/len(src_loader):.0f}/batch")

        # 保存模型
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
    else:
        # 加载已有模型
        logger.info(f"Loading pre-trained neg_embed from {save_path}")
        ckpt = torch.load(save_path, map_location='cpu')
        neg_embed.data.copy_(ckpt['neg_embed'].to(neg_embed.device, neg_embed.dtype))
        logger.info(f"Loaded: S={ckpt.get('dataset', '?')}, C={ckpt['C']}, D={ckpt['D']}")

    # ===== 遍历所有 T 域评估 =====
    pos_text_cpu = pos_text.cpu()
    neg_embed_cpu = neg_embed.detach().cpu()
    neg_embed_cpu = neg_embed_cpu / neg_embed_cpu.norm(dim=-1, keepdim=True)  # L2 归一化，对齐 pos_text
    neg_text_cpu = {k: v.cpu() for k, v in neg_text_feats.items()}

    all_results = {}  # t_name -> metrics

    for t_idx, t_name in enumerate(cfg.domain):
        test_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + t_name + '_list.txt'
        txt_test = open(test_path).readlines()
        dsets_t = ImageList_idx(txt_test, transform=image_test_clip())
        tgt_loader = DataLoader(dsets_t, batch_size=cfg.TEST.BATCH_SIZE * 3,
                                shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)

        logger.info(f"Evaluating T={t_name} ({len(tgt_loader.dataset)} samples)...")

        all_img_feats = []
        all_labels = []
        with torch.no_grad():
            for data in tgt_loader:
                inputs, labels = data[0].cuda(), data[1]
                img_feats = clip_model.encode_image(inputs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                all_img_feats.append(img_feats.cpu())
                all_labels.append(labels)

        img_feats = torch.cat(all_img_feats, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # 辅助函数：计算全套指标
        def compute_metrics(logits, is_neg=True):
            """is_neg=True: argmin 预测; is_neg=False: argmax 预测"""
            if is_neg:
                _, pred = torch.min(logits, dim=1)
            else:
                _, pred = torch.max(logits, dim=1)
            acc = (pred == labels).float().mean().item() * 100

            ranks = logits.argsort(dim=1).argsort(dim=1)
            if is_neg:
                true_ranks = ranks.gather(1, labels.unsqueeze(1)).float()
            else:
                true_ranks = (C - 1 - ranks).gather(1, labels.unsqueeze(1)).float()
            rank1 = (true_ranks == 0).float().mean().item() * 100
            mean_rank = true_ranks.mean().item()

            if pos_wrong.sum() > 0:
                fixes = (pos_wrong & (pred == labels)).float().sum().item()
                fix_rate = fixes / pos_wrong.float().sum().item() * 100
            else:
                fix_rate = 0.0
            return acc, rank1, mean_rank, fix_rate

        # Baseline: positive only
        pos_logits = logit_scale * img_feats @ pos_text_cpu.t()
        _, pos_pred = torch.max(pos_logits, dim=1)
        pos_acc = (pos_pred == labels).float().mean().item() * 100
        pos_wrong = ~(pos_pred == labels)

        # 文本编码 neg
        text_metrics = {}
        for key in ["not", "without", "other_than"]:
            nlogits = logit_scale * img_feats @ neg_text_cpu[key].t()
            neg_acc, rank1, mean_rank, neg_fix = compute_metrics(nlogits, is_neg=True)
            combined = pos_logits - nlogits
            comb_acc, _, _, comb_fix = compute_metrics(combined, is_neg=False)
            text_metrics[key] = {
                'neg_acc': neg_acc, 'comb_acc': comb_acc,
                'rank1': rank1, 'mean_rank': mean_rank,
                'neg_fix': neg_fix, 'comb_fix': comb_fix,
            }

        # 可学习 neg (S→T)
        neg_logits = logit_scale * img_feats @ neg_embed_cpu.t()
        neg_acc, rank1, mean_rank, neg_fix = compute_metrics(neg_logits, is_neg=True)
        combined = pos_logits - neg_logits
        comb_acc, _, _, comb_fix = compute_metrics(combined, is_neg=False)

        # alpha sweep
        alpha_learned, comb_best = sweep_alpha_neg(pos_logits, neg_logits, labels)

        # 文本编码 neg 的 alpha sweep（取 other_than）
        txt_nlogits = logit_scale * img_feats @ neg_text_cpu['other_than'].t()
        alpha_txt, comb_best_txt = sweep_alpha_neg(pos_logits, txt_nlogits, labels)

        # ===== 诊断：pos vs neg 量级与互补性 =====
        logger.info(f"--- 诊断 T={t_name} ---")
        logger.info(f"pos_logits: mean={pos_logits.mean():.3f} std={pos_logits.std():.3f} "
                    f"min={pos_logits.min():.3f} max={pos_logits.max():.3f}")
        logger.info(f"neg_logits: mean={neg_logits.mean():.3f} std={neg_logits.std():.3f} "
                    f"min={neg_logits.min():.3f} max={neg_logits.max():.3f}")

        # pos: 真类在降序中的 rank（rank0=最高分=最好）
        pos_rank_of_true = (pos_logits.argsort(descending=True) == labels.unsqueeze(1)).nonzero()[:, 1].float()
        # neg: 真类在升序中的 rank（rank0=最低分=最好）
        neg_rank_of_true = (neg_logits.argsort(dim=1) == labels.unsqueeze(1)).nonzero()[:, 1].float()
        logger.info(f"pos true class rank (降序, rank0=最高): mean={pos_rank_of_true.mean():.2f} "
                    f"median={pos_rank_of_true.median():.2f}")
        logger.info(f"neg true class rank (升序, rank0=最低): mean={neg_rank_of_true.mean():.2f} "
                    f"median={neg_rank_of_true.median():.2f}")

        # 互补性核心：pos 错的样本里，neg 有没有把真类排在远离倒数的位置？
        pos_pred = pos_logits.argmax(1)
        neg_pred = neg_logits.argmin(1)
        pos_correct = (pos_pred == labels)
        pos_wrong = ~pos_correct
        neg_excludes_correct = (neg_pred != labels)  # neg 正确 = 没把真类排到最低

        logger.info(f"pos_acc={pos_correct.float().mean()*100:.1f}%")
        logger.info(f"neg排除真类准确率(neg_pred != label)={neg_excludes_correct.float().mean()*100:.1f}%")

        if pos_wrong.sum() > 0:
            neg_rank_on_wrong = (neg_logits[pos_wrong].argsort(dim=1) == labels[pos_wrong].unsqueeze(1)).nonzero()[:, 1].float()
            logger.info(f"pos错误样本({pos_wrong.sum().item()}个)上 neg true rank(升序): "
                        f"mean={neg_rank_on_wrong.mean():.2f} median={neg_rank_on_wrong.median():.2f} "
                        f"(越大越好, 远离倒数)")

        # 交叉检查：pos 和 neg 同时错误的样本
        both_wrong = pos_wrong & (neg_pred == labels)
        logger.info(f"pos错且neg也错(neg把真类排到最低)={both_wrong.float().sum().item()}个 "
                    f"({both_wrong.float().mean()*100:.1f}%)")

        # ===== 排序相关性：neg 是否提供独立于 pos 的信息？ =====
        from scipy.stats import spearmanr
        pos_np = pos_logits.cpu().numpy()
        neg_np = neg_logits.cpu().numpy()
        spearman_per_sample = []
        for i in range(pos_np.shape[0]):
            r, _ = spearmanr(pos_np[i], neg_np[i])
            spearman_per_sample.append(r)
        spearman_per_sample = np.array(spearman_per_sample)
        logger.info(f"pos-neg Spearman 相关系数: mean={spearman_per_sample.mean():.4f} "
                    f"median={np.median(spearman_per_sample):.4f} "
                    f"std={spearman_per_sample.std():.4f}")
        # 在 pos 错误样本上的相关性
        if pos_wrong.sum() > 0:
            sp_wrong = spearman_per_sample[pos_wrong.cpu().numpy()]
            logger.info(f"  pos错误样本上: mean={sp_wrong.mean():.4f} median={np.median(sp_wrong):.4f} "
                        f"(越低越独立)")

        # ===== 动态 alpha 可行性分析：alpha 能否区分 neg 有帮助/有害的样本？ =====
        pos_right = pos_correct
        logger.info(f"--- 动态 alpha 分析 T={t_name} ---")
        logger.info(f"{'alpha':>6s} {'wrong→acc':>10s} {'right→acc':>10s} {'all→acc':>9s} {'gap':>8s}")
        logger.info("-" * 50)
        best_gap = -999
        best_alpha = 0.0
        for alpha_val in torch.linspace(0, 2.0, 21):
            combined = pos_logits - alpha_val.item() * neg_logits
            acc_wrong = (combined[pos_wrong].argmax(1) == labels[pos_wrong]).float().mean().item() * 100
            acc_right = (combined[pos_right].argmax(1) == labels[pos_right]).float().mean().item() * 100
            acc_all = (combined.argmax(1) == labels).float().mean().item() * 100
            gap = acc_wrong - acc_right
            logger.info(f"{alpha_val.item():6.2f} {acc_wrong:9.1f}% {acc_right:9.1f}% {acc_all:8.1f}% {gap:+7.1f}%")
            if gap > best_gap:
                best_gap = gap
                best_alpha = alpha_val.item()
        logger.info(f"→ 固定最优: alpha={best_alpha:.2f} (wrong-right gap={best_gap:+.1f}%)")

        # 置信度驱动动态 alpha：低置信 → alpha 大（neg 介入），高置信 → alpha 小（pos 独断）
        pos_conf = pos_logits.softmax(1).max(1).values
        dyn_alpha = ((1.0 - pos_conf) * 2.0).clamp(0, 1.5)  # (B,)
        combined_dyn = pos_logits - dyn_alpha.unsqueeze(1) * neg_logits
        acc_dyn = (combined_dyn.argmax(1) == labels).float().mean().item() * 100
        acc_dyn_wrong = (combined_dyn[pos_wrong].argmax(1) == labels[pos_wrong]).float().mean().item() * 100
        acc_dyn_right = (combined_dyn[pos_right].argmax(1) == labels[pos_right]).float().mean().item() * 100
        gap_dyn = acc_dyn_wrong - acc_dyn_right
        logger.info(f"→ 置信度动态 alpha: wrong={acc_dyn_wrong:.1f}% right={acc_dyn_right:.1f}% "
                    f"all={acc_dyn:.1f}% gap={gap_dyn:+.1f}%  "
                    f"α∈[{dyn_alpha.min():.2f},{dyn_alpha.max():.2f}] mean={dyn_alpha.mean():.2f}")
        logger.info("-" * 60)

        all_results[t_name] = {
            'N': img_feats.shape[0], 'pos_acc': pos_acc,
            'text': text_metrics,
            'learned': {
                'neg_acc': neg_acc, 'comb_acc': comb_acc,
                'rank1': rank1, 'mean_rank': mean_rank,
                'neg_fix': neg_fix, 'comb_fix': comb_fix,
                'alpha': alpha_learned, 'comb_best': comb_best,
            },
            'txt_alpha': alpha_txt, 'txt_comb_best': comb_best_txt,
        }

    # ===== 输出总表 =====
    logger.info("")
    logger.info("=" * 110)
    logger.info(f"S = {s_name}  |  C = {C}  |  arch = {arch}  |  train={do_train}")
    logger.info("")

    # 表1: 核心指标
    hdr1 = (f"{'T':>12s} | {'PosAcc':>7s} | "
            f"{'NegAcc':>7s} {'CombAcc':>7s} {'Rank1':>7s} {'NFix%':>7s} {'CFix%':>7s} | "
            f"{'txt-Neg':>7s} {'txt-Cmb':>7s}")
    sep1 = "-" * len(hdr1.expandtabs())
    logger.info("--- Core Metrics ---")
    logger.info(hdr1)
    logger.info(sep1)

    for t_name, r in all_results.items():
        is_src = "(S)" if t_name == s_name else ""
        L = r['learned']
        txt = r['text']['other_than']
        logger.info(
            f"{t_name+' '+is_src:>12s} | {r['pos_acc']:6.2f}% | "
            f"{L['neg_acc']:6.2f}% {L['comb_acc']:6.2f}% "
            f"{L['rank1']:6.2f}% {L['neg_fix']:6.1f}% {L['comb_fix']:6.1f}% | "
            f"{txt['neg_acc']:6.2f}% {txt['comb_acc']:6.2f}%"
        )

    cross_t = [r for t_name, r in all_results.items() if t_name != s_name]
    if cross_t:
        avg = lambda key: np.mean([r['learned'][key] for r in cross_t])
        avg_txt = lambda key: np.mean([r['text']['other_than'][key] for r in cross_t])
        logger.info(sep1)
        logger.info(
            f"{'AVG (cross)':>12s} | {np.mean([r['pos_acc'] for r in cross_t]):6.2f}% | "
            f"{avg('neg_acc'):6.2f}% {avg('comb_acc'):6.2f}% "
            f"{avg('rank1'):6.2f}% {avg('neg_fix'):6.1f}% {avg('comb_fix'):6.1f}% | "
            f"{avg_txt('neg_acc'):6.2f}% {avg_txt('comb_acc'):6.2f}%"
        )

    # 表2: alpha sweep
    logger.info("")
    logger.info("--- Alpha Sweep: combined = pos - alpha * neg ---")
    hdr2 = (f"{'T':>12s} | {'PosAcc':>7s} | "
            f"{'learned α':>10s} {'Comb*':>7s} | "
            f"{'txt α':>10s} {'txt Comb*':>7s}")
    sep2 = "-" * len(hdr2.expandtabs())
    logger.info(hdr2)
    logger.info(sep2)

    for t_name, r in all_results.items():
        is_src = "(S)" if t_name == s_name else ""
        L = r['learned']
        logger.info(
            f"{t_name+' '+is_src:>12s} | {r['pos_acc']:6.2f}% | "
            f"{L['alpha']:9.2f}  {L['comb_best']:6.2f}% | "
            f"{r['txt_alpha']:9.2f}  {r['txt_comb_best']:6.2f}%"
        )

    if cross_t:
        avg_alpha = np.mean([r['learned']['alpha'] for r in cross_t])
        avg_comb_best = np.mean([r['learned']['comb_best'] for r in cross_t])
        avg_txt_alpha = np.mean([r['txt_alpha'] for r in cross_t])
        avg_txt_best = np.mean([r['txt_comb_best'] for r in cross_t])
        pos_mean = np.mean([r['pos_acc'] for r in cross_t])
        logger.info(sep2)
        logger.info(
            f"{'AVG (cross)':>12s} | {pos_mean:6.2f}% | "
            f"{avg_alpha:9.2f}  {avg_comb_best:6.2f}% | "
            f"{avg_txt_alpha:9.2f}  {avg_txt_best:6.2f}%"
        )

    logger.info("=" * 110)
    logger.info(f"Model: {save_path}")

    return {'s_name': s_name, 'all_results': all_results, 'save_path': save_path}


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

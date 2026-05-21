#prode_diff.py
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from src.utils import loss,prompt_tuning,IID_losses
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import  ImageList_idx, ImageList_idx_aug_fix_old
from sklearn.metrics import confusion_matrix
from clip.custom_clip import get_coop
import clip
from src.utils.utils import *

logger = logging.getLogger(__name__)



def data_load(cfg):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
    if not cfg.DA == 'uda':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.tar_classes:
                if int(reci[1]) in cfg.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    dsets["target"] = ImageList_idx_aug_fix_old(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx_aug_fix_old(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source"] = ImageList_idx_aug_fix_old(txt_test, transform=image_test())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=2, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True)
    return dset_loaders

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
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def test_time_tuning(model, inputs, optimizer, cfg, target_output):
    target_output = target_output.cuda()
    target_output = nn.Softmax(dim=1)(target_output)

    for j in range(cfg.ProDe.TTA_STEPS):
        with torch.amp.autocast('cuda'):
            output_logits, _ = model(inputs)
            output = nn.Softmax(dim=1)(output_logits)
            iic_loss = IID_losses.IID_loss_old(output, target_output)
        optimizer.zero_grad()
        iic_loss.backward()
        optimizer.step()

    return output

def test_time_adapt_eval(input, model, optimizer, optim_state, cfg, target_output):
    optimizer.load_state_dict(optim_state)
    output = test_time_tuning(model, input, optimizer, cfg, target_output)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            model.eval()
            output,_ = model(input)
    return output

@torch.no_grad()
@torch.no_grad()
def eval_neg_metrics(dset_loaders, model, neg_embed, neg_alpha):
    """评估负向嵌入在测试集上的表现"""
    all_pos_logits = []
    all_neg_logits = []
    all_labels = []

    for data in dset_loaders["test"]:
        inputs = data[0][0].cuda()
        labels = data[1]

        img_feats = model.image_encoder(inputs.type(model.dtype))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        text_feats = model.get_text_features()

        pos_logits = model.logit_scale.exp() * img_feats @ text_feats.t()
        neg_logits = model.logit_scale.exp() * img_feats @ neg_embed.t()

        all_pos_logits.append(pos_logits.cpu())
        all_neg_logits.append(neg_logits.cpu())
        all_labels.append(labels)

    pos_logits = torch.cat(all_pos_logits)
    neg_logits = torch.cat(all_neg_logits)
    labels = torch.cat(all_labels)
    C = neg_logits.shape[1]

    pos_pred = pos_logits.argmax(1)
    pos_acc = (pos_pred == labels).float().mean().item() * 100
    pos_wrong = ~(pos_pred == labels)

    neg_pred = neg_logits.argmin(1)
    neg_acc = (neg_pred == labels).float().mean().item() * 100

    combined = pos_logits - neg_alpha * neg_logits
    comb_pred = combined.argmax(1)
    comb_acc = (comb_pred == labels).float().mean().item() * 100

    ranks = neg_logits.argsort(dim=1).argsort(dim=1)
    true_ranks = ranks.gather(1, labels.unsqueeze(1)).float()
    rank1 = (true_ranks == 0).float().mean().item() * 100
    mean_rank = true_ranks.mean().item()

    nfix = (pos_wrong & (neg_pred == labels)).float().sum().item() / max(pos_wrong.sum().item(), 1) * 100
    cfix = (pos_wrong & (comb_pred == labels)).float().sum().item() / max(pos_wrong.sum().item(), 1) * 100

    return {'pos_acc': pos_acc, 'neg_acc': neg_acc, 'comb_acc': comb_acc,
            'rank1': rank1, 'mean_rank': mean_rank, 'neg_fix': nfix, 'comb_fix': cfix}


def analyze_ensemble(cfg, netF, netB, netC, model, dset_loaders):
    text_feats = model.get_text_features()
    logit_scale = model.logit_scale.exp()

    all_src_logits = []
    all_clip_logits = []
    all_labels = []

    for data in dset_loaders["test"]:
        inputs = data[0][0].cuda()
        labels = data[1]
        indx = data[-1]

        src_logits = netC(netB(netF(inputs))).cpu()

        img_feats = model.image_encoder(inputs.type(model.dtype))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        clip_logits = (logit_scale * img_feats @ text_feats.t()).cpu()

        all_src_logits.append(src_logits)
        all_clip_logits.append(clip_logits)
        all_labels.append(labels)

    src_logits = torch.cat(all_src_logits)
    clip_logits = torch.cat(all_clip_logits)
    labels = torch.cat(all_labels)

    src_pred = src_logits.argmax(1)
    clip_pred = clip_logits.argmax(1)
    ensemble_pred = (src_logits + clip_logits).argmax(1)

    src_acc = (src_pred == labels).float().mean().item() * 100
    clip_acc = (clip_pred == labels).float().mean().item() * 100
    ens_acc = (ensemble_pred == labels).float().mean().item() * 100

    correct_src = (src_pred == labels)
    correct_clip = (clip_pred == labels)
    correct_ens = (ensemble_pred == labels)

    both_right = correct_clip & correct_src
    clip_only = correct_clip & ~correct_src
    src_only = ~correct_clip & correct_src
    both_wrong = ~correct_clip & ~correct_src

    # ---- Normalization strategies for scale alignment ----
    s_mean = src_logits.mean(1, keepdim=True)
    s_std = src_logits.std(1, keepdim=True)
    c_mean = clip_logits.mean(1, keepdim=True)
    c_std = clip_logits.std(1, keepdim=True)
    s_min = src_logits.min(1, keepdim=True).values
    s_max = src_logits.max(1, keepdim=True).values
    c_min = clip_logits.min(1, keepdim=True).values
    c_max = clip_logits.max(1, keepdim=True).values
    logger.info(f"Scale stats: src mean={s_mean.mean().item():.1f} std={s_std.mean().item():.1f} | "
                f"clip mean={c_mean.mean().item():.1f} std={c_std.mean().item():.1f}")
    logger.info("Ensemble normalization comparison:")
    for name, ens in [
        ("raw", src_logits + clip_logits),
        ("z-score", (src_logits - s_mean) / (s_std + 1e-6) + (clip_logits - c_mean) / (c_std + 1e-6)),
        ("minmax", (src_logits - s_min) / (s_max - s_min + 1e-6) + (clip_logits - c_min) / (c_max - c_min + 1e-6)),
        ("prob", torch.softmax(src_logits, dim=1) + torch.softmax(clip_logits, dim=1)),
    ]:
        acc = (ens.argmax(1) == labels).float().mean().item() * 100
        logger.info(f"  ens_{name}: {acc:.2f}%")

    n = len(labels)
    bw = both_wrong.sum().item()
    logger.info("Ensemble Analysis "
                f"src={src_acc:.1f}% clip={clip_acc:.1f}% ens={ens_acc:.1f}% | "
                f"both_right={both_right.sum().item()} clip_only={clip_only.sum().item()} "
                f"src_only={src_only.sum().item()} both_wrong={bw}")

    if bw > 0:
        bw_same = (src_pred[both_wrong] == clip_pred[both_wrong])
        logger.info(f"both_wrong={bw}: same_error={bw_same.sum().item()}/{bw} "
                    f"({bw_same.float().mean().item()*100:.1f}%)")

        bw_ens_fixed = (both_wrong & correct_ens).sum().item()
        logger.info(f"Ensemble fixes {bw_ens_fixed}/{bw} both_wrong "
                    f"({bw_ens_fixed/bw*100:.1f}%) | "
                    f"side_effect: clip_only_broken={(clip_only & ~correct_ens).sum().item()}, "
                    f"src_only_broken={(src_only & ~correct_ens).sum().item()}")

        # Per-class both_wrong count (top 5)
        bw_labels = labels[both_wrong]
        class_counts = torch.bincount(bw_labels, minlength=cfg.class_num)
        top5 = class_counts.argsort(descending=True)[:5]
        top_str = ' | '.join(f"c{i}({class_counts[i].item()})" for i in top5 if class_counts[i] > 0)
        logger.info(f"both_wrong top classes: {top_str}")

        # Top-k analysis on diff_error (both wrong, different mistakes)
        bw_diff = both_wrong & (src_pred != clip_pred)
        nd = bw_diff.sum().item()
        if nd > 0:
            bw_diff_src_logits = src_logits[bw_diff]
            bw_diff_clip_logits = clip_logits[bw_diff]
            bw_diff_labels = labels[bw_diff]
            for k in [2, 3, 5]:
                src_topk = bw_diff_src_logits.topk(k, dim=1).indices
                clip_topk = bw_diff_clip_logits.topk(k, dim=1).indices
                gt = bw_diff_labels.unsqueeze(1)
                src_hit = (src_topk == gt).any(1).float().mean().item() * 100
                clip_hit = (clip_topk == gt).any(1).float().mean().item() * 100
                union = ((src_topk == gt).any(1) | (clip_topk == gt).any(1)).float().mean().item() * 100
                logger.info(f"diff_error({nd}): top-{k} src={src_hit:.1f}% clip={clip_hit:.1f}% union={union:.1f}%")

        # ---- Oracle experiments ----
        # 1. Oracle: pick the model that is correct (upper bound for reliability estimation)
        oracle_pred = torch.where(correct_clip, clip_pred, src_pred)
        oracle_acc = (oracle_pred == labels).float().mean().item() * 100
        logger.info(f"Oracle (pick correct model): {oracle_acc:.2f}%")

        # 2. Oracle: per-sample, pick which model to trust (clip or src)
        n_samples = len(labels)
        n_clip_right_src_wrong = (~correct_src & correct_clip).sum().item()
        n_src_right_clip_wrong = (correct_src & ~correct_clip).sum().item()
        logger.info(f"Reliability oracle: clip_right_src_wrong={n_clip_right_src_wrong} "
                    f"src_right_clip_wrong={n_src_right_clip_wrong} "
                    f"gain_if_perfect={n_clip_right_src_wrong + n_src_right_clip_wrong} "
                    f"({(n_clip_right_src_wrong + n_src_right_clip_wrong)/n_samples*100:.1f}%)")

        # 3. Oracle top-k union on both_wrong (can re-ranking in top-k union solve it?)
        if bw > 0:
            src_top5 = src_logits[both_wrong].topk(5, dim=1).indices
            clip_top5 = clip_logits[both_wrong].topk(5, dim=1).indices
            bw_labels = labels[both_wrong]
            hits = []
            for i in range(bw):
                union_set = set(src_top5[i].tolist()) | set(clip_top5[i].tolist())
                hits.append(bw_labels[i].item() in union_set)
            union_top5_hit = torch.tensor(hits).float().mean().item() * 100
            logger.info(f"both_wrong({bw}): union top-5 contains true = {union_top5_hit:.1f}% "
                        f"({int(union_top5_hit/100*bw)}/{bw})")

            # Among diff_error specifically
            if nd > 0:
                src_top5_d = src_logits[bw_diff].topk(5, dim=1).indices
                clip_top5_d = clip_logits[bw_diff].topk(5, dim=1).indices
                bw_diff_labels = labels[bw_diff]
                hits_d = []
                for i in range(nd):
                    union_set = set(src_top5_d[i].tolist()) | set(clip_top5_d[i].tolist())
                    hits_d.append(bw_diff_labels[i].item() in union_set)
                union_top5_d_hit = torch.tensor(hits_d).float().mean().item() * 100
                logger.info(f"diff_error({nd}): union top-5 contains true = {union_top5_d_hit:.1f}% "
                            f"({int(union_top5_d_hit/100*nd)}/{nd})")

        # ---- Confidence-based routing (margin, entropy, agreement) ----
        src_prob = torch.softmax(src_logits, dim=1)
        clip_prob = torch.softmax(clip_logits, dim=1)
        src_top2 = src_logits.topk(2, dim=1).values
        clip_top2 = clip_logits.topk(2, dim=1).values
        margin_src = src_top2[:, 0] - src_top2[:, 1]
        margin_clip = clip_top2[:, 0] - clip_top2[:, 1]
        ent_src = -(src_prob * torch.log(src_prob + 1e-6)).sum(1)
        ent_clip = -(clip_prob * torch.log(clip_prob + 1e-6)).sum(1)
        agree = (src_pred == clip_pred).float()

        # Router 1: margin-based
        w_clip_m = margin_clip / (margin_clip + margin_src + 1e-6)
        routed_m = w_clip_m.unsqueeze(1) * clip_logits + (1 - w_clip_m).unsqueeze(1) * src_logits
        routem_acc = (routed_m.argmax(1) == labels).float().mean().item() * 100

        # Router 2: entropy-based (lower entropy = more confident)
        w_clip_e = (1 / (ent_clip + 1e-6)) / ((1 / (ent_src + 1e-6)) + (1 / (ent_clip + 1e-6)) + 1e-6)
        routed_e = w_clip_e.unsqueeze(1) * clip_logits + (1 - w_clip_e).unsqueeze(1) * src_logits
        routee_acc = (routed_e.argmax(1) == labels).float().mean().item() * 100

        # Router 3: agreement-gated (disagree -> pick more confident)
        more_confident = (margin_clip > margin_src).float()
        routed_a = torch.where(agree.bool().unsqueeze(1),
                               (src_logits + clip_logits) / 2,
                               torch.where((more_confident == 1).unsqueeze(1), clip_logits, src_logits))
        routea_acc = (routed_a.argmax(1) == labels).float().mean().item() * 100

        # Router 4: agreement-gated with entropy
        more_confident_e = (ent_clip < ent_src).float()
        routed_ae = torch.where(agree.bool().unsqueeze(1),
                                (src_logits + clip_logits) / 2,
                                torch.where((more_confident_e == 1).unsqueeze(1), clip_logits, src_logits))
        routeae_acc = (routed_ae.argmax(1) == labels).float().mean().item() * 100

        logger.info(f"Routing: margin={routem_acc:.1f}% entropy={routee_acc:.1f}% "
                    f"agree+margin={routea_acc:.1f}% agree+ent={routeae_acc:.1f}% "
                    f"vs ens={ens_acc:.1f}% oracle={oracle_acc:.1f}%")

        logger.info(f"\n=== Summary: src={src_acc:.1f}% clip={clip_acc:.1f}% ens={ens_acc:.1f}% "
                    f"oracle={oracle_acc:.1f}% | both_wrong={bw} diff_error={bw_diff.sum().item() if bw>0 else 0} ===\n")


def train_target(cfg):
    text_inputs = clip_pre_text(cfg)
    dset_loaders = data_load(cfg)
    model = get_coop(cfg.ProDe.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.ProDe.N_CTX, cfg.ProDe.CTX_INIT)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    modelpath = cfg.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    # --- 加载 learned neg_embed（PositiveCoOp 风格，显式 CE 训练） ---
    neg_embed = None
    opt_neg = None
    if cfg.ProDe.USE_LEARNED_NEG:
        neg_path = osp.join(cfg.output_dir_src, f"learned_neg_{cfg.domain[cfg.SETTING.S]}.pt")
        if osp.exists(neg_path):
            ckpt = torch.load(neg_path, map_location='cpu')
            neg_embed = nn.Parameter(ckpt['neg_embed'].cuda())
            logger.info(f"Loaded learned neg_embed: {neg_path}  shape={neg_embed.shape}")
        else:
            logger.info(f"Learned neg_embed not found: {neg_path}, skipping")

    # --- netC_proxy: N+N classifier for IID loss ---
    netC_proxy = network.feat_classifier(type='wn', class_num=cfg.class_num * 2, bottleneck_dim=cfg.bottleneck).cuda()
    with torch.no_grad():
        netC_proxy.fc.weight_g.data[:cfg.class_num].copy_(netC.fc.weight_g.data)
        netC_proxy.fc.weight_v.data[:cfg.class_num].copy_(netC.fc.weight_v.data)
        netC_proxy.fc.bias.data[:cfg.class_num].copy_(netC.fc.bias.data)
        torch.nn.init.kaiming_uniform_(netC_proxy.fc.weight_g.data[cfg.class_num:], a=0, mode='fan_in', nonlinearity='linear')
        netC_proxy.fc.bias.data[cfg.class_num:].fill_(-3.0)

    # --- Text-guided semantic similarity（已由 learned neg_embed 替代） ---
    model.reset_classnames(cfg.classname, cfg.ProDe.ARCH)

    param_group = []
    param_group_ib = []

    for k, v in netF.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if cfg.OPTIM.LR_DECAY2 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2}]
        else:
            v.requires_grad = False
    for k, v in model.prompt_learner.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    optimizer_ib = optim.SGD(param_group_ib)
    optimizer_ib = op_copy(optimizer_ib)
    opt_proxyC = optim.SGD([{'params': netC_proxy.parameters(), 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}])
    opt_proxyC = op_copy(opt_proxyC)
    if neg_embed is not None:
        opt_neg = optim.SGD([{'params': neg_embed, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}])
        opt_neg = op_copy(opt_neg)
    optim_state = deepcopy(optimizer_ib.state_dict())
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    num_sample=len(dset_loaders["target"].dataset)
    logtis_bank = torch.randn(num_sample, cfg.class_num).cuda()

    loader = dset_loaders["source"]
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
            indx=data[-1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            outputs = netC(output)
            outputs=nn.Softmax(dim=1)(outputs)
            logtis_bank[indx] = outputs.detach().clone()


    while iter_num < max_iter:
        try:
            (inputs_test, inputs_test_augs), _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, inputs_test_augs), _, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        inputs_test_augs = inputs_test_augs.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        with torch.no_grad():
            outputs_test_new = outputs_test.clone().detach()

        inputs_test_clip = inputs_test_augs

        clip_score = test_time_adapt_eval(inputs_test_clip, model, optimizer_ib, optim_state, cfg,
                                           outputs_test_new)
        clip_score = clip_score.float()

        # 负向嵌入：显式 CE 训练（仿 pretrain），然后组合
        if neg_embed is not None:
            img_feats = model.image_encoder(inputs_test_clip.type(model.dtype))
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            neg_logits = model.logit_scale.exp() * img_feats @ neg_embed.t()

            term_a = outputs_test_new - logtis_bank[tar_idx].cuda()
            term_b = clip_score.cuda()
            new_clip = term_a + term_b - cfg.ProDe.NEG_ALPHA * neg_logits

            # 显式训练 neg_embed：真类的 neg_logits 应最小
            with torch.no_grad():
                pseudo_labels = new_clip.argmax(1)
            loss_neg = nn.CrossEntropyLoss()(-neg_logits, pseudo_labels)
            opt_neg.zero_grad()
            loss_neg.backward()
            opt_neg.step()
        else:
            term_a = outputs_test_new - logtis_bank[tar_idx].cuda()
            term_b = clip_score.cuda()
            new_clip = term_a + term_b

        clip_score_sm = nn.Softmax(dim=1)(new_clip)
        _, clip_index_new = torch.max(new_clip, 1)

        if iter_num % 50 == 1:
            neg_str = f' neg_loss={loss_neg.item():.4f}' if (neg_embed is not None) else ''
            logger.info(
                f'Iter {iter_num}: term_a(mean/std)={term_a.mean().item():.2f}/{term_a.std().item():.2f}'
                f' term_b={term_b.mean().item():.2f}/{term_b.std().item():.2f}{neg_str}')

        # N+N IID loss with temperature-controlled proxy
        outputs_proxy2N = netC_proxy(features_test)
        softmax_out_2N = nn.Softmax(dim=1)(outputs_proxy2N)
        # with torch.no_grad():
        #     T_proxy = cfg.ProDe.PROXY_WEIGHT + 1.0  # ~2.0, larger = less proxy influence
        #     clip_logits_2N = torch.cat([new_clip, residual], dim=1)
        #     clip_softmax_2N = nn.Softmax(dim=1)(clip_logits_2N)
        # iic_loss, _ = IID_losses.IID_loss(cfg.class_num, softmax_out_2N, clip_softmax_2N,
        #                                   w_base=1.0, w_aux=0.3, bn=-0.3, lamb=1.0)
        iic_loss = IID_losses.IID_loss_old(softmax_out,clip_score_sm)
        classifier_loss = cfg.ProDe.IIC_PAR * iic_loss
        msoftmax = softmax_out.mean(dim=0)

        if  cfg.SETTING.DATASET=='office':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if  cfg.SETTING.DATASET=='office-home':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if  cfg.SETTING.DATASET=='VISDA-C':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 0.1 * gentropy_loss
        pred = clip_index_new
        entropy_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        classifier_loss =  cfg.ProDe.GENT_PAR*entropy_loss + classifier_loss

        optimizer.zero_grad()
        optimizer_ib.zero_grad()
        opt_proxyC.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        optimizer_ib.step()
        opt_proxyC.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            analyze_ensemble(cfg, netF, netB, netC, model, dset_loaders)
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te, classifier_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te, classifier_loss)

            # 负向嵌入评估
            if neg_embed is not None:
                nm = eval_neg_metrics(dset_loaders, model, neg_embed, cfg.ProDe.NEG_ALPHA)
                log_str += '\n  NegEval: PosAcc={pos_acc:.1f}% NegAcc={neg_acc:.1f}% CombAcc={comb_acc:.1f}% Rank1={rank1:.1f}% meanR={mean_rank:.1f} NFix={neg_fix:.1f}% CFix={comb_fix:.1f}%'.format(**nm)

            logger.info(log_str)
            netF.train()
            netB.train()
            netC.train()

    if cfg.ISSAVE:
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))

    return netF, netB, netC


def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.ProDe.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.utils import loss, IID_losses
import pandas as pd
import time  # ← 新增

from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList_idx
from sklearn.metrics import confusion_matrix
import clip
from src.utils.utils import *
from src.utils.prompt_tuning_diff_new_optim_1130 import *

logger = logging.getLogger(__name__)

import wandb
import copy


# ============================================================
# GPU 显存 & 时间监控工具
# ============================================================
class GPUMonitor:
    """
    轻量级 GPU 显存 + 耗时监控器。
    用法：
        monitor = GPUMonitor(device=0)
        monitor.reset_peak()          # 重置峰值
        t = monitor.start_timer()     # 开始计时
        ...
        elapsed = monitor.stop_timer(t)  # 停止计时，返回秒数
        monitor.log_memory(tag="训练迭代")
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.enabled = torch.cuda.is_available()
        if self.enabled:
            torch.cuda.init()   # 确保 CUDA 上下文已建立
            self.reset_peak()

    def reset_peak(self):
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def current_mb(self) -> float:
        if not self.enabled:
            return 0.0
        return torch.cuda.memory_allocated(self.device) / 1024 ** 2

    def peak_mb(self) -> float:
        if not self.enabled:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / 1024 ** 2

    def reserved_mb(self) -> float:
        """PyTorch 向驱动申请的总显存（含碎片缓存）"""
        if not self.enabled:
            return 0.0
        return torch.cuda.memory_reserved(self.device) / 1024 ** 2

    def log_memory(self, tag: str = ""):
        if not self.enabled:
            return
        cur = self.current_mb()
        peak = self.peak_mb()
        rsv = self.reserved_mb()
        msg = (
            f"[显存] {tag} | "
            f"当前已分配: {cur:.1f} MB | "
            f"峰值已分配: {peak:.1f} MB | "
            f"已向驱动申请: {rsv:.1f} MB"
        )
        logger.info(msg)
        print(msg)

    @staticmethod
    def start_timer() -> float:
        """返回当前时间戳（秒）"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # 等待 GPU 完成，保证计时准确
        return time.perf_counter()

    @staticmethod
    def stop_timer(start: float) -> float:
        """返回从 start 到现在经过的秒数"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - start

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s"
# ============================================================


def copy_netC(netC, netC_aux, num_known_classes):
    state_dict_C = netC.state_dict()
    state_dict_C_aux = netC_aux.state_dict()

    for key in state_dict_C.keys():
        if key in state_dict_C_aux:
            if state_dict_C[key].shape[0] == num_known_classes and state_dict_C_aux[key].shape[0] > num_known_classes:
                state_dict_C_aux[key][:num_known_classes] = state_dict_C[key].clone()
            elif state_dict_C[key].shape[0] == num_known_classes and state_dict_C_aux[key].shape[0] == num_known_classes:
                state_dict_C_aux[key] = state_dict_C[key].clone()

    netC_aux.load_state_dict(state_dict_C_aux, strict=False)


def copy_back_netC(netC, netC_aux, num_known_classes):
    state_dict_C = netC.state_dict()
    state_dict_C_aux = netC_aux.state_dict()

    for key in state_dict_C.keys():
        if key in state_dict_C_aux:
            if state_dict_C[key].shape[0] == num_known_classes:
                if state_dict_C_aux[key].shape[0] >= num_known_classes:
                    state_dict_C[key] = state_dict_C_aux[key][:num_known_classes].clone()
                else:
                    state_dict_C[key] = state_dict_C_aux[key].clone()

    netC.load_state_dict(state_dict_C, strict=False)


def print_grad_hook(grad):
    if grad.norm().item() > 100:
        print('Hook: Grad recieved with norm {}'.format(grad.norm().item()))


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def image_test_50(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=[0.26862954, 0.26130258, 0.27577711])
    return transforms.Compose([
        transforms.Resize(crop_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(cfg, preprocess):
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
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["clip_test"] = ImageList_idx(txt_test, transform=preprocess)
    dset_loaders["clip_test"] = DataLoader(dsets["clip_test"], batch_size=train_bs * 3, shuffle=False,
                                           num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return transforms.Compose([
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


def lr_scheduler2(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.5
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, netF, netB, netC, cfg, flag=False, monitor: GPUMonitor = None):
    """
    新增 monitor 参数（可选）。
    如果传入，会在推理结束后打印推理耗时和显存状况。
    """
    start_test = True

    # ── 推理计时开始 ──────────────────────────────────────────
    if monitor:
        t_infer_start = GPUMonitor.start_timer()
    # ─────────────────────────────────────────────────────────

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            outputs = outputs[:, :cfg.class_num]
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # ── 推理计时结束 ──────────────────────────────────────────
    if monitor:
        infer_time = GPUMonitor.stop_timer(t_infer_start)
        n_samples = all_label.size(0)
        msg = (
            f"[推理耗时] 共 {n_samples} 张图 | "
            f"总耗时: {GPUMonitor.format_time(infer_time)} | "
            f"均值: {infer_time / n_samples * 1000:.2f} ms/张"
        )
        logger.info(msg)
        print(msg)
        monitor.log_memory(tag="推理结束")
    # ─────────────────────────────────────────────────────────

    num_known_classes = cfg.class_num
    all_output_known = all_output[:, :num_known_classes]
    _, predict = torch.max(all_output_known, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output_known))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def cal_satistical_info(loader, netF, netB, netC, clip_softmax_score, cfg, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
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

    clip_softmax_score_cpu = clip_softmax_score.cpu()
    base_softmax_score = nn.Softmax(dim=1)(all_output)
    base_softmax_score_known = base_softmax_score[:, :cfg.class_num]
    clip_softmax_known = clip_softmax_score_cpu[:, :cfg.class_num]

    _, base_predict = torch.max(base_softmax_score_known, 1)
    _, clip_predict = torch.max(clip_softmax_known, 1)
    base_sign = (torch.squeeze(base_predict).float() == all_label)
    clip_sign = (torch.squeeze(clip_predict).float() == all_label)
    base_corr_clip_wrong = (base_sign & (~clip_sign))
    base_wrong_clip_corr = ((~base_sign) & clip_sign)
    data_size = all_label.size()[0]

    base_corr_clip_wrong_num = torch.sum(base_corr_clip_wrong).item()
    base_wrong_clip_corr_num = torch.sum(base_wrong_clip_corr).item()
    log_str = 'The base_corr_clip_wrong= {:.2f}%;The base_wrong_clip_corr={:.2f}%'.format(
        100 * base_corr_clip_wrong_num / data_size, 100 * base_wrong_clip_corr_num / data_size)
    logger.info(log_str)

    B_True_C_False_base_soft = base_softmax_score[base_corr_clip_wrong]
    B_True_C_False_clip_soft = clip_softmax_score[base_corr_clip_wrong]
    groud_truth_index_BTCF = all_label[base_corr_clip_wrong]
    CFBT_base_mean, CFBT_base_var = evaluesoftmax_goodpredictiom(B_True_C_False_base_soft)
    CFBT_clip_mean, CFBT_clip_var = evaluesoftmax_badpredictiom(B_True_C_False_clip_soft, groud_truth_index_BTCF)
    log_str = 'The CFBT_base_mean= {}; The CFBT_base_var={};The CFBT_clip_mean={};The CFBT_clip_var={}'.format(
        CFBT_base_mean, CFBT_base_var, CFBT_clip_mean, CFBT_clip_var)
    logger.info(log_str)

    B_False_C_True_base_soft = base_softmax_score[base_wrong_clip_corr]
    B_False_C_True_clip_soft = clip_softmax_score[base_wrong_clip_corr]
    groud_truth_index_BFCT = all_label[base_wrong_clip_corr]
    BFCT_base_mean, BFCT_base_var = evaluesoftmax_goodpredictiom(B_False_C_True_clip_soft)
    BFCT_clip_mean, BFCT_clip_var = evaluesoftmax_badpredictiom(B_False_C_True_base_soft, groud_truth_index_BFCT)
    log_str = 'The BFCT_base_mean= {}; The BFCT_base_var={};The BFCT_clip_mean={};The BFCT_clip_var={}'.format(
        BFCT_base_mean, BFCT_base_var, BFCT_clip_mean, BFCT_clip_var)
    logger.info(log_str)

    base_True_clip_True = (base_sign & clip_sign)
    B_True_C_True_base_soft = base_softmax_score[base_True_clip_True]
    B_True_C_True_clip_soft = clip_softmax_score[base_True_clip_True]
    groud_truth_index_BTCT = all_label[base_True_clip_True]
    BTCT_base_mean, BTCT_base_var = evaluesoftmax_goodpredictiom(B_True_C_True_base_soft)
    BTCT_clip_mean, BTCT_clip_var = evaluesoftmax_goodpredictiom(B_True_C_True_clip_soft)
    log_str = 'The BTCT_base_mean= {}; The BTCT_base_var={};The BTCT_clip_mean={};The BTCT_clip_var={}'.format(
        BTCT_base_mean, BTCT_base_var, BTCT_clip_mean, BTCT_clip_var)
    logger.info(log_str)

    base_False_clip_False = (~base_sign & ~clip_sign)
    B_False_C_False_base_soft = base_softmax_score[base_False_clip_False]
    B_False_C_False_clip_soft = clip_softmax_score[base_False_clip_False]
    ground_truth_index_BFCF = all_label[base_False_clip_False]
    BFCF_base_mean, BFCF_base_var = evaluesoftmax_badpredictiom(B_False_C_False_base_soft, ground_truth_index_BFCF)
    BFCF_clip_mean, BFCF_clip_var = evaluesoftmax_badpredictiom(B_False_C_False_clip_soft, ground_truth_index_BFCF)
    log_str = 'The BFCF_base_mean= {}; The BFCF_base_var={};The BFCF_clip_mean={};The BFCF_clip_var={}'.format(
        BFCF_base_mean, BFCF_base_var, BFCF_clip_mean, BFCF_clip_var)
    logger.info(log_str)


def evaluesoftmax_goodpredictiom(softmaxscore):
    acc, predict = torch.topk(softmaxscore, k=2, dim=1)
    distance = acc[:, 0] - acc[:, 1]
    return torch.mean(distance), torch.var(distance)


def evaluesoftmax_badpredictiom(softmaxscore, labelgroundtruth):
    size = labelgroundtruth.shape[0]
    tmp = torch.zeros(size, dtype=torch.float32, device='cpu')
    for i in range(size):
        tmp[i] = softmaxscore[i, labelgroundtruth[i].int()]
    distance = torch.max(softmaxscore, dim=1)[0].cpu() - tmp
    return torch.mean(distance), torch.var(distance)


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(cfg, beta_pre_1=1, beta_pre_2=0.1, use_iid=True, use_cls=True, use_clip=True, aux_begin=0.5):
    print("using the new opt file")
    print("iid:{},cls:{},clip:{}".format(use_iid, use_cls, use_clip))
    acc = 0
    is_merge_stratg_use = True
    abulations = 0

    beta_pre_1 = beta_pre_1
    beta_pre_2 = beta_pre_2
    aux_begin = aux_begin
    aux_end = 0.1

    print("aux_begin:", aux_begin, "aux_end:", aux_end)
    print("is_merge_stratg_use", is_merge_stratg_use, "beta_pre_1:", beta_pre_1, "beta_pre_2:", beta_pre_2)

    # ── 初始化显存监控器 ─────────────────────────────────────
    monitor = GPUMonitor(device=0)
    monitor.log_memory(tag="训练开始前")
    t_total_start = GPUMonitor.start_timer()   # 整体训练计时
    # ────────────────────────────────────────────────────────

    clip_model, preprocess, _ = clip.load(cfg.DIFF_ALIGN.ARCH)
    clip_model.float()
    text_inputs = clip_pre_text(cfg)
    dset_loaders = data_load(cfg, preprocess)

    print("using the arch:", cfg.MODEL.ARCH, "+++++++++++++++++++++++++++++")
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
    netC_aux = network.feat_classifier(type='wn', class_num=cfg.class_num + cfg.class_num_hidden,
                                       bottleneck_dim=cfg.bottleneck).cuda()

    print("normal")
    modelpath = cfg.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath), strict=True)
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath), strict=True)
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    old_fc = netC.fc
    new_fc = torch.nn.utils.weight_norm(
        nn.Linear(old_fc.in_features, cfg.class_num + cfg.class_num_hidden).cuda())
    with torch.no_grad():
        new_fc.weight_g.data[:cfg.class_num].copy_(old_fc.weight_g.data)
        new_fc.weight_v.data[:cfg.class_num].copy_(old_fc.weight_v.data)
        new_fc.bias.data[:cfg.class_num].copy_(old_fc.bias.data)
        if cfg.class_num_hidden > 0:
            print("Yes,we use hidden classes.")
            torch.nn.init.kaiming_uniform_(new_fc.weight_g.data[cfg.class_num:], a=0, mode='fan_in',
                                           nonlinearity='linear')
            new_fc.bias.data[cfg.class_num:].zero_()
    netC_aux.fc = new_fc
    netC.eval()

    monitor.log_memory(tag="模型加载完成后")  # ← 新增

    param_group = []
    param_group_C = []
    param_group_aux = []

    for k, v in netC.named_parameters():
        v.requires_grad = True
        param_group_C += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1 * 0.3}]
    if cfg.class_num_hidden > 0:
        for k, v in netC_aux.named_parameters():
            v.requires_grad = True
            param_group_aux += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
    else:
        param_group_aux = param_group_C

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

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    opt_aux = optim.SGD(param_group_aux)
    opt_aux = op_copy(opt_aux)
    opt_C = optim.SGD(param_group_C)
    opt_C = op_copy(opt_C)

    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    score_bank = torch.randn(num_sample, cfg.class_num).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    netC_aux.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            indx = data[-1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            outputs = netC(output)
            outputs = nn.Softmax(dim=1)(outputs)
            score_bank[indx] = outputs.detach().clone()

    if cfg.SETTING.DATASET == 'office-home':
        max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"]) * 2
        interval_iter = max_iter // 5 // cfg.TEST.INTERVAL
        copy_iter = max_iter // 15 // cfg.TEST.INTERVAL
    else:
        max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
        interval_iter = max_iter * 7 // cfg.TEST.INTERVAL
        copy_iter = max_iter // 15 // cfg.TEST.INTERVAL

    iter_num = 0
    text_features = None

    mypromt = promote_learning_diff_promote_learning(cfg, clip_model).to(
        clip_model.visual.conv1.weight.device)

    # ── 用于统计平均训练迭代时间 ──────────────────────────────
    train_iter_times = []          # 存放每次迭代的耗时（秒）
    MEM_LOG_INTERVAL = interval_iter  # 每 interval_iter 次迭代打印一次显存
    # ────────────────────────────────────────────────────────

    while iter_num < max_iter:

        current_aux = aux_begin + (aux_end - aux_begin) * (iter_num / max_iter)
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        inputs_test = inputs_test.cuda()

        if inputs_test.size(0) == 1:
            continue

        if cfg.SETTING.DATASET == 'office-home' and iter_num % (copy_iter) == 0:
            copy_netC(netC, netC_aux, cfg.class_num)
        if cfg.SETTING.DATASET == ' VISDA-C' and iter_num % (copy_iter // 8) == 0:
            copy_netC(netC, netC_aux, cfg.class_num)

        if iter_num % interval_iter == 0 and cfg.DIFF_ALIGN.CLS_PAR > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            netC_aux.eval()
            test_clip_all_output = CLIP_model_eval(dset_loaders['test'], text_inputs, text_features, clip_model, cfg)
            netF.train()
            netB.train()
            netC.train()
            netC_aux.train()

        logit_clip_score, text_features = mypromt(inputs_test)
        text_features_all = text_features
        logit_clip_score_include_hidden = clip_text(clip_model, text_features_all, inputs_test)

        if 'RN' in cfg.DIFF_ALIGN.ARCH:
            prompt_lr = cfg.OPTIM.LR * 0.1
            data_transform = image_test_50()
        else:
            data_transform = image_test()
            if iter_num < 3000:
                prompt_lr = cfg.OPTIM.LR
            elif iter_num < 6000:
                prompt_lr = 0.003
            else:
                prompt_lr = 0.001

        try:
            opt_clip.zero_grad()
        except:
            prompt_parameters = [mypromt.prompt_learner.ctx]
            opt_clip = torch.optim.SGD(prompt_parameters, prompt_lr, weight_decay=5e-4, momentum=0.9,
                                       nesterov=False)
            opt_clip.zero_grad()
            if mypromt.prompt_learner.ctx.grad is not None:
                mypromt.prompt_learner.ctx.grad.data.zero_()

        iter_num += 1

        # ── 单次训练迭代计时开始 ─────────────────────────────
        t_iter_start = GPUMonitor.start_timer()
        # ─────────────────────────────────────────────────────

        lr_num = 2000 if cfg.SETTING.DATASET == 'office-home' else 10500
        if iter_num < lr_num:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        else:
            lr_scheduler2(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_shared = netF(inputs_test)
        features_classify = netB(features_shared)
        outputs_test = netC_aux(features_classify)
        base_outputs_test = netC(features_classify)
        base_softmax_out = nn.Softmax(dim=1)(base_outputs_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        ln_sam = softmax_out.shape[0]

        if is_merge_stratg_use:
            abulations = beta_pre_1 * iter_num / max_iter + beta_pre_2
            data = np.random.exponential(scale=abulations, size=ln_sam)
            data[data > 1] = 1
        else:
            data = np.random.exponential(scale=0.1, size=ln_sam)

        data = np.expand_dims(data, axis=1)
        data = torch.from_numpy(data)
        data = data.numpy()

        K = base_softmax_out.size(1)

        base_clip_all_output = nn.Softmax(dim=1)(logit_clip_score_include_hidden[:, :cfg.class_num])
        clip_all_output = nn.Softmax(dim=1)(logit_clip_score_include_hidden)
        _, predict = torch.max(score_bank[tar_idx], 1)
        _, clip_predict = torch.max(base_clip_all_output, 1)

        predict_one = np.eye(K)[predict.cpu()]
        clip_one = np.eye(K)[clip_predict.cpu()]
        predict_mix = data * predict_one + (1 - data) * clip_one
        predict_mix = torch.from_numpy(predict_mix).cuda()

        if cfg.DIFF_ALIGN.CLS_PAR > 0 and use_cls:
            targets = predict_mix
            outputs_log_softmax = base_outputs_test
            pseudo_label_loss = (- targets * outputs_log_softmax).sum(dim=1).mean()
            if cfg.SETTING.DATASET == 'VISDA-C':
                if iter_num < 3000:
                    cls_par = 0.25
                elif iter_num < 6000:
                    cls_par = 0.3
                else:
                    cls_par = 0.4
            else:
                cls_par = 0.4
            classifier_loss = pseudo_label_loss * 0.4
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        neg_iter = 0
        warmup_iter = 0
        warmup_iter = 80
        neg_iter = 100
        bb = 1 - current_aux
        nnn = 1 - current_aux if iter_num > neg_iter else 0.01
        bn = -0.3
        loss_aux, loss_mat = IID_losses.IID_loss(
            cfg.class_num,
            softmax_out.detach() if iter_num < warmup_iter else softmax_out,
            clip_all_output,
            bb, nnn, bn,
            lamb=1.0
        )

        m_base_sft = base_softmax_out.mean(dim=0)
        loss_iid = loss_aux
        if not use_iid:
            loss_iid = torch.tensor(0.0).cuda()
        gentropy_loss = torch.sum(-m_base_sft * torch.log(m_base_sft + cfg.LCFD.EPSILON))

        with torch.no_grad():
            score_bank[tar_idx] = base_softmax_out.detach().clone()

        optimizer.zero_grad()
        opt_C.zero_grad()
        opt_aux.zero_grad()
        total_loss = classifier_loss + loss_iid - cfg.DIFF_ALIGN.GENT_PAR * gentropy_loss
        total_loss.backward()
        optimizer.step()
        opt_C.step()
        opt_aux.step()

        if use_clip:
            if iter_num < (max_iter / 2):
                with torch.no_grad():
                    promt_learning_rate = 0.002
                    mypromt.prompt_learner.ctx -= mypromt.prompt_learner.ctx.grad * promt_learning_rate
            else:
                with torch.no_grad():
                    promt_learning_rate = 0.0001
                    mypromt.prompt_learner.ctx -= mypromt.prompt_learner.ctx.grad * promt_learning_rate

        # ── 单次训练迭代计时结束 ─────────────────────────────
        iter_elapsed = GPUMonitor.stop_timer(t_iter_start)
        train_iter_times.append(iter_elapsed)
        # ─────────────────────────────────────────────────────

        if iter_num % interval_iter == 0 or iter_num == max_iter:

            # ── 评估阶段：传入 monitor，自动打印推理耗时 + 显存 ──
            netF.eval()
            netB.eval()
            netC.eval()
            if cfg.SETTING.DATASET == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(
                    dset_loaders['test'], netF, netB, netC, cfg, True, monitor=monitor)
                acc_s_te_aux, _ = cal_acc(
                    dset_loaders['test'], netF, netB, netC_aux, cfg, True, monitor=None)
                log_str = (
                    'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%,{:.2f}%;'
                    'loss ={},iid={},cls={},gen={}'
                ).format(cfg.name, iter_num, max_iter,
                         acc_s_te, acc_s_te_aux,
                         total_loss, loss_iid, classifier_loss, gentropy_loss) + '\n' + acc_list
                if iter_num == max_iter:
                    acc = acc_s_te
            else:
                acc_s_te, _ = cal_acc(
                    dset_loaders['test'], netF, netB, netC, cfg, False, monitor=monitor)
                acc_s_te_aux, _ = cal_acc(
                    dset_loaders['test'], netF, netB, netC_aux, cfg, False, monitor=None)
                log_str = (
                    'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%,{:.2f}%;loss ={}'
                ).format(cfg.name, iter_num, max_iter,
                         acc_s_te, acc_s_te_aux, total_loss)
                if iter_num == max_iter:
                    acc = acc_s_te
            # ── 训练迭代耗时统计（每次评估时汇报一次）────────────
            if train_iter_times:
                avg_t = sum(train_iter_times) / len(train_iter_times)
                total_train_so_far = sum(train_iter_times)
                train_summary = (
                    f"[训练耗时] 最近 {len(train_iter_times)} 次迭代 | "
                    f"均值: {avg_t * 1000:.1f} ms/iter | "
                    f"合计: {GPUMonitor.format_time(total_train_so_far)}"
                )
                logger.info(train_summary)
                print(train_summary)
                train_iter_times.clear()   # 清空，下次再统计
            # ─────────────────────────────────────────────────────

            monitor.log_memory(tag=f"评估完成 iter={iter_num}")   # ← 新增

            logger.info(log_str)
            netF.train()
            netB.train()

    # ── 整体训练耗时 ─────────────────────────────────────────
    total_elapsed = GPUMonitor.stop_timer(t_total_start)
    summary_msg = (
        f"\n{'=' * 50}\n"
        f"[总耗时] 训练完成 | 总时间: {GPUMonitor.format_time(total_elapsed)}\n"
        f"{'=' * 50}"
    )
    logger.info(summary_msg)
    print(summary_msg)
    monitor.log_memory(tag="训练全部结束")
    # ─────────────────────────────────────────────────────────

    from datetime import datetime
    time_now = datetime.now()

    if False:
        acc_str = f"{acc:.2f}"
        torch.save(netF.state_dict(),
                   osp.join(cfg.output_dir,
                            "target_F_{}_{}_acc={}_{}.pt".format(cfg.savename, cfg.name, acc_str, time_now)))
        torch.save(netB.state_dict(),
                   osp.join(cfg.output_dir,
                            "target_B_{}_{}_acc={}_{}.pt".format(cfg.savename, cfg.name, acc_str, time_now)))
        torch.save(netC.state_dict(),
                   osp.join(cfg.output_dir,
                            "target_C_{}_{}_acc={}_{}.pt".format(cfg.savename, cfg.name, acc_str, time_now)))
        torch.save(netC_aux.state_dict(),
                   osp.join(cfg.output_dir,
                            "target_C_aux_{}_{}_acc={}_{}.pt".format(cfg.savename, cfg.name, acc_str, time_now)))
        torch.save(mypromt.prompt_learner.state_dict(),
                   osp.join(cfg.output_dir,
                            "CLIP_{}_{}_acc={}_{}.pt".format(cfg.savename, cfg.name, acc_str, time_now)))

    return netF, netB, netC


def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.DIFF_ALIGN.CTX_INIT.replace("_", " ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts


def clip_text(model, text_features, inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits
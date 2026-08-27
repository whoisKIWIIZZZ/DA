import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
# from src.utils import loss,prompt_tuning,IID_losses
from src.utils import loss,IID_losses
import pandas as pd

from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList_idx_aug_fix_old
from sklearn.metrics import confusion_matrix
import clip
from src.utils.utils import *
from src.utils.prompt_tuning_diff_new_optim_1130 import *
from src.utils.loss import entropy

logger = logging.getLogger(__name__)

import wandb
import copy

def normalized_confidence(probabilities):
    num_classes = probabilities.size(1)
    entropy_value = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1)
    return (1.0 - entropy_value / np.log(num_classes)).clamp(0.0, 1.0)


def js_stability(first, second):
    midpoint = 0.5 * (first + second)
    first_kl = (first * (torch.log(first + 1e-8) - torch.log(midpoint + 1e-8))).sum(dim=1)
    second_kl = (second * (torch.log(second + 1e-8) - torch.log(midpoint + 1e-8))).sum(dim=1)
    return (1.0 - 0.5 * (first_kl + second_kl) / np.log(2.0)).clamp(0.0, 1.0)


def adaptive_mix_weight(q_target, q_clip, target_output, clip_history, clip_history_seen,
                        observation_count, target_loss, clip_loss, cfg):
    target_confidence = normalized_confidence(q_target)
    clip_confidence = normalized_confidence(q_clip)
    target_stability = js_stability(q_target, target_output)
    clip_reference = torch.where(clip_history_seen.unsqueeze(1), clip_history, q_clip)
    clip_stability = js_stability(q_clip, clip_reference)

    confidence_weight = cfg.DIFF_ALIGN.MERGE_CONFIDENCE_WEIGHT
    stability_weight = cfg.DIFF_ALIGN.MERGE_STABILITY_WEIGHT
    weight_sum = confidence_weight + stability_weight
    target_reliability = (
        confidence_weight * target_confidence + stability_weight * target_stability
    ) / weight_sum
    clip_reliability = (
        confidence_weight * clip_confidence + stability_weight * clip_stability
    ) / weight_sum

    initial_target_weight = cfg.DIFF_ALIGN.MERGE_INITIAL_TARGET_WEIGHT
    initial_logit = np.log(initial_target_weight / (1.0 - initial_target_weight))
    prior_bias = initial_logit * torch.exp(
        -observation_count / cfg.DIFF_ALIGN.MERGE_PRIOR_DECAY
    )
    prior_logit = prior_bias + (
        target_reliability - clip_reliability
    ) / cfg.DIFF_ALIGN.MERGE_GATE_TEMPERATURE
    min_weight = cfg.DIFF_ALIGN.MERGE_MIN_WEIGHT
    prior = torch.sigmoid(prior_logit).clamp(min_weight, 1.0 - min_weight)

    robust_logit = torch.logit(prior) + (
        target_loss.detach() - clip_loss.detach()
    ) / cfg.DIFF_ALIGN.MERGE_ROBUST_TEMPERATURE
    alpha = torch.sigmoid(robust_logit).clamp(min_weight, 1.0 - min_weight)
    return (
        alpha.detach(), prior.detach(), target_confidence.detach(), clip_confidence.detach(),
        target_stability.detach(), clip_stability.detach()
    )


def gate_diagnostics(alpha_bank, alpha_seen, prior_bank, target_prediction, clip_prediction,
                     label_bank, target_confidence, clip_confidence, target_stability,
                     clip_stability, cfg):
    valid = alpha_seen & (label_bank >= 0) & (target_prediction >= 0) & (clip_prediction >= 0)
    if not valid.any():
        return 'Gate: no observations'

    alpha = alpha_bank[valid]
    target_prediction = target_prediction[valid]
    clip_prediction = clip_prediction[valid]
    labels = label_bank[valid]
    target_correct = target_prediction == labels
    clip_correct = clip_prediction == labels
    disagreement = target_prediction != clip_prediction
    selected_prediction = torch.where(alpha >= 0.5, target_prediction, clip_prediction)
    quantiles = torch.quantile(alpha, torch.tensor([0.1, 0.5, 0.9], device=alpha.device))

    target_only = target_correct & ~clip_correct
    clip_only = clip_correct & ~target_correct
    target_only_alpha = alpha[target_only].mean().item() if target_only.any() else float('nan')
    clip_only_alpha = alpha[clip_only].mean().item() if clip_only.any() else float('nan')
    if disagreement.any():
        target_disagreement_acc = target_correct[disagreement].float().mean().item() * 100
        clip_disagreement_acc = clip_correct[disagreement].float().mean().item() * 100
        selected_disagreement_acc = (
            selected_prediction[disagreement] == labels[disagreement]
        ).float().mean().item() * 100
    else:
        target_disagreement_acc = float('nan')
        clip_disagreement_acc = float('nan')
        selected_disagreement_acc = float('nan')

    low_confidence = cfg.DIFF_ALIGN.MERGE_LOW_CONFIDENCE
    both_low = (
        (target_confidence[valid] < low_confidence)
        & (clip_confidence[valid] < low_confidence)
    ).float().mean().item() * 100
    min_weight = cfg.DIFF_ALIGN.MERGE_MIN_WEIGHT
    saturated_low = (alpha <= min_weight + 1e-6).float().mean().item() * 100
    saturated_high = (alpha >= 1.0 - min_weight - 1e-6).float().mean().item() * 100

    return (
        'Gate: n={}; alpha mean={:.3f}, p10/50/90={:.3f}/{:.3f}/{:.3f}; prior={:.3f}; '
        'conf target/clip={:.3f}/{:.3f}; stability target/clip={:.3f}/{:.3f}; '
        'sat low/high={:.1f}%/{:.1f}%; disagree={:.1f}%; '
        'disagree acc target/clip/selected={:.2f}%/{:.2f}%/{:.2f}%; '
        'oracle alpha target-only/clip-only={:.3f}/{:.3f}; both-low-conf={:.1f}%'
    ).format(
        alpha.numel(), alpha.mean().item(), quantiles[0].item(), quantiles[1].item(),
        quantiles[2].item(), prior_bank[valid].mean().item(),
        target_confidence[valid].mean().item(), clip_confidence[valid].mean().item(),
        target_stability[valid].mean().item(), clip_stability[valid].mean().item(),
        saturated_low, saturated_high, disagreement.float().mean().item() * 100,
        target_disagreement_acc, clip_disagreement_acc, selected_disagreement_acc,
        target_only_alpha, clip_only_alpha, both_low
    )




def print_grad_hook(grad):
    if grad.norm().item() > 100:
        print('Hook: Grad recieved with norm {}'.format(grad.norm().item()))




def image_test_50(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=[0.26862954, 0.26130258, 0.27577711])

    return transforms.Compose([
        transforms.Resize(crop_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(cfg,preprocess):
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
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                        drop_last=False)
    dsets["test"] = ImageList_idx_aug_fix_old(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["clip_test"] = ImageList_idx_aug_fix_old(txt_test, transform=image_test())
    dset_loaders["clip_test"] = DataLoader(dsets["clip_test"], batch_size=train_bs * 3, shuffle=False,num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    # else:
    # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    # else:
    # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)  # 0<decay<1，且越来越小
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        # param_group['momentum'] = 0.9
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def lr_scheduler2(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        # param_group['momentum'] = 0.9
        param_group['momentum'] = 0.5
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, netF, netB, netC, cfg,flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            outputs = outputs[:,:cfg.class_num]
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)


    num_known_classes = cfg.class_num
    all_output_known = all_output[:, :num_known_classes]
    _, predict = torch.max(all_output_known, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output_known))).cpu().data.item()
    # _, predict = torch.max(all_output, 1)
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def cal_satistical_info(loader, netF, netB, netC, clip_softmax_score,cfg, flag=False):
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
        base_softmax_score_known = base_softmax_score[:,:cfg.class_num]
        clip_softmax_known = clip_softmax_score_cpu[:, :cfg.class_num]


        _, base_predict = torch.max(base_softmax_score_known, 1)
        _, clip_predict = torch.max(clip_softmax_known, 1)
        base_sign = (torch.squeeze(base_predict).float() == all_label)
        clip_sign = (torch.squeeze(clip_predict).float() == all_label)
        # base是对的，但clip是错的
        base_corr_clip_wrong = (base_sign & (~clip_sign))
        # base是错的，但clip是对的
        base_wrong_clip_corr = ((~base_sign) & clip_sign)
        data_size = all_label.size()[0]

        base_corr_clip_wrong_num = torch.sum(base_corr_clip_wrong).item()
        base_wrong_clip_corr_num = torch.sum(base_wrong_clip_corr).item()
        # wandb.log({"base_corr_clip_wrong": base_corr_clip_wrong_num / data_size,
        #            "base_wrong_clip_corr": base_wrong_clip_corr_num / data_size})
        log_str = 'The base_corr_clip_wrong= {:.2f}%;The base_wrong_clip_corr={:.2f}%'.format(
            100 * base_corr_clip_wrong_num / data_size, 100 * base_wrong_clip_corr_num / data_size)
        logger.info(log_str)

        # base正确但clip错误的base和clip的softmax的平均值和第二类的差距
        B_True_C_False_base_soft = base_softmax_score[base_corr_clip_wrong]
        B_True_C_False_clip_soft = clip_softmax_score[base_corr_clip_wrong]
        groud_truth_index_BTCF = all_label[base_corr_clip_wrong]
        CFBT_base_mean, CFBT_base_var = evaluesoftmax_goodpredictiom(B_True_C_False_base_soft)
        CFBT_clip_mean, CFBT_clip_var = evaluesoftmax_badpredictiom(B_True_C_False_clip_soft, groud_truth_index_BTCF)
        # wandb.log({" CFBT_base_mean": CFBT_base_mean, "CFBT_base_var": CFBT_base_var, "CFBT_clip_mean": CFBT_clip_mean,
        #            "CFBT_clip_var": CFBT_clip_var})
        log_str = 'The CFBT_base_mean= {}; The CFBT_base_var={};The CFBT_clip_mean={};The CFBT_clip_var={}'.format(
            CFBT_base_mean, CFBT_base_var, CFBT_clip_mean, CFBT_clip_var)
        logger.info(log_str)

        # clip正确base错误softmax的平均值和第二类的差距
        B_False_C_True_base_soft = base_softmax_score[base_wrong_clip_corr]
        B_False_C_True_clip_soft = clip_softmax_score[base_wrong_clip_corr]
        groud_truth_index_BFCT = all_label[base_wrong_clip_corr]
        BFCT_base_mean, BFCT_base_var = evaluesoftmax_goodpredictiom(B_False_C_True_clip_soft)
        BFCT_clip_mean, BFCT_clip_var = evaluesoftmax_badpredictiom(B_False_C_True_base_soft, groud_truth_index_BFCT)
        # wandb.log({" BFCT_base_mean": BFCT_base_mean, "BFCT_base_var": BFCT_base_var, "BFCT_clip_mean": BFCT_clip_mean,
        #            "BFCT_clip_var": BFCT_clip_var})
        log_str = 'The BFCT_base_mean= {}; The BFCT_base_var={};The BFCT_clip_mean={};The BFCT_clip_var={}'.format(
            BFCT_base_mean, BFCT_base_var, BFCT_clip_mean, BFCT_clip_var)
        logger.info(log_str)

        # base正确clip正确，与第二类的差值
        base_True_clip_True = (base_sign & clip_sign)
        B_True_C_True_base_soft = base_softmax_score[base_True_clip_True]
        B_True_C_True_clip_soft = clip_softmax_score[base_True_clip_True]
        groud_truth_index_BTCT = all_label[base_True_clip_True]
        BTCT_base_mean, BTCT_base_var = evaluesoftmax_goodpredictiom(B_True_C_True_base_soft)
        BTCT_clip_mean, BTCT_clip_var = evaluesoftmax_goodpredictiom(B_True_C_True_clip_soft)
        # wandb.log({" BTCT_base_mean": BTCT_base_mean, "BTCT_base_var": BTCT_base_var, "BTCT_clip_mean": BTCT_clip_mean,
        #            "BTCT_clip_var": BTCT_clip_var})
        log_str = 'The BTCT_base_mean= {}; The BTCT_base_var={};The BTCT_clip_mean={};The BTCT_clip_var={}'.format(
            BTCT_base_mean, BTCT_base_var, BTCT_clip_mean, BTCT_clip_var)
        logger.info(log_str)

        # base错误clip错误，分别与groundtruth的softmax上差异
        base_False_clip_False = (~base_sign & ~clip_sign)
        B_False_C_False_base_soft = base_softmax_score[base_False_clip_False]
        B_False_C_False_clip_soft = clip_softmax_score[base_False_clip_False]
        ground_truth_index_BFCF = all_label[base_False_clip_False]
        BFCF_base_mean, BFCF_base_var = evaluesoftmax_badpredictiom(B_False_C_False_base_soft, ground_truth_index_BFCF)
        BFCF_clip_mean, BFCF_clip_var = evaluesoftmax_badpredictiom(B_False_C_False_clip_soft, ground_truth_index_BFCF)
        # wandb.log({" BFCF_base_mean": BFCF_base_mean, "BFCF_base_var": BFCF_base_var, "BFCF_clip_mean": BFCF_clip_mean,
        #            "BFCF_clip_var": BFCF_clip_var})
        log_str = 'The BFCF_base_mean= {}; The BFCF_base_var={};The BFCF_clip_mean={};The BFCF_clip_var={}'.format(
            BFCF_base_mean, BFCF_base_var, BFCF_clip_mean, BFCF_clip_var)
        logger.info(log_str)


# 输出变量
def evaluesoftmax_goodpredictiom(softmaxscore):
    acc, predict = torch.topk(softmaxscore, k=2, dim=1)
    distance = acc[:, 0] - acc[:, 1]
    return torch.mean(distance), torch.var(distance)


def evaluesoftmax_badpredictiom(softmaxscore, labelgroundtruth):  # 411*65
    size = labelgroundtruth.shape[0]
    tmp = torch.zeros(size, dtype=torch.float32, device='cpu')
    for i in range(size):
        tmp[i] = softmaxscore[i, labelgroundtruth[i].int()]

    # tmp_index = torch.cat(torch.arange(size),labelgroundtruth,dim=1)
    # tmp_index = torch.cat(torch.arange(size),labelgroundtruth,dim=1)
    distance = torch.max(softmaxscore, dim=1)[0].cpu() - tmp
    return torch.mean(distance), torch.var(distance)


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer






def train_target(cfg, beta_pre_1=1, beta_pre_2=0.1, use_iid=True, use_cls=True,
                 use_clip=True, aux_begin=0.3):
    print("using the new opt file")
    print("iid:{},cls:{},clip:{}".format(use_iid, use_cls, use_clip))
    acc = 0
    aux_end = 0.1

    print("aux_begin:", aux_begin, "aux_end:", aux_end)
    print(
        "entropy bank beta:", beta_pre_1, beta_pre_2,
        "merge initial target weight:", cfg.DIFF_ALIGN.MERGE_INITIAL_TARGET_WEIGHT,
        "confidence/stability:", cfg.DIFF_ALIGN.MERGE_CONFIDENCE_WEIGHT,
        cfg.DIFF_ALIGN.MERGE_STABILITY_WEIGHT,
        "robust temperature:", cfg.DIFF_ALIGN.MERGE_ROBUST_TEMPERATURE
    )
    # clip模型参数初始化
    clip_model, preprocess, _ = clip.load(cfg.DIFF_ALIGN.ARCH)  # type: ignore # vit-B32
    clip_model.float()
    # 引入clip后，自构造text的输入
    text_inputs = clip_pre_text(cfg)  # 65*77
    # 训练测试数据导入
    dset_loaders = data_load(cfg,preprocess)

    print("using the arch:", cfg.MODEL.ARCH, "+++++++++++++++++++++++++++++")
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
 
    modelpath = cfg.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath),strict=True) # type: ignore
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath),strict=True)
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))


    netC.eval()

    param_group = []
    param_group_C = []
    

    
    

    for k,v in netC.named_parameters():
        v.requires_grad = True
        param_group_C +=  [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1*0.3}]


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

    

    opt_C = optim.SGD(param_group_C)
    opt_C = op_copy(opt_C)

    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    # score_bank包含所有样本
    score_bank = torch.randn(num_sample, cfg.class_num).cuda()  # 4365目标样本,65类
    source_score_bank = torch.zeros(num_sample).cuda()
    clip_score_bank = torch.zeros(num_sample, cfg.class_num).cuda()
    clip_score_seen = torch.zeros(num_sample, dtype=torch.bool, device=score_bank.device)
    gate_observation_bank = torch.zeros(num_sample, device=score_bank.device)
    alpha_bank = torch.zeros(num_sample, device=score_bank.device)
    alpha_seen = torch.zeros(num_sample, dtype=torch.bool, device=score_bank.device)
    target_confidence_bank = torch.zeros(num_sample, device=score_bank.device)
    clip_confidence_bank = torch.zeros(num_sample, device=score_bank.device)
    prior_bank = torch.zeros(num_sample, device=score_bank.device)
    target_stability_bank = torch.zeros(num_sample, device=score_bank.device)
    clip_stability_bank = torch.zeros(num_sample, device=score_bank.device)
    target_score_bank = torch.zeros(num_sample, device=score_bank.device)
    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
            indx = data[-1]  # 把对应的score_bank里面的内容更新
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            outputs = netC(output)
            outputs = nn.Softmax(dim=1)(outputs)
            # Pad the outputs to match the score_bank dimension including hidden classes
            score_bank[indx] = outputs.detach().clone()
            source_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean()
            source_score_bank[indx] = source_score.detach().clone()

    if cfg.SETTING.DATASET == 'office-home':
        
        max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])*2  
        interval_iter = max_iter  // cfg.TEST.INTERVAL  
        copy_iter = max_iter  //15// cfg.TEST.INTERVAL  
    else:
        
        max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
        interval_iter = max_iter // cfg.TEST.INTERVAL
        copy_iter = max_iter  //15// cfg.TEST.INTERVAL
   
  

    iter_num = 0
    epoch_num = 0
    text_features = None
    loss_neg = torch.tensor(0.0)


    mypromt = promote_learning_diff_promote_learning(cfg, clip_model).to(clip_model.visual.conv1.weight.device)
    gate_log_str = 'Gate: no observations'
    # modelpath = "/data02/home/scv8780/run/sfda-my/output_Models/uda/VISDA-C/TV/diff_align/CLIP_diff_align_TV_acc=90.18_2026-01-10 01:53:47.762324.pt"
    # mypromt.prompt_learner.load_state_dict(torch.load(modelpath))
   # mypromt.prompt_learner.ctx.register_hook(print_grad_hook)
    while iter_num < max_iter:

        current_aux = aux_begin + (aux_end - aux_begin) * (iter_num / max_iter)
        try:
            (inputs_test, inputs_clip), _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, inputs_clip), _, tar_idx = next(iter_test)

        inputs_test = inputs_test.cuda()
        inputs_clip = inputs_clip.cuda()
        
        if inputs_test.size(0) == 1:
            continue
        evaluate_now = iter_num == 0 or (iter_num + 1) % interval_iter == 0 or iter_num + 1 == max_iter
        if evaluate_now and cfg.DIFF_ALIGN.CLS_PAR > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            epoch_num = int((iter_num + 1) / interval_iter)
            test_clip_all_output = CLIP_model_eval(
                dset_loaders['clip_test'], text_inputs, text_features, clip_model, cfg
            )
            clip_prediction_bank = test_clip_all_output.argmax(dim=1).to(score_bank.device)
            target_prediction_bank = torch.full(
                (num_sample,), -1, dtype=torch.long, device=score_bank.device
            )
            eval_label_bank = torch.full_like(target_prediction_bank, -1)
            target_score_bank.zero_()
            with torch.no_grad():
                test_iter = iter(dset_loaders["test"])
                for _ in range(len(dset_loaders["test"])):
                    test_data = next(test_iter)
                    test_inputs = test_data[0][0]
                    test_labels = test_data[1]
                    test_indx = test_data[-1].to(score_bank.device)
                    test_inputs = test_inputs.cuda()
                    test_outputs = netC(netB(netF(test_inputs)))
                    test_softmax = nn.Softmax(dim=1)(test_outputs)
                    test_scores = -torch.sum(test_softmax * torch.log(test_softmax + 1e-9), dim=1)
                    target_score_bank[test_indx] = test_scores
                    target_prediction_bank[test_indx] = test_softmax.argmax(dim=1)
                    eval_label_bank[test_indx] = test_labels.to(score_bank.device).long()

            gate_log_str = gate_diagnostics(
                alpha_bank, alpha_seen, prior_bank, target_prediction_bank,
                clip_prediction_bank, eval_label_bank, target_confidence_bank,
                clip_confidence_bank, target_stability_bank, clip_stability_bank, cfg
            )
            netF.train()
            netB.train()
            netC.train()


        logit_clip_score, text_features = mypromt(inputs_clip)
        text_features_all = text_features
        logit_clip_score_include_hidden = clip_text(clip_model, text_features_all, inputs_clip)
        
        # trainable_param = deepcopy(mypromt.prompt_learner.ctx)

        if 'RN' in cfg.DIFF_ALIGN.ARCH:
            prompt_lr = cfg.OPTIM.LR * 0.1
            data_transform = image_test_50()
        else:  # 走这一边
            data_transform = image_test()
            #prompt_lr = cfg.OPTIM.LR
            if iter_num < 3000:
                prompt_lr =cfg.OPTIM.LR
            elif iter_num< 6000:
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
        lr_num = 2000 if cfg.SETTING.DATASET == 'office-home' else 10500
        if iter_num < lr_num:  # 600,2000最好
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        else:
            lr_scheduler2(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_shared = netF(inputs_test)
        features_classify = netB(features_shared)
        
        base_outputs_test = netC(features_classify)
        base_softmax_out = nn.Softmax(dim=1)(base_outputs_test)
        softmax_out = base_softmax_out
        #softmax_out = nn.Softmax(dim=1)(outputs_test)

        d = (beta_pre_1 / 20) * iter_num / max_iter + (beta_pre_2 / 10)
        source_score_bank[tar_idx] = (
            (1 - d) * source_score_bank[tar_idx] + d * target_score_bank[tar_idx]
        )
        current_batch_scores = source_score_bank[tar_idx]
        avg_score = current_batch_scores.mean().item()

        diff = torch.abs(
            -torch.sum(base_softmax_out * torch.log(base_softmax_out + 1e-5), 1) - avg_score
        )
        diff = diff.detach().clone()

        margin = 0.01 * (1.01 ** (epoch_num - 1))
        loss_ent = entropy(base_softmax_out, avg_score, margin)
        mask_greater = diff > margin
        indices_greater = torch.where(mask_greater)[0]

        base_clip_all_output = nn.Softmax(dim=1)(
            logit_clip_score_include_hidden[:, :cfg.class_num]
        )
        clip_all_output = base_clip_all_output
        q_target = score_bank[tar_idx].detach()
        q_clip = base_clip_all_output.detach()
        log_prob = F.log_softmax(base_outputs_test, dim=1)
        target_loss_per_sample = -(q_target * log_prob).sum(dim=1)
        clip_loss_per_sample = -(q_clip * log_prob).sum(dim=1)
        alpha, prior, target_confidence, clip_confidence, target_stability, clip_stability = (
            adaptive_mix_weight(
                q_target,
                q_clip,
                base_softmax_out.detach(),
                clip_score_bank[tar_idx],
                clip_score_seen[tar_idx],
                gate_observation_bank[tar_idx],
                target_loss_per_sample,
                clip_loss_per_sample,
                cfg
            )
        )
        dd =np.random.exponential(scale = ((iter_num/max_iter)+0.1),size = softmax_out.shape[0]) 
        dd = np.expand_dims(dd,axis=1)
        dd = torch.from_numpy(dd)
        dd[dd>1]=1
        x_used = True
        d = alpha.unsqueeze(1)  if x_used else dd
        predict_mix = alpha.unsqueeze(1) * q_target + (1.0 - alpha.unsqueeze(1)) * q_clip

        has_selected_samples = indices_greater.numel() > 0
        if cfg.DIFF_ALIGN.CLS_PAR > 0 and use_cls and has_selected_samples:
            pseudo_label_loss = -(
                predict_mix[indices_greater] * log_prob[indices_greater]
            ).sum(dim=1).mean()
            classifier_loss = pseudo_label_loss * cfg.DIFF_ALIGN.CLS_PAR
        else:
            classifier_loss = base_outputs_test.new_zeros(())


        if use_iid and has_selected_samples:
            loss_iid, loss_mat = IID_losses.IID_loss_old(
                softmax_out[indices_greater].detach(),
                clip_all_output[indices_greater],
            )
        else:
            loss_iid = base_outputs_test.new_zeros(())
            loss_mat = None

        m_base_sft = base_softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-m_base_sft * torch.log(m_base_sft + cfg.LCFD.EPSILON)) 
      
        with torch.no_grad():
            score_bank[tar_idx] = base_softmax_out.detach()
            clip_score_bank[tar_idx] = q_clip
            clip_score_seen[tar_idx] = True
            gate_observation_bank[tar_idx] += 1
            alpha_bank[tar_idx] = alpha
            alpha_seen[tar_idx] = True
            target_confidence_bank[tar_idx] = target_confidence
            clip_confidence_bank[tar_idx] = clip_confidence
            prior_bank[tar_idx] = prior
            target_stability_bank[tar_idx] = target_stability
            clip_stability_bank[tar_idx] = clip_stability

        optimizer.zero_grad()
        opt_C.zero_grad()
       
        total_loss = classifier_loss + loss_iid - cfg.DIFF_ALIGN.GENT_PAR * gentropy_loss + cfg.DIFF_ALIGN.LENT_PAR * loss_ent
        total_loss.backward()
        optimizer.step()
        
        opt_C.step()
       
        if use_clip:
            if iter_num <  (max_iter/2):
                with torch.no_grad():
                    #opt_clip.step()
                    promt_learning_rate = 0.002
                    mypromt.prompt_learner.ctx  -=  mypromt.prompt_learner.ctx.grad * promt_learning_rate
            else:
                with torch.no_grad():
                    #opt_clip.step()
                    promt_learning_rate = 0.0001
                    mypromt.prompt_learner.ctx  -=  mypromt.prompt_learner.ctx.grad * promt_learning_rate

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if cfg.SETTING.DATASET == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC,cfg, True)
                
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%,{:.2f}%;loss ={},iid={},cls={},gen={}'.format(cfg.name, iter_num, max_iter,
                                                                                     acc_s_te,
                                                                                     total_loss,loss_iid,classifier_loss,gentropy_loss) + '\n' + acc_list
                if iter_num == max_iter:
                    acc =acc_s_te
               
               # wandb.log({"accuracy": acc_s_te, "loss ": total_loss})
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC,cfg, False)
               
                neg_str = ''
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}{}'.format(cfg.name, iter_num, max_iter,
                                                                                     acc_s_te,
                                                                                     total_loss, neg_str)

                if iter_num == max_iter:
                    acc = acc_s_te
             #   wandb.log({"accuracy": acc_s_te, "loss ": total_loss})
                    
            log_str += '\n' + gate_log_str
            logger.info(log_str)

           
            netF.train()
            netB.train()

    return netF, netB, netC




def clip_pre_text(cfg):
    #M = cfg.class_num_hidden
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
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # 把text变为数字信息

    return tokenized_prompts



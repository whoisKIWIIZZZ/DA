import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import  torch
import logging
import torch.nn as nn
import numpy as np
from src.utils import loss
from sklearn.metrics import confusion_matrix
import os.path as osp
logger = logging.getLogger(__name__)
def cal_acc(loader, netF, netB, netC, cfg,flag=False):
   # print(11)
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            #outputs = outputs[:,:cfg.class_num]
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    #print(all_output.shape)
    num_known_classes = cfg.class_num
    all_output_known = all_output[:, :num_known_classes]
    _, predict = torch.max(all_output_known, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output_known))).cpu().data.item()
    # _, predict = torch.max(all_output, 1)
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    accuracy_neg = 0

    if all_output.size(1) > num_known_classes:
        #print(1)
        all_output_neg = all_output[:, num_known_classes:]
        _, predict = torch.min(all_output_neg, 1)
        accuracy_neg = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    #mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output_known))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, accuracy_neg,mean_ent


def analyze_and_plot_confidence(base_outputs, clip_outputs, labels, iter_num, difficult_data_indices, difficult_data_initial_quadrant_colors, slightly_wrong_indices, output_dir, base_confidence, clip_confidence):
    print("----- Starting Confidence Analysis and Plotting -----")
    
    base_softmax = nn.Softmax(dim=1)(base_outputs).cpu()
    clip_softmax = nn.Softmax(dim=1)(clip_outputs).cpu()
    labels = labels.cpu()
    
    _, base_preds = torch.max(base_softmax, 1)
    _, clip_preds = torch.max(clip_softmax, 1)

    base_correct = (base_preds == labels)
    clip_correct = (clip_preds == labels)
    
    btct_mask = base_correct & clip_correct
    bfcf_mask = (~base_correct) & (~clip_correct)
    btcf_mask = base_correct & (~clip_correct)
    bfct_mask = (~base_correct) & clip_correct

    total_samples = len(labels)
    btct_rate = btct_mask.sum().item() / total_samples
    bfcf_rate = bfcf_mask.sum().item() / total_samples
    btcf_rate = btcf_mask.sum().item() / total_samples
    bfct_rate = bfct_mask.sum().item() / total_samples

    logging.info(f"BTCT (Base True, CLIP True) Rate: {btct_rate:.2%}")
    logging.info(f"BFCF (Base False, CLIP False) Rate: {bfcf_rate:.2%}")
    logging.info(f"BTCF (Base True, CLIP False) Rate: {btcf_rate:.2%}")
    logging.info(f"BFCT (Base False, CLIP True) Rate: {bfct_rate:.2%}")

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(base_confidence[btct_mask], clip_confidence[btct_mask], color='g', label=f'Base True, CLIP True ({btct_mask.sum().item()})', alpha=0.6, s=50)
    ax.scatter(base_confidence[bfcf_mask], clip_confidence[bfcf_mask], color='r', label=f'Base False, CLIP False ({bfcf_mask.sum().item()})', alpha=0.6, s=50)
    ax.scatter(base_confidence[btcf_mask], clip_confidence[btcf_mask], color='orange', label=f'Base True, CLIP False ({btcf_mask.sum().item()})', alpha=0.6, s=50)
    ax.scatter(base_confidence[bfct_mask], clip_confidence[bfct_mask], color='b', label=f'Base False, CLIP True ({bfct_mask.sum().item()})', alpha=0.6, s=50)

    if slightly_wrong_indices is not None:
        ax.scatter(base_confidence[slightly_wrong_indices], clip_confidence[slightly_wrong_indices], color='cyan', marker='D', s=100, edgecolor='black', label=f'Tracked Slightly Wrong ({len(slightly_wrong_indices)})')

    if difficult_data_indices is not None and difficult_data_initial_quadrant_colors is not None:
        for i, initial_color in zip(difficult_data_indices, difficult_data_initial_quadrant_colors):
            ax.scatter(base_confidence[i], clip_confidence[i], color=initial_color, marker='*', s=150, edgecolor='black')

    # Draw the rectangle
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='purple', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    ax.set_title(f'Confidence Distribution (Analysis: {iter_num})', fontsize=18)
    ax.set_xlabel('Base Model Confidence Score (P_true - P_max_other)', fontsize=14)
    ax.set_ylabel('CLIP Model Confidence Score (P_true - P_max_other)', fontsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    
    plot_path = osp.join(output_dir, f'standalone_confidence_plot_{iter_num}.png')
    fig.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Confidence plot saved to: {plot_path}")


def cal_satistical_info(loader, netF, netB, netC, clip_softmax_score,cfg, iter_num, difficult_data_indices, difficult_data_initial_quadrant_colors, slightly_wrong_indices, flag=False):
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
        
        def calculate_confidence(softmax_scores, true_labels):
            scores = []
            for i in range(len(true_labels)):
                true_label = int(true_labels[i].item())
                p_true = softmax_scores[i, true_label]
                
                other_scores = np.delete(softmax_scores[i].numpy(), true_label)
                p_max_other = np.max(other_scores) if other_scores.size > 0 else 0
                
                scores.append(p_true.item() - p_max_other)
            return np.array(scores)
        
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

        # Calculate confidence scores for all samples to identify difficult data
        base_confidence_all = calculate_confidence(base_softmax_score.detach(), all_label)
        clip_confidence_all = calculate_confidence(clip_softmax_score_cpu.detach(), all_label)

        btct_mask_all = base_sign & clip_sign
        bfcf_mask_all = ~base_sign & ~clip_sign
        btcf_mask_all = base_sign & ~clip_sign
        bfct_mask_all = ~base_sign & clip_sign

        if difficult_data_indices is None:
            square_min = -0.05
            square_max = 0.05
            difficult_mask = (base_confidence_all >= square_min) & (base_confidence_all <= square_max) & \
                             (clip_confidence_all >= square_min) & (clip_confidence_all <= square_max)
            
            difficult_data_indices = [i for i, val in enumerate(difficult_mask) if val]
            
            difficult_data_initial_quadrant_colors = []
            for idx in difficult_data_indices:
                if btct_mask_all[idx]:
                    difficult_data_initial_quadrant_colors.append('g') # Green for BTCT
                elif bfcf_mask_all[idx]:
                    difficult_data_initial_quadrant_colors.append('r') # Red for BFCF
                elif btcf_mask_all[idx]:
                    difficult_data_initial_quadrant_colors.append('orange') # Orange for BTCF
                elif bfct_mask_all[idx]:
                    difficult_data_initial_quadrant_colors.append('b') # Blue for BFCT
        
        if slightly_wrong_indices is None:
            special_bfct_mask = bfct_mask_all & (base_confidence_all > -0.25) & (clip_confidence_all > 0.8)
            slightly_wrong_indices = [i for i, val in enumerate(special_bfct_mask) if val]

        analyze_and_plot_confidence(all_output, clip_softmax_score, all_label, iter_num, difficult_data_indices, difficult_data_initial_quadrant_colors, slightly_wrong_indices, cfg.output_dir, base_confidence_all, clip_confidence_all)

        return difficult_data_indices, difficult_data_initial_quadrant_colors, slightly_wrong_indices



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
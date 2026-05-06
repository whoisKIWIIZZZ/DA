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
from src.utils import loss

logger = logging.getLogger(__name__)

# ==========================================================================================
# Helper Functions (Copied from source_pretrain_neg_objective.py)
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
    
    if cfg.SOURCE.TRTE == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs*2, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)

    return dset_loaders

def cal_acc_neg(loader, netF, netB, netC):
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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.min(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    return accuracy*100, mean_ent

# ==========================================================================================
# Main Training Function
# ==========================================================================================

def train_source_neg_objective(cfg):
    logger.info("Starting to train a dedicated negative head...")

    # --- 1. Data Loading ---
    dset_loaders = data_load(cfg)
    
    # --- 2. Model Initialization ---
    if cfg.MODEL.ARCH.startswith('res'):
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
        netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    else:
        raise ValueError(f"Unsupported architecture: {cfg.MODEL.ARCH}")

    # Initialize the negative head classifier
    netC_aux = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
 

    # --- 3. Load and Freeze Pre-trained Models ---
    try:
        f_path = osp.join(cfg.output_dir_src, "source_F.pt")
        b_path = osp.join(cfg.output_dir_src, "source_B.pt")
        c_path = osp.join("./Models/source/uda/office-home/A", "source_C_aux_neg_head.pt")
        netF.load_state_dict(torch.load(f_path))
        netB.load_state_dict(torch.load(b_path))
        netC_aux.load_state_dict(torch.load(c_path))
        logger.info(f"Loaded pre-trained netF and netB from {cfg.output_dir_src}")
    except FileNotFoundError:
        logger.error("Pre-trained source_F.pt or source_B.pt not found! Please run standard source training first.")
        return

    netF.eval()
    netB.eval()
    for param in netF.parameters():
        param.requires_grad = False
    for param in netB.parameters():
        param.requires_grad = False
    
    # --- 4. Optimizer ---
    # The optimizer will only update the parameters of netC_aux
    optimizer = op_copy(optim.SGD(netC_aux.parameters(), lr=cfg.OPTIM.LR, momentum=0.9, weight_decay=1e-3, nesterov=True))

    # --- 5. Training Loop ---
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0
    acc_init = 0

    p_low = 2e-5
    p_uniform = (1.0 - p_low) / (cfg.class_num - 1)

    netC_aux.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(source_iter)
        except (NameError, StopIteration):
            source_iter = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(source_iter)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        # --- Loss Calculation ---
        with torch.no_grad(): # Features are extracted without computing gradients
            features = netB(netF(inputs_source))
        
        outputs_aux = netC_aux(features)
        
        batch_size = inputs_source.size(0)
        target_distribution_neg = torch.full((batch_size, cfg.class_num), p_uniform).cuda()
        target_distribution_neg.scatter_(1, labels_source.unsqueeze(1), p_low)
        
        log_p_neg = F.log_softmax(outputs_aux, dim=1)
        loss = F.kl_div(log_p_neg, target_distribution_neg, reduction='batchmean')

        # --- Optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter or iter_num == 1:
            netC_aux.eval()
            neg_acc, _ = cal_acc_neg(dset_loaders['source_te'], netF, netB, netC_aux)
            
            
            num_samples_to_print = 3
            if batch_size > num_samples_to_print:
                rand_indices = torch.randperm(batch_size)[:num_samples_to_print]
            else:
                rand_indices = torch.arange(batch_size)

            debug_log_str = "\n--- Sampled Predictions for Debugging ---"
            for i in rand_indices:
                label = labels_source[i].item()
                debug_log_str += f"\n[Sample GT: {label}]"
               
                # ResNet Aux (Negative)
                softmax_aux = F.softmax(outputs_aux[i], dim=0).detach().cpu().numpy()
                debug_log_str += f"\n  netC_aux (neg)  softmax: {np.array_str(softmax_aux, precision=5, suppress_small=True)}"

            debug_log_str += "\n-----------------------------------------"
            logger.info(debug_log_str)
            
            log_str = (f"Iter: {iter_num}/{max_iter}, Loss: {loss.item():.4f}, "
                       f"Neg Acc: {neg_acc:.2f}%")
            logger.info(log_str)

            if neg_acc >= acc_init:
                acc_init = neg_acc
                best_netC_aux = netC_aux.state_dict()
                logger.info(f"Best model updated at iter {iter_num} with Neg Acc: {neg_acc:.2f}%")

            netC_aux.train()

    # --- 6. Save Model ---
    if True:
        logger.info(f"Saving best negative head (from iter with Neg Acc: {acc_init:.2f}%)")
        output_dir = cfg.output_dir_src
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
            
        torch.save(best_netC_aux, osp.join(output_dir, "source_C_aux_neg_head.pt"))
        logger.info(f"Model saved to {output_dir}")


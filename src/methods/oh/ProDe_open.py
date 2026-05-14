#prode_open.py
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

# --- Open-set SFDA for ProDe ---
# Source model: 65 classes (closed-set). Extract 25 known class weights.
# Target: all 65 classes (25 known + 40 unknown, unknown mapped to class index 25).
# Classifier: 26-way (25 known + 1 unknown).
# CLIP prompts: only for the 25 known classes (via get_coop).
# Unknown class: initialized as negative mean of known class weights.


def data_load(cfg):
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
    if cfg.DA == 'open':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            old_label = int(reci[1])
            if old_label in cfg.src_classes:
                line = reci[0] + ' ' + str(label_map_s[old_label]) + '\n'
                new_tar.append(line)
            else:
                line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    elif not cfg.DA == 'uda':
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
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source"] = ImageList_idx(txt_test, transform=image_test())
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


def cal_acc_os(loader, netF, netB, netC, num_known):
    """Open-set evaluation: OS acc, known acc, unknown acc, H-score"""
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
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)

    os_acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    known_mask = all_label < num_known
    if known_mask.sum() > 0:
        known_pred = predict[known_mask]
        known_label = all_label[known_mask]
        known_acc = torch.sum(known_pred == known_label).item() / float(known_mask.sum())
    else:
        known_acc = 0.0

    unknown_mask = all_label == num_known
    if unknown_mask.sum() > 0:
        unknown_acc = torch.sum(predict[unknown_mask] == num_known).item() / float(unknown_mask.sum())
    else:
        unknown_acc = 0.0

    if known_acc > 0 and unknown_acc > 0:
        h_score = 2.0 / (1.0 / known_acc + 1.0 / unknown_acc)
    else:
        h_score = 0.0

    log_str = 'OS={:.2f}% K={:.2f}% U={:.2f}% H={:.2f}%'.format(
        os_acc * 100, known_acc * 100, unknown_acc * 100, h_score * 100)
    logger.info(log_str)
    return os_acc * 100, known_acc * 100, unknown_acc * 100, h_score * 100


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
            output_logits,_ = model(inputs)
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


def train_target(cfg):
    num_known = len(cfg.src_classes)
    print("Open-set ProDe: known={}, classifier={} (25 known + 1 unknown)".format(num_known, cfg.class_num))
    text_inputs = clip_pre_text(cfg, num_known)
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

    # Load 65-class source classifier, transfer known-class weights to 26-way netC
    old_netC = network.feat_classifier(type='wn', class_num=65, bottleneck_dim=cfg.bottleneck).cuda()
    modelpath = cfg.output_dir_src + '/source_C.pt'
    old_netC.load_state_dict(torch.load(modelpath))
    with torch.no_grad():
        netC.fc.weight_g.data[:num_known].copy_(old_netC.fc.weight_g.data[:num_known])
        netC.fc.weight_v.data[:num_known].copy_(old_netC.fc.weight_v.data[:num_known])
        netC.fc.bias.data[:num_known].copy_(old_netC.fc.bias.data[:num_known])
        # Unknown class: negative mean of known class weights
        netC.fc.weight_g.data[num_known] = -old_netC.fc.weight_g.data[:num_known].mean(dim=0)
        netC.fc.weight_v.data[num_known] = -old_netC.fc.weight_v.data[:num_known].mean(dim=0)
        netC.fc.bias.data[num_known] = 0.0
    del old_netC

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
    print("now we dont have netC")
    for k, v in model.prompt_learner.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    optimizer_ib = optim.SGD(param_group_ib)
    optimizer_ib = op_copy(optimizer_ib)
    optim_state = deepcopy(optimizer_ib.state_dict())
    model.reset_classnames(cfg.classname, cfg.ProDe.ARCH)
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
            inputs = data[0]
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
        inputs_test_augs = inputs_test_augs[0].cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        with torch.no_grad():
            outputs_test_new = outputs_test.clone().detach()

        if cfg.ProDe.ARCH == 'RN50':
            inputs_test_clip = inputs_test_augs
        else:
            inputs_test_clip = inputs_test

        # CLIP TTA only works on known classes
        clip_score = test_time_adapt_eval(inputs_test_clip, model, optimizer_ib, optim_state, cfg, outputs_test_new[:, :num_known])
        clip_score = clip_score.float()

        with torch.no_grad():
            new_clip = (outputs_test_new[:, :num_known] - 1*logtis_bank[tar_idx].cuda()[:, :num_known]) + clip_score.cuda()
            clip_score_sm = nn.Softmax(dim=1)(new_clip)

        _,clip_index_new = torch.max(new_clip, 1)
        clip_score_sm = nn.Softmax(dim=1)(new_clip)

        # IID loss: only on known classes (CLIP has no "unknown" concept)
        iic_loss = IID_losses.IID_loss_old(softmax_out[:, :num_known], clip_score_sm)
        classifier_loss = cfg.ProDe.IIC_PAR * iic_loss

        # Unknown class entropy minimization
        unknown_output = softmax_out[:, num_known]
        loss_unknown_ent = -torch.log(unknown_output + 1e-5).mean()
        classifier_loss = classifier_loss + 0.1 * loss_unknown_ent

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

        # Cross-entropy using CLIP predictions (known classes)
        pred = clip_index_new
        entropy_loss = nn.CrossEntropyLoss()(outputs_test[:, :num_known], pred)
        classifier_loss = cfg.ProDe.GENT_PAR*entropy_loss + classifier_loss

        optimizer.zero_grad()
        optimizer_ib.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        optimizer_ib.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            os_acc, known_acc, unknown_acc, h_score = cal_acc_os(dset_loaders['test'], netF, netB, netC, num_known)
            log_str = 'Task: {}, Iter:{}/{}; OS={:.2f}% K={:.2f}% UK={:.2f}% H={:.2f}%'.format(
                cfg.name, iter_num, max_iter, os_acc, known_acc, unknown_acc, h_score)

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


def clip_pre_text(cfg, num_known=65):
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    classnames = classnames[:num_known]
    cfg.classname = classnames
    prompt_prefix = cfg.ProDe.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts

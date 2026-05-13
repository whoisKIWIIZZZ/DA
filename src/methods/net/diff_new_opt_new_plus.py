import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.utils import loss, IID_losses
import pandas as pd

from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList_idx, ImageList_idx_aug_fix_old
from sklearn.metrics import confusion_matrix
import clip
from src.utils.utils import *
from src.utils.prompt_tuning_diff_new_optim_1130 import *
from src.data.datasets.data_loading import get_test_loader, get_test_loader_aug
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_V_MASK
from data.imagnet_prompts import imagenet_classes
from src.models.model import get_model, ImageNetXWrapper

logger = logging.getLogger(__name__)

import copy


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


def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def image_test_50(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=[0.26862954, 0.26130258, 0.27577711])
    return transforms.Compose([
        transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def image_train(resize_size=256, crop_size=224):
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
    # decay = 1
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


def cal_acc(loader, netF, netB, netC, cfg, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0] if isinstance(data[0], tuple) else data[0]
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

    _, predict = torch.max(all_output[:, :cfg.class_num], 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output[:, :cfg.class_num]))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def clip_pre_text(cfg):
    """net-style: handles imagenet datasets with mask filtering"""
    List_rd = []
    if 'image' in cfg.SETTING.DATASET:
        classnames_all = imagenet_classes
        classnames = []
        if cfg.SETTING.DATASET.split('_')[-1] in ['a', 'r', 'v']:
            label_mask = eval("imagenet_{}_mask".format(cfg.SETTING.DATASET.split('_')[-1]))
            if 'r' in cfg.SETTING.DATASET:
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    else:
        with open(cfg.name_file) as f:
            for line in f:
                List_rd.extend([i for i in line.split()])
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


def data_load_oh(cfg):
    """oh-style data loading for office-home/VISDA-C/office"""
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
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx_aug_fix_old(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["clip_test"] = ImageList_idx(txt_test, transform=image_test_50())
    dset_loaders["clip_test"] = DataLoader(dsets["clip_test"], batch_size=train_bs * 3, shuffle=False,
                                           num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders


def data_load_net(cfg):
    """net-style data loading for imagenet/domainnet126 datasets"""
    dset_loaders = {}
    domain_name = cfg.domain[cfg.SETTING.T]

    dset_loaders["target"] = get_test_loader_aug(adaptation=cfg.ADAPTATION,
                                                  dataset_name=cfg.SETTING.DATASET,
                                                  root_dir=cfg.DATA_DIR,
                                                  domain_name=domain_name,
                                                  rng_seed=cfg.SETTING.SEED,
                                                  batch_size=cfg.TEST.BATCH_SIZE,
                                                  shuffle=True,
                                                  workers=cfg.NUM_WORKERS)

    dset_loaders["test"] = get_test_loader(adaptation=cfg.ADAPTATION,
                                            dataset_name=cfg.SETTING.DATASET,
                                            root_dir=cfg.DATA_DIR,
                                            domain_name=domain_name,
                                            rng_seed=cfg.SETTING.SEED,
                                            batch_size=cfg.TEST.BATCH_SIZE * 3,
                                            shuffle=False,
                                            workers=cfg.NUM_WORKERS)

    dset_loaders["clip_test"] = get_test_loader_aug(adaptation=cfg.ADAPTATION,
                                                 dataset_name=cfg.SETTING.DATASET,
                                                 root_dir=cfg.DATA_DIR,
                                                 domain_name=domain_name,
                                                 rng_seed=cfg.SETTING.SEED,
                                                 batch_size=cfg.TEST.BATCH_SIZE * 3,
                                                 shuffle=False,
                                                 workers=cfg.NUM_WORKERS)
    return dset_loaders


def train_target(cfg, beta_pre_1=1, beta_pre_2=0.1, use_iid=True, use_cls=True, use_clip=True, aux_begin=0.3):
    print("using the new opt plus file (net version)")
    print("iid:{}, cls:{}, clip:{}".format(use_iid, use_cls, use_clip))
    acc = 0
    is_merge_stratg_use = True
    abulations = 0

    beta_pre_1 = beta_pre_1
    beta_pre_2 = beta_pre_2
    aux_begin = aux_begin
    aux_end = 0

    print("aux_begin:", aux_begin, "aux_end:", aux_end)
    print("is_merge_stratg_use", is_merge_stratg_use, "beta_pre_1:", beta_pre_1, "beta_pre_2:", beta_pre_2)

    is_imagenet = 'image' in cfg.SETTING.DATASET

    # CLIP model
    clip_model, preprocess, _ = clip.load(cfg.DIFF_ALIGN.ARCH)
    clip_model.float()
    text_inputs = clip_pre_text(cfg)

    # Data loading: net-style for imagenet, oh-style for office-home/VISDA-C/office
    if is_imagenet:
        cfg.ADAPTATION = 'tent'
        dset_loaders = data_load_net(cfg)
    else:
        dset_loaders = data_load_oh(cfg)

    # Model architecture: netF + netB + netC + netC_aux (core logic unchanged)
    print("using the arch:", cfg.MODEL.ARCH, "+++++++++++++++++++++++++++++")
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
    netC_aux = network.feat_classifier(type='wn', class_num=cfg.class_num + cfg.class_num_hidden,
                                        bottleneck_dim=cfg.bottleneck).cuda()

    # Load source weights
    modelpath = cfg.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath), strict=True)
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath), strict=True)
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    # Initialize netC_aux from netC
    old_fc = netC.fc
    new_fc = torch.nn.utils.weight_norm(nn.Linear(old_fc.in_features, cfg.class_num + cfg.class_num_hidden).cuda())
    with torch.no_grad():
        new_fc.weight_g.data[:cfg.class_num].copy_(old_fc.weight_g.data)
        new_fc.weight_v.data[:cfg.class_num].copy_(old_fc.weight_v.data)
        new_fc.bias.data[:cfg.class_num].copy_(old_fc.bias.data)
        if cfg.class_num_hidden > 0:
            print("Yes, we use hidden classes.")
            torch.nn.init.kaiming_uniform_(new_fc.weight_g.data[cfg.class_num:], a=0, mode='fan_in',
                                           nonlinearity='linear')
            new_fc.bias.data[cfg.class_num:].zero_()
    netC_aux.fc = new_fc
    netC.eval()

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

    # Score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    score_bank = torch.randn(num_sample, cfg.class_num).cuda()
    source_score_bank = torch.zeros(num_sample).cuda()
    netF.eval()
    netB.eval()
    netC.eval()
    netC_aux.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0] if isinstance(data[0], tuple) else data[0]
            indx = data[-1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            outputs = netC(output)
            outputs = nn.Softmax(dim=1)(outputs)
            score_bank[indx] = outputs.detach().clone()
            source_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean()
            source_score_bank[indx] = source_score.detach().clone()

    if cfg.SETTING.DATASET == 'office-home':
        max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"]) * 2
        interval_iter = max_iter // cfg.TEST.INTERVAL
        copy_iter = max_iter // 15 // cfg.TEST.INTERVAL
    else:
        max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
        interval_iter = max_iter // cfg.TEST.INTERVAL
        copy_iter = max_iter // 15 // cfg.TEST.INTERVAL

    iter_num = 0
    epoch_num = 0
    text_features = None

    mypromt = promote_learning_diff_promote_learning(cfg, clip_model).to(clip_model.visual.conv1.weight.device)

    while iter_num < max_iter:

        current_aux = aux_begin + (aux_end - aux_begin) * (iter_num / max_iter)
        try:
            if is_imagenet:
                # net-style: dataset returns (img_1, img_2, label, idx)
                # img_1 = base model view, img_2 = CLIP-preprocessed view
                inputs_test, inputs_clip, _, tar_idx = next(iter_test)
            else:
                # oh-style: dataset returns ((img_base, img_clip), label, idx)
                (inputs_test, inputs_clip), _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            if is_imagenet:
                inputs_test, inputs_clip, _, tar_idx = next(iter_test)
            else:
                (inputs_test, inputs_clip), _, tar_idx = next(iter_test)

        inputs_test = inputs_test.cuda()
        inputs_clip = inputs_clip.cuda()

        if inputs_test.size(0) == 1:
            continue
        if cfg.SETTING.DATASET == 'office-home' and iter_num % (copy_iter) == 0:
            copy_netC(netC, netC_aux, cfg.class_num)
        if cfg.SETTING.DATASET == 'VISDA-C' and iter_num % (copy_iter // 8) == 0:
            copy_netC(netC, netC_aux, cfg.class_num)

        if iter_num % interval_iter == 0 and cfg.DIFF_ALIGN.CLS_PAR > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            netC_aux.eval()
            epoch_num = int(iter_num / interval_iter)
            # Inline CLIP evaluation: oh data = (img, label, idx), net data = (img_1, img_2, label, idx)
            start_clip_eval = True
            with torch.no_grad():
                clip_eval_iter = iter(dset_loaders["clip_test"])
                for _ in range(len(dset_loaders["clip_test"])):
                    clip_data = next(clip_eval_iter)
                    if is_imagenet:
                        clip_inputs = clip_data[1]
                        clip_labels = clip_data[2]
                    else:
                        clip_inputs = clip_data[0]
                        clip_labels = clip_data[1]
                    clip_inputs = clip_inputs.cuda()
                    if text_features is not None:
                        clip_score = clip_text(clip_model, text_features, clip_inputs)
                    else:
                        clip_score, _ = clip_model(clip_inputs, text_inputs)
                    clip_score = clip_score.cpu()
                    if start_clip_eval:
                        all_clip_score = clip_score.float().cpu()
                        all_label = clip_labels.float().cpu()
                        start_clip_eval = False
                    else:
                        all_label = torch.cat((all_label, clip_labels.float()), 0)
                        all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)
                all_clip_score = all_clip_score[:, :cfg.class_num]
                test_clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()
            target_score_bank = torch.zeros(num_sample).cuda()
            with torch.no_grad():
                test_iter = iter(dset_loaders["test"])
                for _ in range(len(dset_loaders["test"])):
                    test_data = next(test_iter)
                    test_inputs = test_data[0][0] if isinstance(test_data[0], tuple) else test_data[0]
                    test_indx = test_data[-1]
                    test_inputs = test_inputs.cuda()
                    test_outputs = netC(netB(netF(test_inputs)))
                    test_softmax = nn.Softmax(dim=1)(test_outputs)
                    test_scores = -torch.sum(test_softmax * torch.log(test_softmax + 1e-9), dim=1)
                    target_score_bank[test_indx] = test_scores
            netF.train()
            netB.train()
            netC.train()
            netC_aux.train()

        logit_clip_score, text_features = mypromt(inputs_clip)
        text_features_all = text_features
        logit_clip_score_include_hidden = clip_text(clip_model, text_features_all, inputs_clip)

        if 'RN' in cfg.DIFF_ALIGN.ARCH:
            prompt_lr = cfg.OPTIM.LR * 0.1
        else:
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

        source_score_bank[tar_idx] = 0.99 * source_score_bank[tar_idx] + 0.01 * target_score_bank[tar_idx]
        current_batch_scores = source_score_bank[tar_idx]
        avg_score = current_batch_scores.mean().item()

        diff = torch.abs(-torch.sum(base_softmax_out * torch.log(base_softmax_out + 1e-5), 1) - avg_score)
        diff = diff.detach().clone()

        margin = 0.01 * (1.01 ** (epoch_num - 1))
        loss_ent = entropy(base_softmax_out, avg_score, margin)
        mask_greater = diff > margin
        indices_greater = torch.where(mask_greater)[0]

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

        if cfg.DIFF_ALIGN.CLS_PAR > 0 and use_cls == True:
            targets = predict_mix[indices_greater]
            outputs_log_softmax = base_outputs_test[indices_greater]
            pseudo_label_loss = (- targets * outputs_log_softmax).sum(dim=1).mean()
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
            softmax_out[indices_greater].detach() if iter_num < warmup_iter else softmax_out[indices_greater],
            clip_all_output[indices_greater],
            bb,
            nnn,
            bn,
            lamb=1.0
        )

        m_base_sft = base_softmax_out.mean(dim=0)
        loss_iid = loss_aux
        if use_iid == False:
            loss_iid = torch.tensor(0.0).cuda()
        gentropy_loss = torch.sum(-m_base_sft * torch.log(m_base_sft + cfg.LCFD.EPSILON))

        with torch.no_grad():
            score_bank[tar_idx] = base_softmax_out.detach().clone()

        optimizer.zero_grad()
        opt_C.zero_grad()
        opt_aux.zero_grad()
        total_loss = classifier_loss + loss_iid - cfg.DIFF_ALIGN.GENT_PAR * gentropy_loss + cfg.DIFF_ALIGN.LENT_PAR * loss_ent
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

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if cfg.SETTING.DATASET == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, cfg, True)
                acc_s_te_aux, _ = cal_acc(dset_loaders['test'], netF, netB, netC_aux, cfg, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%,{:.2f}%;loss ={},iid={},cls={},gen={}'.format(
                    cfg.name, iter_num, max_iter, acc_s_te, acc_s_te_aux,
                    total_loss, loss_iid, classifier_loss, gentropy_loss) + '\n' + acc_list
                if iter_num == max_iter:
                    acc = acc_s_te
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, cfg, False)
                acc_s_te_aux, _ = cal_acc(dset_loaders['test'], netF, netB, netC_aux, cfg, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%,{:.2f}%;loss ={}'.format(
                    cfg.name, iter_num, max_iter, acc_s_te, acc_s_te_aux, total_loss)
                if iter_num == max_iter:
                    acc = acc_s_te

            logger.info(log_str)
            netF.train()
            netB.train()

    if cfg.ISSAVE:
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))
        torch.save(netC_aux.state_dict(), osp.join(cfg.output_dir, "target_C_aux_" + cfg.savename + ".pt"))

    return netF, netB, netC

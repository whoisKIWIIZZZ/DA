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
from src.utils.prompt_tuning_diff_new_optim_1130 import TextEncoder
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

def cal_clip_acc(loader, model, negative_text_features, N):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][1]
            labels = data[1]
            inputs = inputs.cuda()
            logits, _ = clip_forward_extended(model, inputs, negative_text_features)
            outputs = logits[:, :N]
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy * 100


def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


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


def create_negative_text_features(cfg, clip_model, class_num_hidden):
    # Read class names directly from file (avoid relying on cfg.classname being pre-set)
    classnames_for_negative_prompts = []
    with open(cfg.name_file) as f:
        for line in f:
            classnames_for_negative_prompts.extend([i for i in line.split()])
    classnames_for_negative_prompts = [name.replace("_", " ") for name in classnames_for_negative_prompts]
    negative_prompt_base_prefix = " a photo of not "
    negative_prompts_list = []
    for cls_name in classnames_for_negative_prompts:
        negative_prompt_string = f"{negative_prompt_base_prefix} {cls_name}"
        negative_prompts_list.append(negative_prompt_string)
    hidden_features = torch.cat([clip.tokenize(p) for p in negative_prompts_list]).cuda()
    text_encoder = TextEncoder(clip_model)
    with torch.no_grad():
        embedding = clip_model.token_embedding(hidden_features).type(clip_model.dtype)
        hidden_text_features = text_encoder(embedding, hidden_features)
        hidden_text_features = hidden_text_features / hidden_text_features.norm(dim=-1, keepdim=True)
    return hidden_text_features[:class_num_hidden, :]


def clip_forward_extended(model, inputs, negative_text_features):
    """
    CLIP forward that extends CoOp's N-class output to N+H classes.
    Returns (logits_NH, text_features_NH).
    When negative_text_features is None or empty, reverts to standard N-class output.
    """
    text_features_N = model.get_text_features()  # [N, D], has grad through prompt_learner
    if negative_text_features is not None and negative_text_features.shape[0] > 0:
        text_features_all = torch.cat([text_features_N, negative_text_features], dim=0)  # [N+H, D]
    else:
        text_features_all = text_features_N  # [N, D]
    text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        image_features = model.image_encoder(inputs.type(model.dtype))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features_all.t()
    return logits, text_features_all


def test_time_tuning_diff(model, inputs, optimizer, cfg, target_output,
                          hidden_text_features=None, hidden_optimizer=None, hidden_target=None):
    """
    TTA inner loop. When hidden_text_features is provided, also adapts H-class
    text features to match netC_aux's H-class output.
    """
    target_output = target_output.cuda()
    target_output = nn.Softmax(dim=1)(target_output)
    use_hidden_tta = hidden_text_features is not None

    for j in range(cfg.ProDe.TTA_STEPS):
        with torch.amp.autocast('cuda'):
            if use_hidden_tta:
                image_features = model.image_encoder(inputs.type(model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features_N = model.get_text_features()
                text_features_all = torch.cat([text_features_N, hidden_text_features], dim=0)
                text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits_all = logit_scale * image_features @ text_features_all.t()
                N_cls = text_features_N.shape[0]
                output_N = nn.Softmax(dim=1)(logits_all[:, :N_cls])
                output_H = nn.Softmax(dim=1)(logits_all[:, N_cls:])
                iic_loss_N = IID_losses.IID_loss_old(output_N, target_output)
                iic_loss_H = IID_losses.IID_loss_old(output_H, nn.Softmax(dim=1)(hidden_target))
                iic_loss = iic_loss_N + iic_loss_H
            else:
                output_logits, _ = model(inputs)
                output = nn.Softmax(dim=1)(output_logits)
                iic_loss = IID_losses.IID_loss_old(output, target_output)

        optimizer.zero_grad()
        if use_hidden_tta:
            hidden_optimizer.zero_grad()
        iic_loss.backward()
        optimizer.step()
        if use_hidden_tta:
            hidden_optimizer.step()

    if use_hidden_tta:
        with torch.no_grad():
            image_features = model.image_encoder(inputs.type(model.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_N = model.get_text_features()
            text_features_all = torch.cat([text_features_N, hidden_text_features], dim=0)
            text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits_all = logit_scale * image_features @ text_features_all.t()
            return nn.Softmax(dim=1)(logits_all[:, :text_features_N.shape[0]])
    else:
        return output


def test_time_adapt_eval_diff(inputs, model, optimizer, optim_state, cfg, target_output, negative_text_features, N,
                               hidden_text_features=None, hidden_optimizer=None, hidden_optim_state=None, hidden_target=None):
    optimizer.load_state_dict(optim_state)
    if hidden_optimizer is not None:
        hidden_optimizer.load_state_dict(hidden_optim_state)
    test_time_tuning_diff(model, inputs, optimizer, cfg, target_output,
                          hidden_text_features=hidden_text_features,
                          hidden_optimizer=hidden_optimizer,
                          hidden_target=hidden_target)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            model.eval()
            # Use learnable H features (updated by TTA) if available, else frozen
            h_features = hidden_text_features if hidden_text_features is not None else negative_text_features
            output_logits, _ = clip_forward_extended(model, inputs, h_features)
    # Return logits (same as ProDe). Pseudo-label fusion needs logits for
    # discriminative power; softmax would flatten the CLIP signal.
    return output_logits


def train_target(cfg):
    class_num_hidden = cfg.class_num_hidden
    class_num = cfg.class_num
    use_hidden = (class_num_hidden > 0)
    print("ProDe+diff: class_num={}, class_num_hidden={}, use_hidden={}".format(class_num, class_num_hidden, use_hidden))

    text_inputs = clip_pre_text(cfg)

    # Negative text features only when H>0
    if use_hidden:
        raw_clip_model, _, _ = clip.load(cfg.ProDe.ARCH)
        raw_clip_model.float()
        raw_clip_model.cuda()
        negative_text_features = create_negative_text_features(cfg, raw_clip_model, class_num_hidden)
        del raw_clip_model
        print("Created negative text features: shape =", negative_text_features.shape)
        # Learnable H-class text features for TTA (initialized from frozen negative features)
        hidden_text_features = nn.Parameter(negative_text_features.clone().detach())
    else:
        negative_text_features = None
        hidden_text_features = None

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
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    if use_hidden:
        netC_aux = network.feat_classifier(type='wn', class_num=cfg.class_num + class_num_hidden, bottleneck_dim=cfg.bottleneck).cuda()

    modelpath = cfg.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    # Initialize netC_aux only when H>0
    if use_hidden:
        old_fc = netC.fc
        new_fc = torch.nn.utils.weight_norm(nn.Linear(old_fc.in_features, cfg.class_num + class_num_hidden).cuda())
        with torch.no_grad():
            new_fc.weight_g.data[:cfg.class_num].copy_(old_fc.weight_g.data)
            new_fc.weight_v.data[:cfg.class_num].copy_(old_fc.weight_v.data)
            new_fc.bias.data[:cfg.class_num].copy_(old_fc.bias.data)
            if class_num_hidden > 0:
                # Init hidden class k as NEGATIVE of known class k.
                # weight_v[N+k] = -weight_v[k], bias[N+k] = -bias[k]
                # => hidden_logit[k] ≈ -known_logit[k] at init.
                # A "dog" feature naturally points AWAY from "not dog".
                num_copy = min(class_num_hidden, cfg.class_num)
                new_fc.weight_g.data[cfg.class_num:cfg.class_num+num_copy].copy_(old_fc.weight_g.data[:num_copy])
                new_fc.weight_v.data[cfg.class_num:cfg.class_num+num_copy].copy_(-old_fc.weight_v.data[:num_copy])
                new_fc.bias.data[cfg.class_num:cfg.class_num+num_copy].copy_(-old_fc.bias.data[:num_copy])
                # Extra hidden classes beyond N: random init with small negative bias
                if class_num_hidden > num_copy:
                    extra = class_num_hidden - num_copy
                    torch.nn.init.kaiming_uniform_(new_fc.weight_g.data[cfg.class_num+num_copy:], a=0, mode='fan_in', nonlinearity='linear')
                    new_fc.bias.data[cfg.class_num+num_copy:].fill_(-3.0)
        netC_aux.fc = new_fc

    netC.eval()

    param_group = []
    param_group_ib = []
    param_group_aux = []

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
    # netC is frozen (same as ProDe)
    for k, v in netC.named_parameters():
        v.requires_grad = False
    # netC_aux is trained only when H>0
    if use_hidden:
        for k, v in netC_aux.named_parameters():
            v.requires_grad = True
            param_group_aux += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]

    for k, v in model.prompt_learner.named_parameters():
        if v.requires_grad:
            param_group_ib += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    optimizer_ib = optim.SGD(param_group_ib)
    optimizer_ib = op_copy(optimizer_ib)
    optim_state = deepcopy(optimizer_ib.state_dict())

    if use_hidden:
        opt_aux = optim.SGD(param_group_aux)
        opt_aux = op_copy(opt_aux)
        hidden_optimizer_ib = optim.SGD([hidden_text_features], lr=cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1)
        hidden_optimizer_ib = op_copy(hidden_optimizer_ib)
        hidden_optim_state = deepcopy(hidden_optimizer_ib.state_dict())

    model.reset_classnames(cfg.classname, cfg.ProDe.ARCH)

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    copy_iter = max_iter // 15 // cfg.TEST.INTERVAL
    iter_num = 0
    num_sample = len(dset_loaders["target"].dataset)
    logtis_bank = torch.randn(num_sample, cfg.class_num).cuda()

    loader = dset_loaders["source"]
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
            indx = data[-1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            outputs = netC(output)
            outputs = nn.Softmax(dim=1)(outputs)
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

        # Periodic sync netC -> netC_aux (only when H>0)
        if use_hidden and iter_num % copy_iter == 0:
            copy_netC(netC, netC_aux, cfg.class_num)

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        if use_hidden:
            outputs_test_aux = netC_aux(features_test)
            softmax_out_aux = nn.Softmax(dim=1)(outputs_test_aux)

        with torch.no_grad():
            outputs_test_new = outputs_test.clone().detach()
            if use_hidden:
                outputs_test_aux_new = outputs_test_aux.clone().detach()

        # CLIP input selection: always use aug view (CLIP preprocessing)
        inputs_test_clip = inputs_test_augs

        # Prepare H-class TTA target (raw logits from netC_aux H dims)
        hidden_target = None
        if use_hidden:
            hidden_target = outputs_test_aux_new[:, class_num:]

        # clip_score is logits (same as ProDe test_time_adapt_eval).
        # Pseudo-label fusion needs logits for discriminative power.
        clip_score = test_time_adapt_eval_diff(
            inputs_test_clip, model, optimizer_ib, optim_state, cfg,
            outputs_test_new, negative_text_features, class_num,
            hidden_text_features=hidden_text_features,
            hidden_optimizer=hidden_optimizer_ib,
            hidden_optim_state=hidden_optim_state,
            hidden_target=hidden_target
        )
        clip_score = clip_score.float()

        # N-class logits for pseudo-label fusion
        clip_logits_N = clip_score[:, :class_num] if use_hidden else clip_score

        # Pseudo-label fusion (same as ProDe: softmax - softmax + logits)
        with torch.no_grad():
            new_clip = (outputs_test_new - 1 * logtis_bank[tar_idx].cuda()) + clip_logits_N.cuda()
            clip_score_sm = nn.Softmax(dim=1)(new_clip)

        _, clip_index_new = torch.max(new_clip, 1)

        if use_hidden:
            # Separate softmax for N and H (prevent hidden from diluting known)
            score_N = nn.Softmax(dim=1)(clip_score[:, :class_num])
            score_H = nn.Softmax(dim=1)(clip_score[:, class_num:])
            # Hybrid target: N-class fused pseudo-label + H-class CLIP softmax
            clip_score_hybrid = torch.cat([clip_score_sm, score_H], dim=1)
            iic_loss, _ = IID_losses.IID_loss(
                class_num,
                softmax_out_aux,
                clip_score_hybrid,
                w_base=1.0,
                w_aux=1.0,
                bn=-0.3,
                lamb=1.0
            )
        else:
            # H=0: exact ProDe IID loss — netC output vs fused pseudo-label
            iic_loss = IID_losses.IID_loss_old(softmax_out, clip_score_sm)

        classifier_loss = cfg.ProDe.IIC_PAR * iic_loss

        msoftmax = softmax_out.mean(dim=0)
        if cfg.SETTING.DATASET == 'office':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if cfg.SETTING.DATASET == 'office-home':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if cfg.SETTING.DATASET == 'VISDA-C':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 0.1 * gentropy_loss

        pred = clip_index_new
        entropy_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        classifier_loss = cfg.ProDe.GENT_PAR * entropy_loss + classifier_loss

        optimizer.zero_grad()
        optimizer_ib.zero_grad()
        if use_hidden:
            opt_aux.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        optimizer_ib.step()
        if use_hidden:
            opt_aux.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            model.eval()
            clip_acc = cal_clip_acc(dset_loaders['test'], model, negative_text_features, class_num)
            if cfg.SETTING.DATASET == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%; CLIP_Acc = {:.2f}%; loss = {}'.format(cfg.name, iter_num, max_iter, acc_s_te, clip_acc, classifier_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%; CLIP_Acc = {:.2f}%; loss = {}'.format(cfg.name, iter_num, max_iter, acc_s_te, clip_acc, classifier_loss)

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

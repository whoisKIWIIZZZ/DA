import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import loss,prompt_tuning,IID_losses
from src.models import network,shot_model
from sklearn.metrics import confusion_matrix
import clip
from clip.custom_clip import get_coop
from src.utils.utils import *
from src.data.datasets.data_loading import get_test_loader, get_test_loader_aug
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK,IMAGENET_V_MASK
from src.models.model import *
from data.imagnet_prompts import imagenet_classes
from src.utils.prompt_tuning_diff_new_optim_1130 import TextEncoder

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, base_model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = base_model(inputs)
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
            inputs = data[0]
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


def copy_netC_aux(base_model, netC_aux_fc, num_known_classes):
    old_fc = base_model.fc
    with torch.no_grad():
        netC_aux_fc.weight_g.data[:num_known_classes].copy_(old_fc.weight_g.data)
        netC_aux_fc.weight_v.data[:num_known_classes].copy_(old_fc.weight_v.data)
        netC_aux_fc.bias.data[:num_known_classes].copy_(old_fc.bias.data)


def create_negative_text_features(cfg, clip_model, class_num_hidden):
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
    text_features_N = model.get_text_features()
    if negative_text_features is not None and negative_text_features.shape[0] > 0:
        text_features_all = torch.cat([text_features_N, negative_text_features], dim=0)
    else:
        text_features_all = text_features_N
    text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        image_features = model.image_encoder(inputs.type(model.dtype))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features_all.t()
    return logits, text_features_all


def test_time_tuning_diff(model, inputs, optimizer, cfg, target_output,
                          hidden_text_features=None, hidden_optimizer=None, hidden_target=None):
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
            return logits_all
    else:
        return output


def test_time_adapt_eval_diff(inputs, model, optimizer, optim_state, cfg, target_output, negative_text_features, N,
                               hidden_text_features=None, hidden_optimizer=None, hidden_optim_state=None, hidden_target=None):
    optimizer.load_state_dict(optim_state)
    if hidden_optimizer is not None:
        hidden_optimizer.load_state_dict(hidden_optim_state)
    logits = test_time_tuning_diff(model, inputs, optimizer, cfg, target_output,
                                   hidden_text_features=hidden_text_features,
                                   hidden_optimizer=hidden_optimizer,
                                   hidden_target=hidden_target)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            model.eval()
            h_features = hidden_text_features if hidden_text_features is not None else negative_text_features
            output_logits, _ = clip_forward_extended(model, inputs, h_features)
    return output_logits


def train_target(cfg):
    class_num_hidden = cfg.class_num_hidden
    class_num = cfg.class_num
    use_hidden = (class_num_hidden > 0)
    print("ProDe+diff (net): class_num={}, class_num_hidden={}, use_hidden={}".format(class_num, class_num_hidden, use_hidden))

    text_inputs = clip_pre_text(cfg)

    # Negative text features only when H>0
    if use_hidden:
        raw_clip_model, _, _ = clip.load(cfg.ProDe.ARCH)
        raw_clip_model.float()
        raw_clip_model.cuda()
        negative_text_features = create_negative_text_features(cfg, raw_clip_model, class_num_hidden)
        del raw_clip_model
        print("Created negative text features: shape =", negative_text_features.shape)
        hidden_text_features = nn.Parameter(negative_text_features.clone().detach())
    else:
        negative_text_features = None
        hidden_text_features = None

    model = get_coop(cfg.ProDe.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.ProDe.N_CTX, cfg.ProDe.CTX_INIT)

    if 'image' in cfg.SETTING.DATASET:
        if cfg.MODEL.ARCH[0:3] == 'res':
            netF = network.ResBase(res_name=cfg.MODEL.ARCH)
        elif cfg.MODEL.ARCH[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH)
        netC = network.Net2(2048,1000)
        base_model = get_model(cfg, cfg.class_num)
        netC.linear.load_state_dict(base_model.model.fc.state_dict())
        del base_model
        Shot_model = shot_model.OfficeHome_Shot(netF,netC)
        base_model = Shot_model
        if cfg.SETTING.DATASET == "imagenet_a":
            base_model = ImageNetXWrapper(base_model, IMAGENET_A_MASK)
        elif cfg.SETTING.DATASET == "imagenet_r":
            base_model = ImageNetXWrapper(base_model, IMAGENET_R_MASK)
        elif cfg.SETTING.DATASET == "imagenet_d109":
            base_model = ImageNetXWrapper(base_model, IMAGENET_D109_MASK)
        elif cfg.SETTING.DATASET == "imagenet_v":
            base_model = ImageNetXWrapper(base_model, IMAGENET_V_MASK)
    else :
        base_model = get_model(cfg, cfg.class_num)
    base_model = base_model.cuda()

    # ---- netC_aux: N+H classifier sharing encoder ----
    if use_hidden:
        old_fc = base_model.fc
        netC_aux_fc = torch.nn.utils.weight_norm(
            nn.Linear(old_fc.in_features, class_num + class_num_hidden).cuda())
        with torch.no_grad():
            netC_aux_fc.weight_g.data[:class_num].copy_(old_fc.weight_g.data)
            netC_aux_fc.weight_v.data[:class_num].copy_(old_fc.weight_v.data)
            netC_aux_fc.bias.data[:class_num].copy_(old_fc.bias.data)
            num_copy = min(class_num_hidden, class_num)
            netC_aux_fc.weight_g.data[class_num:class_num+num_copy].copy_(old_fc.weight_g.data[:num_copy])
            netC_aux_fc.weight_v.data[class_num:class_num+num_copy].copy_(-old_fc.weight_v.data[:num_copy])
            netC_aux_fc.bias.data[class_num:class_num+num_copy].copy_(-old_fc.bias.data[:num_copy])
            if class_num_hidden > num_copy:
                extra = class_num_hidden - num_copy
                torch.nn.init.kaiming_uniform_(netC_aux_fc.weight_g.data[class_num+num_copy:], a=0, mode='fan_in', nonlinearity='linear')
                netC_aux_fc.bias.data[class_num+num_copy:].fill_(-3.0)

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    param_group = []
    param_group_ib = []
    param_group_aux = []

    for k, v in base_model.named_parameters():
        if 'netC' in k or 'fc' in k:
            v.requires_grad = False
        else:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR}]
    if use_hidden:
        for k, v in netC_aux_fc.named_parameters():
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

    classnames = cfg.classname
    model.reset_classnames(classnames, cfg.ProDe.ARCH)

    cfg.ADAPTATION = 'tent'
    domain_name = cfg.domain[cfg.SETTING.T]
    target_data_loader = get_test_loader_aug(adaptation=cfg.ADAPTATION,
                                        dataset_name=cfg.SETTING.DATASET,
                                        root_dir=cfg.DATA_DIR,
                                        domain_name=domain_name,
                                        rng_seed=cfg.SETTING.SEED,
                                        batch_size=cfg.TEST.BATCH_SIZE,
                                        shuffle=True,
                                        workers=cfg.NUM_WORKERS)

    test_data_loader = get_test_loader(adaptation=cfg.ADAPTATION,
                                    dataset_name=cfg.SETTING.DATASET,
                                    root_dir=cfg.DATA_DIR,
                                    domain_name=domain_name,
                                    rng_seed=cfg.SETTING.SEED,
                                    batch_size=cfg.TEST.BATCH_SIZE*3,
                                    shuffle=False,
                                    workers=cfg.NUM_WORKERS)

    source_data_loader = get_test_loader(adaptation=cfg.ADAPTATION,
                                        dataset_name=cfg.SETTING.DATASET,
                                        root_dir=cfg.DATA_DIR,
                                        domain_name=domain_name,
                                        rng_seed=cfg.SETTING.SEED,
                                        batch_size=cfg.TEST.BATCH_SIZE,
                                        shuffle=False,
                                        workers=cfg.NUM_WORKERS)

    num_sample=len(target_data_loader.dataset)
    logtis_bank = torch.randn(num_sample, cfg.class_num).cuda()

    with torch.no_grad():
        iter_test = iter(source_data_loader)
        for i in range(len(source_data_loader)):
            data = next(iter_test)
            inputs = data[0]
            indx=data[-1]
            inputs = inputs.cuda()
            outputs = base_model(inputs)
            outputs=nn.Softmax(dim=1)(outputs)
            logtis_bank[indx] = outputs.detach().clone()

    max_iter = cfg.TEST.MAX_EPOCH * len(target_data_loader)
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, inputs_test_augs, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
            inputs_test, inputs_test_augs, _, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        inputs_test_augs = inputs_test_augs.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        # Forward through shared encoder
        feat = base_model.encoder(inputs_test)
        feat = torch.flatten(feat, 1)
        outputs_test = base_model.fc(feat)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        if use_hidden:
            outputs_test_aux = netC_aux_fc(feat)
            softmax_out_aux = nn.Softmax(dim=1)(outputs_test_aux)

        with torch.no_grad():
            outputs_test_new = outputs_test.clone().detach()
            if use_hidden:
                outputs_test_aux_new = outputs_test_aux.clone().detach()

        # CLIP always uses aug view (CLIP preprocessing)
        inputs_test_clip = inputs_test_augs

        hidden_target = None
        if use_hidden:
            hidden_target = outputs_test_aux_new[:, class_num:]

        clip_score = test_time_adapt_eval_diff(
            inputs_test_clip, model, optimizer_ib, optim_state, cfg,
            outputs_test_new, negative_text_features, class_num,
            hidden_text_features=hidden_text_features,
            hidden_optimizer=hidden_optimizer_ib,
            hidden_optim_state=hidden_optim_state,
            hidden_target=hidden_target
        )
        clip_score = clip_score.float()

        clip_logits_N = clip_score[:, :class_num] if use_hidden else clip_score

        with torch.no_grad():
            new_clip = (outputs_test_new - 1*logtis_bank[tar_idx].cuda()) + clip_logits_N.cuda()
            clip_score_sm = nn.Softmax(dim=1)(new_clip)

        _, clip_index_new = torch.max(new_clip, 1)

        if use_hidden:
            score_N = nn.Softmax(dim=1)(clip_score[:, :class_num])
            score_H = nn.Softmax(dim=1)(clip_score[:, class_num:])
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
        if cfg.SETTING.DATASET == 'domainnet126':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 0.5 * gentropy_loss

        pred = clip_index_new
        entropy_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        classifier_loss = cfg.ProDe.GENT_PAR*entropy_loss + classifier_loss

        optimizer.zero_grad()
        optimizer_ib.zero_grad()
        if use_hidden:
            opt_aux.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        optimizer_ib.step()
        if use_hidden:
            opt_aux.step()

        # Periodic sync: copy base_model.fc -> netC_aux_fc (known class weights)
        if use_hidden and iter_num % (max_iter // 15 // cfg.TEST.INTERVAL) == 0:
            copy_netC_aux(base_model, netC_aux_fc, class_num)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_model.eval()
            model.eval()
            clip_acc = cal_clip_acc(test_data_loader, model, negative_text_features, class_num)
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(test_data_loader, base_model, True)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%; CLIP_Acc = {:.2f}%; loss = {}'.format(cfg.name, iter_num, max_iter, acc_s_te, clip_acc, classifier_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(test_data_loader, base_model, False)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%; CLIP_Acc = {:.2f}%; loss = {}'.format(cfg.name, iter_num, max_iter, acc_s_te, clip_acc, classifier_loss)

            logger.info(log_str)
            base_model.train()
            model.train()

    if cfg.ISSAVE:
        torch.save(base_model.state_dict(), osp.join(cfg.output_dir, "target" + cfg.savename + ".pt"))

    return base_model


def clip_pre_text(cfg):
    List_rd = []
    if 'image' in cfg.SETTING.DATASET:
        classnames_all = imagenet_classes
        classnames = []
        if cfg.SETTING.DATASET.split('_')[-1] in ['a','r','v']:
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
        f.close()
        classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.ProDe.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts

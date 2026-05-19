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
def analyze_ensemble(cfg, netF, netB, netC, model, class_sim, dset_loaders):
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

    # --- netC_proxy: N+N classifier for IID loss ---
    netC_proxy = network.feat_classifier(type='wn', class_num=cfg.class_num * 2, bottleneck_dim=cfg.bottleneck).cuda()
    with torch.no_grad():
        netC_proxy.fc.weight_g.data[:cfg.class_num].copy_(netC.fc.weight_g.data)
        netC_proxy.fc.weight_v.data[:cfg.class_num].copy_(netC.fc.weight_v.data)
        netC_proxy.fc.bias.data[:cfg.class_num].copy_(netC.fc.bias.data)
        torch.nn.init.kaiming_uniform_(netC_proxy.fc.weight_g.data[cfg.class_num:], a=0, mode='fan_in', nonlinearity='linear')
        netC_proxy.fc.bias.data[cfg.class_num:].fill_(-3.0)

    # --- Text-guided semantic similarity matrix ---
    model.reset_classnames(cfg.classname, cfg.ProDe.ARCH)
    with torch.no_grad():
        text_feats = model.get_text_features()
        class_sim = text_feats @ text_feats.T
        class_sim.fill_diagonal_(0)
        class_sim = class_sim / (class_sim.sum(dim=1, keepdim=True) + 1e-6)

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

        clip_score = test_time_adapt_eval(inputs_test_clip, model, optimizer_ib, optim_state, cfg, outputs_test_new)
        clip_score = clip_score.float()

        with torch.no_grad():
            # Text-guided negative prior: log-space residu
            p = torch.softmax(clip_score, dim=1)
            neg_prior = p  @ class_sim
            res = neg_prior - p

            residual = torch.log((p - 0.1 * res).clamp(min=1e-6)) - torch.log(p.clamp(min=1e-6))
            # Scale log-prob residual to logit space (logit_scale ≈ 100 for ViT-B/32)
            residual = residual * model.logit_scale.exp()

            term_a = outputs_test_new - logtis_bank[tar_idx].cuda()
            term_b = clip_score.cuda()
            term_c = cfg.ProDe.PROXY_WEIGHT * residual
            new_clip = term_a + term_b + term_c

            if iter_num % 50 == 1:
                logger.info(
                    'term_a(base-bank): mean={:.2f} std={:.2f} | '
                    'term_b(clip_score): mean={:.2f} std={:.2f} | '
                    'term_c(w*residual): mean={:.3f} std={:.3f} | '
                    'p max/mean={:.3f}/{:.3f} | '
                    'neg_prior max/mean={:.3f}/{:.3f}'.format(
                        term_a.mean().item(), term_a.std().item(),
                        term_b.mean().item(), term_b.std().item(),
                        term_c.mean().item(), term_c.std().item(),
                        p.max().item(), p.mean().item(),
                        neg_prior.max().item(), neg_prior.mean().item()))
        clip_score_sm = nn.Softmax(dim=1)(new_clip)
        _, clip_index_new = torch.max(new_clip, 1)

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
            # Recompute class_sim with updated prompt
            with torch.no_grad():
                text_feats = model.get_text_features()
                class_sim = text_feats @ text_feats.T
                class_sim.fill_diagonal_(0)
                class_sim = class_sim / (class_sim.sum(dim=1, keepdim=True) + 1e-6)
            analyze_ensemble(cfg, netF, netB, netC, model, class_sim, dset_loaders)
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te, classifier_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te, classifier_loss)

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

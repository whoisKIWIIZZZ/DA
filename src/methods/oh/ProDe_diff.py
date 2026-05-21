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
            with torch.no_grad():
                img_feats = model.image_encoder(inputs_test_clip.type(model.dtype))
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            neg_logits = model.logit_scale.exp() * img_feats @ neg_embed.t()

            term_a = outputs_test_new - logtis_bank[tar_idx].cuda()
            term_b = clip_score.cuda()
            new_clip = term_a + term_b - cfg.ProDe.NEG_ALPHA * neg_logits.detach()

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
            #analyze_ensemble(cfg, netF, netB, netC, model, dset_loaders)
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

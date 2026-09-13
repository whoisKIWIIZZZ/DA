#prode,py
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
from sklearn.cluster import KMeans
from clip.custom_clip import get_coop
import clip
from src.utils.utils import *
from src.utils.adaptive_alpha import (adaptive_alpha_from_logits,
                                      disagreement_diagnostics,
                                      gate_diagnostics, TargetLogitBank)
logger = logging.getLogger(__name__)



def data_load(cfg): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
    txt_src = open(cfg.s_dset_path).readlines()
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

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()
    dsets["target"] = ImageList_idx_aug_fix_old(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx_aug_fix_old(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source"] = ImageList_idx_aug_fix_old(txt_test, transform=image_test())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=2, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True)
    dsets["source_test"] = ImageList_idx_aug_fix_old(txt_src, transform=image_test())
    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size=train_bs, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #else:
    #normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
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
  #else:
   # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
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

def cal_acc(loader, netF, netB, netC, cfg, flag=False, evaluate_open_set=True):
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

    probabilities = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(probabilities, 1)
    if cfg.DA == 'open' and evaluate_open_set:
        entropy = torch.sum(
            -probabilities * torch.log(probabilities + cfg.ProDe.EPSILON), dim=1
        ) / np.log(cfg.class_num)
        centers = np.array([[0.0], [1.0]])
        threshold = KMeans(
            n_clusters=2, random_state=0, init=centers, n_init=1
        ).fit(entropy.numpy().reshape(-1, 1)).cluster_centers_.mean()
        predict[entropy > threshold] = cfg.class_num

        labels = np.arange(cfg.class_num + 1)
        matrix = confusion_matrix(all_label.numpy(), predict.numpy(), labels=labels)
        per_class_acc = np.divide(
            matrix.diagonal(), matrix.sum(axis=1),
            out=np.zeros(cfg.class_num + 1, dtype=float),
            where=matrix.sum(axis=1) != 0,
        ) * 100
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        os_acc = per_class_acc.mean()
        h_score = 0.0
        if known_acc + unknown_acc > 0:
            h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)
        return os_acc, known_acc, unknown_acc, h_score

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(probabilities)).cpu().data.item()

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


def test_on_source(cfg, netF, netB, netC, dset_loaders):
    ## evaluate the adapted model on the source split (GSFDA)
    netF.eval()
    netB.eval()
    netC.eval()
    if cfg.SETTING.DATASET == 'VISDA-C':
        acc_s, acc_list = cal_acc(
            dset_loaders['source_test'], netF, netB, netC, cfg, True,
            evaluate_open_set=False,
        )
        log_str = 'GSFDA: Task: {}, Source Accuracy = {:.2f}%'.format(cfg.name_src, acc_s) + '\n' + acc_list
    else:
        acc_s, _ = cal_acc(
            dset_loaders['source_test'], netF, netB, netC, cfg, False,
            evaluate_open_set=False,
        )
        log_str = 'GSFDA: Task: {}, Source Accuracy = {:.2f}%'.format(cfg.name_src, acc_s)
    logger.info(log_str)
    netF.train()
    netB.train()
    netC.train()
    return acc_s


def test_time_tuning(model, inputs, optimizer, cfg, target_output):
    target_output = target_output.cuda()
    target_output =  nn.Softmax(dim=1)(target_output)
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
    if cfg.GSFDA:
        print("GSFDA,test on source")
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
    # for k, v in netC.named_parameters():
    #     if cfg.OPTIM.LR_DECAY1 > 0:
    #         param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
    #     else:
    #         v.requires_grad = False
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
    alpha_observation_bank = torch.zeros(num_sample, device=logtis_bank.device)
    alpha_bank = torch.zeros(num_sample, device=logtis_bank.device)
    alpha_seen = torch.zeros(num_sample, dtype=torch.bool, device=logtis_bank.device)
    prior_bank = torch.zeros(num_sample, device=logtis_bank.device)
    target_confidence_bank = torch.zeros(num_sample, device=logtis_bank.device)
    clip_confidence_bank = torch.zeros(num_sample, device=logtis_bank.device)
    target_stability_bank = torch.zeros(num_sample, device=logtis_bank.device)
    clip_stability_bank = torch.zeros(num_sample, device=logtis_bank.device)
    prior_term_bank = torch.zeros(num_sample, device=logtis_bank.device)
    reliability_term_bank = torch.zeros(num_sample, device=logtis_bank.device)
    loss_term_bank = torch.zeros(num_sample, device=logtis_bank.device)
    loss_gap_bank = torch.zeros(num_sample, device=logtis_bank.device)
    target_loss_bank = torch.zeros(num_sample, device=logtis_bank.device)
    clip_loss_bank = torch.zeros(num_sample, device=logtis_bank.device)
    target_prediction_bank = torch.full((num_sample,), -1, dtype=torch.long, device=logtis_bank.device)
    clip_prediction_bank = torch.full((num_sample,), -1, dtype=torch.long, device=logtis_bank.device)
    selected_prediction_bank = torch.full((num_sample,), -1, dtype=torch.long, device=logtis_bank.device)
    label_bank = torch.full((num_sample,), -1, dtype=torch.long, device=logtis_bank.device)
    alpha_sum = 0.0
    alpha_count = 0
    logger.info('ProDe adaptive alpha: %s', int(cfg.ProDe.USE_ADAPTIVE_ALPHA) == 1)

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
            #outputs=nn.Softmax(dim=1)(outputs)
            logtis_bank[indx] = outputs.detach().clone()

    # Independent target logit history bank.  Unlike the source bank above
    # (which stays fixed), this one directly stores the most recent target
    # logits per sample -- no EMA sliding update.  Seeded with the source
    # logits so the first visit of each sample has a valid history; the
    # source bank itself is never modified.
    target_history_bank = TargetLogitBank(
        num_sample, cfg.class_num, logtis_bank.device, init_logits=logtis_bank
    )

    if cfg.SETTING.DATASET == "office":
        print("HOW")
    while iter_num < max_iter:
        try:
            (inputs_test, inputs_test_augs), labels, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, inputs_test_augs), labels, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        inputs_test_augs = inputs_test_augs.cuda()

        if cfg.SETTING.DATASET == "office":
            inputs_test_augs = inputs_test #give clip right aug in different datasets

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        with torch.no_grad():
            outputs_test_new = outputs_test.clone().detach()

        clip_score = test_time_adapt_eval(inputs_test_augs, model, optimizer_ib, optim_state, cfg, outputs_test_new)
        clip_score = clip_score.float()

        with torch.no_grad():
            if int(getattr(cfg.ProDe, 'USE_ADAPTIVE_ALPHA', 0)) == 1:
                details = adaptive_alpha_from_logits(
                    outputs_test_new,
                    clip_score,
                    cfg,
                    target_history_logits=target_history_bank[tar_idx],
                    observation_count=alpha_observation_bank[tar_idx],
                    step=iter_num,
                    max_step=max_iter,
                    section_name='ProDe', return_details=True
                )
                (alpha, prior, target_confidence, clip_confidence,
                 target_stability, clip_stability, prior_term,
                 reliability_term, loss_term, loss_gap, observation_count,
                 target_loss, clip_loss) = details
                new_clip = (
                     alpha.unsqueeze(1)*outputs_test_new
                    + (1.0 - alpha.unsqueeze(1)) * clip_score.cuda()
                    - 0.1 * logtis_bank[tar_idx].cuda()
                )
                alpha_sum += alpha.sum().item()
                alpha_count += alpha.numel()
            else:
                new_clip = (outputs_test_new - 0.1*logtis_bank[tar_idx].cuda()) + clip_score.cuda()
                alpha = outputs_test_new.new_full((outputs_test_new.size(0),), 0.5)

            if int(getattr(cfg.ProDe, 'USE_ADAPTIVE_ALPHA', 0)) == 1:
                alpha_bank[tar_idx] = alpha
                alpha_seen[tar_idx] = True
                prior_bank[tar_idx] = prior
                target_confidence_bank[tar_idx] = target_confidence
                clip_confidence_bank[tar_idx] = clip_confidence
                target_stability_bank[tar_idx] = target_stability
                clip_stability_bank[tar_idx] = clip_stability
                prior_term_bank[tar_idx] = prior_term
                reliability_term_bank[tar_idx] = reliability_term
                loss_term_bank[tar_idx] = loss_term
                loss_gap_bank[tar_idx] = loss_gap
                target_loss_bank[tar_idx] = target_loss
                clip_loss_bank[tar_idx] = clip_loss
            target_prediction_bank[tar_idx] = outputs_test_new.argmax(dim=1)
            clip_prediction_bank[tar_idx] = clip_score.argmax(dim=1)
            selected_prediction_bank[tar_idx] = new_clip.argmax(dim=1)
            label_bank[tar_idx] = labels.to(label_bank.device).long()
        #     for name, t in [
   		#  ("outputs_test_new",   outputs_test_new),
   		#  ("logits_bank term",   1 * logtis_bank[tar_idx].cuda()),
    	# 	 ("clip_score",         clip_score.cuda()),
		# ]:
        #         print(f"{name:20s} | mean_abs={t.abs().mean():.3e} | max_abs={t.abs().max():.3e}")
            clip_score_sm = nn.Softmax(dim=1)(new_clip)
        
        _,clip_index_new = torch.max(new_clip, 1)
        clip_score_sm = nn.Softmax(dim=1)(new_clip)
        iic_loss = IID_losses.IID_loss_old(softmax_out, clip_score_sm)
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
        #optimizer_ib.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        #optimizer_ib.step()
        with torch.no_grad():
            # Directly store the current target logits as the new history
            # (no EMA).  The source bank logtis_bank is left untouched.
            target_history_bank.update(tar_idx, outputs_test_new)
            alpha_observation_bank[tar_idx] += 1
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, cfg, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list
            else:
                metrics = cal_acc(dset_loaders['test'], netF, netB, netC, cfg, False)
                if cfg.DA == 'open':
                    os_acc, known_acc, unknown_acc, h_score = metrics
                    log_str = ('Task: {}, Iter:{}/{}; OS={:.2f}%; OS*={:.2f}%; '
                               'UNK={:.2f}%; H={:.2f}%; loss={}').format(
                        cfg.name, iter_num, max_iter, os_acc, known_acc,
                        unknown_acc, h_score, classifier_loss)
                else:
                    acc_s_te, _ = metrics
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss)

            if alpha_count > 0:
                log_str += '; alpha_mean={:.4f}'.format(alpha_sum / alpha_count)
            if int(getattr(cfg.ProDe, 'USE_ADAPTIVE_ALPHA', 0)) == 1:
                log_str += '\n' + gate_diagnostics(
                    alpha_bank, alpha_seen, prior_bank, target_prediction_bank,
                    clip_prediction_bank, label_bank, target_confidence_bank,
                    clip_confidence_bank, target_stability_bank, clip_stability_bank,
                    prior_term_bank, reliability_term_bank, loss_term_bank,
                    loss_gap_bank, target_loss_bank, clip_loss_bank,
                    alpha_observation_bank, cfg, section_name='ProDe',
                    selected_prediction=selected_prediction_bank
                )
            else:
                log_str += '\nGate: adaptive alpha disabled; ' + disagreement_diagnostics(
                    target_prediction_bank, clip_prediction_bank,
                    selected_prediction_bank, label_bank
                )

            logger.info(log_str)
            netF.train()
            netB.train()
            netC.train()

    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))

    if cfg.GSFDA:
        test_on_source(cfg, netF, netB, netC, dset_loaders)

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
    classnames = List_rd[:cfg.class_num] if cfg.DA == 'open' else List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.ProDe.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts


import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import itertools

# --- Imports from 'net' version and 'oh' version ---
from src.utils import loss, IID_losses
from src.models import network
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from clip.custom_clip import get_coop
from src.utils.utils import *
from src.data.datasets.data_loading import get_test_loader
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK,IMAGENET_V_MASK
from src.models.model import *
from data.imagnet_prompts import imagenet_classes
import clip

logger = logging.getLogger(__name__)

# --- Helper functions from 'net' version ---
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
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
        # Fallback for other datasets like office-home, visda
        name_file_path = os.path.join(cfg.FOLDER, cfg.SETTING.DATASET, 'classname.txt')
        with open(name_file_path) as f:
            classnames = [line.strip() for line in f.readlines()]

    classnames = [name.replace(" ", " ") for name in classnames]
    cfg.classname = classnames
    return classnames

# --- Main training function, merged ---
def train_target(cfg):
    # 1. ========== Data Loading (from 'net' version) ========== 
    domain_name = cfg.domain[cfg.SETTING.T]
    dset_loaders = {}
    dset_loaders["target"] = get_test_loader(adaptation=cfg.ADAPTATION,
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
                                    batch_size=cfg.TEST.BATCH_SIZE*3,
                                    shuffle=False,
                                    workers=cfg.NUM_WORKERS)

    # 2. ========== Model Setup (merged) ========== 
    classnames = clip_pre_text(cfg)
    cfg.classname = classnames
    
    # --- Main Model (F/B/C structure from 'oh' version) ---
    # We will consistently use the F/B/C structure as it's what the core logic expects
    if cfg.MODEL.ARCH.startswith('res'):
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH.startswith('vgg'):
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()
    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
    
    # Load source weights if available
    if cfg.output_dir_src:
        modelpath = cfg.output_dir_src + '/source_F.pt'   
        netF.load_state_dict(torch.load(modelpath))
        modelpath = cfg.output_dir_src + '/source_B.pt'   
        netB.load_state_dict(torch.load(modelpath))
        modelpath = cfg.output_dir_src + '/source_C.pt'    
        netC.load_state_dict(torch.load(modelpath))

    # --- CLIP Model with Prompt Learner (from 'oh' version) ---
    clip_model_prompt = get_coop(cfg.DIFO.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.DIFO.N_CTX, cfg.DIFO.CTX_INIT).cuda()
    clip_model_prompt.reset_classnames(classnames, cfg.DIFO.ARCH)

    # 3. ========== Optimizer Setup (from 'oh' version) ========== 
    # Freeze parts of the models
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
        
    for name, param in clip_model_prompt.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    param_group = [
        {'params': netF.parameters(), 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1},
        {'params': netB.parameters(), 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2},
        {'params': clip_model_prompt.prompt_learner.parameters(), 'lr': cfg.OPTIM.LR * 0.1},
    ]
    
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    # 4. ========== Training Loop (from 'oh' version, adapted) ========== 
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    
    score_bank = torch.randn(len(dset_loaders["target"].dataset), cfg.class_num).cuda()

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        # Batch size check from our previous fix
        batch_size = inputs_test[0].size(0) if isinstance(inputs_test, list) else inputs_test.size(0)
        if batch_size == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        # Handle case where data loader returns a list of augmentations
        if isinstance(inputs_test, list):
            inputs_test = inputs_test[0].cuda()
        else:
            inputs_test = inputs_test.cuda()
        
        # --- a. Target Model Forward Pass ---
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        # --- b. CLIP & Prompt Forward Pass ---
        with torch.no_grad():
            image_features_clip = clip_model_prompt.image_encoder(inputs_test.type(clip_model_prompt.dtype))
            image_features_clip_norm = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)
        
        prompts = clip_model_prompt.prompt_learner()
        tokenized_prompts = clip_model_prompt.prompt_learner.tokenized_prompts
        text_features_clip = clip_model_prompt.text_encoder(prompts, tokenized_prompts)
        text_features_clip_norm = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
        
        logits_clip = 100. * image_features_clip_norm @ text_features_clip_norm.t()
        clip_output_softmax = nn.Softmax(dim=1)(logits_clip)

        # --- c. Loss Calculation ---
        with torch.no_grad():
            K = softmax_out.size(1)
            _, predict_bank = torch.max(score_bank[tar_idx], 1)
            _, predict_clip = torch.max(clip_output_softmax, 1)
            
            predict_one = torch.nn.functional.one_hot(predict_bank, K).float().cuda()
            clip_one = torch.nn.functional.one_hot(predict_clip, K).float().cuda()
            
            mix_ratio = 0.5 
            mixed_targets = mix_ratio * predict_one + (1 - mix_ratio) * clip_one
        
        loss_soft = (- mixed_targets * torch.log_softmax(outputs_test, dim=1)).sum(dim=1)
        classifier_loss = loss_soft.mean() * cfg.DIFO.CLS_PAR

        iic_loss = IID_losses.IID_loss(softmax_out, clip_output_softmax) * cfg.DIFO.IIC_PAR
        
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.LCFD.EPSILON)) * cfg.DIFO.GENT_PAR
        
        total_loss = classifier_loss + iic_loss - gentropy_loss
        
        with torch.no_grad():
            score_bank[tar_idx] = softmax_out.detach().clone()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            eval_model = nn.Sequential(netF, netB, netC)
            eval_model.eval()
            clip_model_prompt.eval()
            
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], eval_model, True)
                log_str = f'Task: {cfg.name}, Iter:{iter_num}/{max_iter}; Accuracy = {acc_s_te:.2f}%; Total Loss={total_loss.item():.4f}' + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], eval_model, False)
                log_str = f'Task: {cfg.name}, Iter:{iter_num}/{max_iter}; Accuracy = {acc_s_te:.2f}%; Total Loss={total_loss.item():.4f}'
            
            logger.info(log_str)
            
            netF.train()
            netB.train()
            clip_model_prompt.train()
            for name, param in clip_model_prompt.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)

    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))
        torch.save(clip_model_prompt.prompt_learner.state_dict(), osp.join(cfg.output_dir, "prompt_learner_" + cfg.savename + ".pt"))
        
    final_model = nn.Sequential(netF, netB, netC)
    return final_model

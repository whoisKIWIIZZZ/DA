import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.utils import loss,prompt_tuning,IID_losses
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import  ImageList_idx, ImageList_idx_aug_fix
from sklearn.metrics import confusion_matrix
from clip.custom_clip import get_coop
import clip
from src.utils.utils import *
from copy import deepcopy
from PIL import Image
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from data.cls_to_names import *
from data.domain_datasets import domain_datasets
from data.fewshot_datasets import fewshot_datasets
from data.imagnet_prompts import imagenet_classes
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
DOWNLOAD_ROOT='/data/wangyanbo/shouzhuoyi/sfda-my/clip-module'
from clip import load,tokenize
import wandb

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
    dsets["target"] = ImageList_idx_aug_fix(txt_tar, transform=image_train())
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

def cal_acc(loader, netF, netB, netC, cfg, flag=False):
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

    num_known_classes = cfg.class_num
    all_output_known = all_output[:, :num_known_classes]
    _, predict = torch.max(all_output_known, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output_known))).cpu().data.item()

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
        s += "{}:{}".format(arg, content)
    return s

def evaluesoftmax_goodpredictiom(softmaxscore):
    acc, predict = torch.topk(softmaxscore, k=2, dim=1)
    distance = acc[:, 0] - acc[:, 1]
    return torch.mean(distance), torch.var(distance)


def evaluesoftmax_badpredictiom(softmaxscore, labelgroundtruth):
    size = labelgroundtruth.shape[0]
    tmp = torch.zeros(size, dtype=torch.float32, device='cpu')
    for i in range(size):
        tmp[i] = softmaxscore[i, labelgroundtruth[i].int()]
    distance = torch.max(softmaxscore, dim=1)[0].cpu() - tmp
    return torch.mean(distance), torch.var(distance)

def cal_satistical_info(loader, netF, netB, netC, clip_softmax_score,cfg, flag=False):
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
        wandb.log({"base_corr_clip_wrong": base_corr_clip_wrong_num / data_size,
                   "base_wrong_clip_corr": base_wrong_clip_corr_num / data_size})
        log_str = 'The base_corr_clip_wrong= \'{:.2f}%\';The base_wrong_clip_corr={:.2f}%'.format(
            100 * base_corr_clip_wrong_num / data_size, 100 * base_wrong_clip_corr_num / data_size)
        logger.info(log_str)

        # base正确但clip错误的base和clip的softmax的平均值和第二类的差距
        B_True_C_False_base_soft = base_softmax_score[base_corr_clip_wrong]
        B_True_C_False_clip_soft = clip_softmax_score[base_corr_clip_wrong]
        groud_truth_index_BTCF = all_label[base_corr_clip_wrong]
        CFBT_base_mean, CFBT_base_var = evaluesoftmax_goodpredictiom(B_True_C_False_base_soft)
        CFBT_clip_mean, CFBT_clip_var = evaluesoftmax_badpredictiom(B_True_C_False_clip_soft, groud_truth_index_BTCF)
        wandb.log({" CFBT_base_mean": CFBT_base_mean, "CFBT_base_var": CFBT_base_var, "CFBT_clip_mean": CFBT_clip_mean,
                   "CFBT_clip_var": CFBT_clip_var})
        log_str = 'The CFBT_base_mean= {}; The CFBT_base_var={};The CFBT_clip_mean={};The CFBT_clip_var={}'.format(
            CFBT_base_mean, CFBT_base_var, CFBT_clip_mean, CFBT_clip_var)
        logger.info(log_str)

        # clip正确base错误softmax的平均值和第二类的差距
        B_False_C_True_base_soft = base_softmax_score[base_wrong_clip_corr]
        B_False_C_True_clip_soft = clip_softmax_score[base_wrong_clip_corr]
        groud_truth_index_BFCT = all_label[base_wrong_clip_corr]
        BFCT_base_mean, BFCT_base_var = evaluesoftmax_goodpredictiom(B_False_C_True_clip_soft)
        BFCT_clip_mean, BFCT_clip_var = evaluesoftmax_badpredictiom(B_False_C_True_base_soft, groud_truth_index_BFCT)
        wandb.log({" BFCT_base_mean": BFCT_base_mean, "BFCT_base_var": BFCT_base_var, "BFCT_clip_mean": BFCT_clip_mean,
                   "BFCT_clip_var": BFCT_clip_var})
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
        wandb.log({" BTCT_base_mean": BTCT_base_mean, "BTCT_base_var": BTCT_base_var, "BTCT_clip_mean": BTCT_clip_mean,
                   "BTCT_clip_var": BTCT_clip_var})
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
        wandb.log({" BFCF_base_mean": BFCF_base_mean, "BFCF_base_var": BFCF_base_var, "BFCF_clip_mean": BFCF_clip_mean,
                   "BFCF_clip_var": BFCF_clip_var})
        log_str = 'The BFCF_base_mean= {}; The BFCF_base_var={};The BFCF_clip_mean={};The BFCF_clip_var={}'.format(
            BFCF_base_mean, BFCF_base_var, BFCF_clip_mean, BFCF_clip_var)
        logger.info(log_str)

def clip_model_eval(loader, myprompt, clip_model, cfg):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            
            inputs_clip = inputs.cuda()
            labels = labels.cuda()

            logit_clip_score, _ = myprompt(inputs_clip)
            
            if start_test:
                all_clip_score = logit_clip_score.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_clip_score = torch.cat((all_clip_score, logit_clip_score.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    clip_softmax_score = nn.Softmax(dim=1)(all_clip_score)
    
    num_known_classes = cfg.class_num
    clip_softmax_known = clip_softmax_score[:, :num_known_classes]
    _, predict = torch.max(clip_softmax_known, 1)
    
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label.cpu()).item() / float(all_label.size()[0])
    
    log_str = 'CLIP Accuracy = \'{:.2f}%\''.format(accuracy * 100)
    logger.info(log_str)
    wandb.log({"clip_accuracy": accuracy})

    return clip_softmax_score

def train_target(cfg):
    text_inputs = clip_pre_text(cfg)
    dset_loaders = data_load(cfg)

    # Initialize CLIP model and our prompt learner
    clip_model, _, _ = clip.load(cfg.ProDe.ARCH, device=int(cfg.GPU_ID))
    clip_model.float()
    myprompt = PromptTuningWithNegativeClasses(cfg, clip_model)
    
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    
    # Load source models
    modelpath = cfg.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))

    # Prepare classifier with hidden classes
    tmp_netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
    modelpath = cfg.output_dir_src + '/source_C.pt'
    tmp_netC.load_state_dict(torch.load(modelpath))

    # Assuming cfg.ProDe.class_num_hidden is available
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num + cfg.ProDe.class_num_hidden, bottleneck_dim=cfg.bottleneck).cuda()

    with torch.no_grad():
        netC.fc.weight_g.data[:cfg.class_num].copy_(tmp_netC.fc.weight_g.data)
        netC.fc.weight_v.data[:cfg.class_num].copy_(tmp_netC.fc.weight_v.data)
        netC.fc.bias.data[:cfg.class_num].copy_(tmp_netC.fc.bias.data)

    if cfg.ProDe.class_num_hidden > 0:
        print("yes,we use hidden classes.")
        netC.fc.weight_g.data[cfg.class_num:].fill_(0)
        netC.fc.bias.data[cfg.class_num:].fill_(0)
    netC.eval()

    param_group = []
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
    for k, v in netC.named_parameters():
        v.requires_grad = False
    
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    # The prompt is now updated manually, so optimizer_prompt is removed.

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    
    # --- New logic from diff_new_opt_new.py ---
    aux_begin = 0.2
    aux_end=0.05
    # --- End new logic ---

    while iter_num < max_iter:
        try:
            (inputs_test, inputs_test_augs), _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, inputs_test_augs), _, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        
        # --- New logic from diff_new_opt_new.py ---
        if cfg.ProDe.ARCH == 'RN50':
            inputs_test_clip = inputs_test_augs[0].cuda()
        else: 
            inputs_test_clip = inputs_test

        logit_clip_score, _ = myprompt(inputs_test_clip)
        clip_all_output = nn.Softmax(dim=1)(logit_clip_score)

        current_aux = aux_begin + (aux_end - aux_begin) * (iter_num / max_iter)
        
        # Assuming IID_losses.IID_loss can handle these arguments
        iic_loss = IID_losses.IID_loss(
            cfg.class_num,
            softmax_out,
            clip_all_output,
            1-current_aux,
            current_aux,
            lamb= 1.0
        )
        classifier_loss = cfg.ProDe.IIC_PAR * iic_loss
        # --- End new logic ---

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
        
        # Pseudo-labeling part from ProDe
        with torch.no_grad():
            _, pred = torch.max(clip_all_output, 1)
        entropy_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        classifier_loss =  cfg.ProDe.GENT_PAR*entropy_loss + classifier_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        # Manual prompt update
        with torch.no_grad():
            if myprompt.prompt_learner.ctx.grad is not None:
                promt_learning_rate = 0.002 if iter_num < (max_iter / 2) else 0.0001
                myprompt.prompt_learner.ctx -= myprompt.prompt_learner.ctx.grad * promt_learning_rate
                myprompt.prompt_learner.ctx.grad.zero_()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            myprompt.eval()

            clip_softmax_score = clip_model_eval(dset_loaders['test'], myprompt, clip_model, cfg)
            cal_satistical_info(dset_loaders['test'], netF, netB, netC, clip_softmax_score, cfg, flag=False)

            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, cfg, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list
                wandb.log({"accuracy": acc_s_te, "loss ": classifier_loss})
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, cfg, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss)
                wandb.log({"accuracy": acc_s_te, "loss ": classifier_loss})

            logger.info(log_str)
            netF.train()
            netB.train()
            netC.train()
            myprompt.train()

    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))
        
    return netF, netB, netC

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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg,classnames,clip_model,  batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        
        ctx_init = cfg.ProDe.CTX_INIT
        n_ctx = cfg.ProDe.N_CTX

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype)
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]
            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames
        self.clip_model = clip_model

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip_model, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat([prefix, ctx, cls, suffix], dim=-2)
            else:
                prompts = torch.cat([prefix, ctx, suffix], dim=-2)
        # ... other positions
        return prompts

class PromptTuningWithNegativeClasses(torch.nn.Module):
    def __init__(self, cfg, clip_model):
        super(PromptTuningWithNegativeClasses, self).__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.class_num_hidden = cfg.ProDe.class_num_hidden
        self.device = clip_model.visual.conv1.weight.device

        classnames = cfg.classname
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(self.clip_model)

        if self.class_num_hidden > 0:
            print(f"Initializing {self.class_num_hidden} learnable hidden prompt classes.")
            negative_prompt_base_prefix = "a photo of " 
            negative_prompts_list = []
            for cls_name in classnames:
                negative_prompt_string = f"{negative_prompt_base_prefix} not {cls_name}"
                negative_prompts_list.append(negative_prompt_string)
                
            hidden_features_tokens = torch.cat([clip.tokenize(p) for p in negative_prompts_list]).to(self.device)

            with torch.no_grad():
                embedding = self.clip_model.token_embedding(hidden_features_tokens).type(self.clip_model.dtype)
                hidden_text_features = self.text_encoder(embedding, hidden_features_tokens)
                hidden_text_features = hidden_text_features / hidden_text_features.norm(dim=-1, keepdim=True)

            self.hidden_features = hidden_text_features[:self.class_num_hidden, :]
        else:
            self.hidden_features = None

    def forward(self, img_inputs):
        image_features = self.clip_model.encode_image(img_inputs.type(self.clip_model.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        base_text_features = self.text_encoder(prompts, self.tokenized_prompts)

        if self.class_num_hidden > 0 and self.hidden_features is not None:
            all_text_features = torch.cat([base_text_features, self.hidden_features], dim=0)
        else:
            all_text_features = base_text_features
        
        all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ all_text_features.t()
        
        return logits, all_text_features

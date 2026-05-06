import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
# from src.utils import loss,prompt_tuning,IID_losses
from src.utils import loss,IID_losses

from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList_idx
from sklearn.metrics import confusion_matrix
import clip
from src.utils.utils import *
from src.utils.prompt_tuning_diff_new_optim import *

from .utils import *
logger = logging.getLogger(__name__)

import wandb
import copy
#初始化
def copy_netC(netC,netC_aux,num_known_classes):
    state_dict_C = netC.state_dict()
    state_dict_C_aux = netC_aux.state_dict()

    for key in state_dict_C.keys():
        if key in state_dict_C_aux:
            #假设参数形状是 [num_classes, ...]，我们只复制前 num_known_classes 行/元素
            #检查是否是权重或偏置相关的键，并且其第一个维度对应类别数
            if state_dict_C[key].shape[0] == num_known_classes and state_dict_C_aux[key].shape[0] > num_known_classes:
                #复制前 num_known_classes 行/元素
                state_dict_C_aux[key][:num_known_classes] = state_dict_C[key].clone() # 使用 clone() 是好习惯
            elif state_dict_C[key].shape[0] == num_known_classes and state_dict_C_aux[key].shape[0] == num_known_classes:
                #如果形状完全一样，直接复制整个张量 (例如，如果 netC_aux 只有 known classes)
                state_dict_C_aux[key] = state_dict_C[key].clone()
    
    netC_aux.load_state_dict(state_dict_C_aux, strict=False)



def print_grad_hook(grad):
    if grad.norm().item() > 100:
        print('Hook: Grad recieved with norm {}'.format(grad.norm().item()))

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def image_test_50(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=[0.26862954, 0.26130258, 0.27577711])

    return transforms.Compose([
        transforms.Resize(crop_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def test_time_tuning(model, inputs, optimizer, cfg, target_output,current_aux):
    target_output = target_output.cuda()
    target_output =  nn.Softmax(dim=1)(target_output)
    for j in range(cfg.ProDe.TTA_STEPS):

        output_logits,_ = model(inputs)
        output = nn.Softmax(dim=1)(output_logits)
        iic_loss = IID_losses.IID_loss(cfg.class_num,target_output,output,1-current_aux, current_aux , lamb=1.0)
        '''
          total_aux_loss = IID_losses.IID_loss(
                    cfg.class_num,
                    softmax_out,
                    clip_all_output,
                    1-current_aux,#base
                    current_aux,#hidden
                    lamb= 1.0)
        '''
        optimizer.zero_grad()
        iic_loss.backward()
        optimizer.step()
    return output

def test_time_adapt_eval(input, model, optimizer, optim_state, cfg, target_output,current_aux):
    optimizer.load_state_dict(optim_state)
    output = test_time_tuning(model, input, optimizer, cfg, target_output,current_aux=current_aux)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            model.eval()
            output,_ = model(input)
    return output

def data_load(cfg,preprocess):
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
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["clip_test"] = ImageList_idx(txt_test, transform=preprocess)
    dset_loaders["clip_test"] = DataLoader(dsets["clip_test"], batch_size=train_bs * 3, shuffle=False,num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    # else:
    # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    # else:
    # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)  # 0<decay<1，且越来越小
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        # param_group['momentum'] = 0.9
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def lr_scheduler2(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        # param_group['momentum'] = 0.9
        param_group['momentum'] = 0.5
        param_group['nesterov'] = True
    return optimizer

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\\n".format(arg, content)
    return s


def train_target(cfg):
    print("using the new opt file")
    is_merge_stratg_use = True  # 加入衰减策略，保持高精度模型对低精度模型的指导(优先选False,还可true)!!!!!!!!!!!!!!!!!!!!
    abulations = 0  # 衰减系数

    beta_pre_1 = 1# 原是1
    beta_pre_2 = 0.1# 原是0.2

    aux_begin = 0.5
    aux_end=0.1

    print("aux_begin:", aux_begin, "aux_end:", aux_end)
    print("is_merge_stratg_use",
          is_merge_stratg_use, "beta_pre_1:", beta_pre_1, "beta_pre_2:", beta_pre_2)
    # clip模型参数初始化
    clip_model, preprocess, _ = clip.load(cfg.DIFF_ALIGN.ARCH)  # vit-B32
    clip_model.float()
    # 引入clip后，自构造text的输入
    text_inputs = clip_pre_text(cfg)  # 65*77
    # 训练测试数据导入
    dset_loaders = data_load(cfg,preprocess)

    print("using the arch:", cfg.MODEL.ARCH, "+++++++++++++++++++++++++++++")
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
    netC_neg = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
    netC_aux = network.feat_classifier(type='wn', class_num=cfg.class_num + cfg.class_num_hidden, bottleneck_dim=cfg.bottleneck).cuda()
      # 提取用source数据训练出的模型
    modelpath = cfg.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath),strict=True)
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath),strict=True)
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    modelpath = cfg.output_dir_src + '/source_C_aux_neg_head.pt'
    netC_neg.load_state_dict(torch.load(modelpath))
    
    old_fc = netC.fc
    old_fc_neg = netC_neg.fc
    new_fc = torch.nn.utils.weight_norm(nn.Linear(old_fc.in_features, cfg.class_num + cfg.class_num_hidden).cuda())
    with torch.no_grad():
        new_fc.weight_g.data[:cfg.class_num].copy_(old_fc.weight_g.data)
        new_fc.weight_v.data[:cfg.class_num].copy_(old_fc.weight_v.data)
        new_fc.bias.data[:cfg.class_num].copy_(old_fc.bias.data)

        if cfg.class_num_hidden > 0:
            print("Yes,we use hidden classes.")
            new_fc.weight_g.data[cfg.class_num:].copy_(old_fc_neg.weight_g.data)
            new_fc.weight_v.data[cfg.class_num:].copy_(old_fc_neg.weight_v.data)
            new_fc.bias.data[cfg.class_num:].copy_(old_fc_neg.bias.data)
    netC_aux.fc = new_fc
    netC.eval()


    param_group = []
    param_group_C = []
    param_group_aux = []

    if cfg.class_num_hidden > 0 :
        for k,v in netC_aux.named_parameters():
            v.requires_grad = False
            if 'weight' in k:
                v[cfg.class_num:,:].requires_grad = True
            elif k == 'bias':
                v[cfg.class_num:].requires_grad = True
            param_group_aux +=  [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]

    for k,v in netC.named_parameters():
        v.requires_grad = True
        param_group_C +=  [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1*0.3}]

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

    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    # score_bank包含所有样本
    score_bank = torch.randn(num_sample, cfg.class_num).cuda()  # 4365目标样本,65类
    netF.eval()
    netB.eval()
    netC.eval()
    netC_aux.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            indx = data[-1]  # 把对应的score_bank里面的内容更新
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            outputs = netC(output)
            outputs = nn.Softmax(dim=1)(outputs)
            # Pad the outputs to match the score_bank dimension including hidden classes
            score_bank[indx] = outputs.detach().clone()

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"]) * 2  # 循环次数*每个数据集包含的batch个数
    interval_iter = max_iter  // cfg.TEST.INTERVAL  # for comment
    # interval_iter = max_iter//3 // cfg.TEST.INTERVAL #for vista
    iter_num = 0
    text_features = None
    difficult_data_indices = None
    difficult_data_initial_quadrant_colors = None
    slightly_wrong_indices = None


    mypromt = promote_learning_diff_promote_learning(cfg, clip_model).to(clip_model.visual.conv1.weight.device)
    init_negative_with_mean(mypromt)
    state_neg = torch.load(
        #osp.join(cfg.output_dir_src, "source_prompt_learner_dual_kl_neg.pt"),
        osp.join("/data/home/scv8780/run/sfda-my/Models/source/uda/office-home/P", "source_prompt_learner_dual_kl_neg.pt"),
        map_location="cuda"
    )

    mypromt.negative_prompt_learner.load_state_dict(state_neg)
    mypromt.positive_prompt_learner.ctx.register_hook(print_grad_hook)
    if iter_num % (interval_iter //9) ==0:
        copy_netC(netC,netC_aux,cfg.class_num)
    while iter_num < max_iter:

        current_aux = aux_begin + (aux_end - aux_begin) * (iter_num / max_iter)
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        inputs_test = inputs_test.cuda()
        
        if inputs_test.size(0) == 1:
            continue

        #copy_netC(netC,netC_aux,cfg.class_num)
        # 就是这一段，在把vil模型固定的同时，使得vil的输入特征变成可以学习的
        if iter_num % interval_iter == 0 and cfg.DIFF_ALIGN.CLS_PAR > 0 and iter_num !=0:
            #copy_netC(netC,netC_aux,cfg.class_num)
            netF.eval()
            netB.eval()
            netC.eval()
            netC_aux.eval()
            # confi-img是图像位置，confi_dis是得到的两模型sofamax后相加的结果，这里是想得到clip+resnet50的总的推断结果
            # 注意到，这里是基于测试数据的。同时也是用于测试clip精度的
            test_clip_all_output,_= CLIP_model_eval(dset_loaders['test'], text_inputs, text_features, clip_model)
            # 分析模型重叠
            difficult_data_indices, difficult_data_initial_quadrant_colors, slightly_wrong_indices = cal_satistical_info(dset_loaders['test'], netF, netB, netC, test_clip_all_output,cfg, iter_num, difficult_data_indices, difficult_data_initial_quadrant_colors, slightly_wrong_indices, flag=False)
            netF.train()
            netB.train()
            netC.train()
            netC_aux.train()

        if 'RN' in cfg.DIFF_ALIGN.ARCH:
            prompt_lr = cfg.OPTIM.LR * 0.1
            data_transform = image_test_50()
        else:  # 走这一边
            data_transform = image_test()
            prompt_lr = cfg.OPTIM.LR

        # if mypromt.positive_prompt_learner.ctx.grad is not None:
        #     mypromt.positive_prompt_learner.ctx.grad.data.zero_()

        iter_num += 1
        if iter_num < 2000:  # 600,2000最好
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        else:
            lr_scheduler2(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_shared = netF(inputs_test)
        features_classify = netB(features_shared)
        outputs_test = netC_aux(features_classify)
        base_outputs_test = netC(features_classify)

        outputs_test_65 = outputs_test[:,:cfg.class_num]
        base_softmax_out = nn.Softmax(dim=1)(base_outputs_test)
        softmax_out_65 = nn.Softmax(dim=1)(outputs_test_65)
        softmax_out = torch.cat(
            (softmax_out_65,
            nn.Softmax(dim=1)(outputs_test[:,cfg.class_num:])),dim=1
        )
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        ln_sam = softmax_out.shape[0]
        # data = basedata[0:ln_sam]


        if is_merge_stratg_use:
            # data = np.random.exponential(scale=1*(np.exp(iter_num/max_iter)/(np.exp(iter_num/max_iter)
            #                                     +np.exp(max_iter/(iter_num+ 1e-6)))) + beta_pre_2, size=ln_sam)
            # data[data>1]=1
            abulations = beta_pre_1 * iter_num / max_iter + beta_pre_2
            data = np.random.exponential(scale=abulations, size=ln_sam)  # 0.5
            data[data > 1] = 1
        else:
            data = np.random.exponential(scale=0.1, size=ln_sam)
        
        data = np.expand_dims(data, axis=1)
        data = torch.from_numpy(data)
        data = data.numpy()

        K = base_softmax_out.size(1)
        
        logit_clip_score, text_features = mypromt(inputs_test)

        clip_all_output = nn.Softmax(dim=1)(logit_clip_score)
        base_clip_all_output = nn.Softmax(dim=1)(logit_clip_score[:,:cfg.class_num])
        _, predict = torch.max(score_bank[tar_idx], 1)  # 从bank来的
        _, clip_predict = torch.max(base_clip_all_output, 1)  # 从test clip来
        
        predict_one = np.eye(K)[predict.cpu()]       
        clip_one = np.eye(K)[clip_predict.cpu()]

        predict_mix = data * predict_one + (1 - data) * clip_one
        predict_mix = torch.from_numpy(predict_mix).cuda()


        if cfg.DIFF_ALIGN.CLS_PAR > 0:
            targets = predict_mix
            outputs_log_softmax = base_outputs_test
            pseudo_label_loss = (- targets * outputs_log_softmax).sum(dim=1).mean()
            classifier_loss = pseudo_label_loss * cfg.DIFF_ALIGN.CLS_PAR
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        neg = logit_clip_score[:,cfg.class_num:].detach() 
        pos = logit_clip_score[:,:cfg.class_num].detach()
        p_max = pos.max(dim=1, keepdim=True).values 
        p_min = pos.min(dim=1,keepdim = True).values
        pos_norm = (pos - p_min) / (p_max - p_min + 1e-8)  # [0,1]
        pos = 1 - pos_norm 


        clip_all_output_for_loss_aux = nn.Softmax(dim=1)(torch.cat([
                logit_clip_score[:,:cfg.class_num], 
                neg     # negative 不参与 loss_aux 的反传
            ], dim=1))
        bb = 1- current_aux
        nnn = current_aux 
        bn = -1- current_aux
        loss_aux = IID_losses.IID_loss(
            cfg.class_num,
            softmax_out,
            clip_all_output_for_loss_aux,
            bb,
            nnn,
            bn,#[CLS] - not a [CLS]
            lamb = 1.0
        )
        loss_aux = loss_aux -((nn.functional.log_softmax(base_outputs_test, dim=1)* nn.Softmax(dim=1)(neg)).sum(dim=1).mean()*0.3*(1-iter_num/max_iter))#试一下衰减策略
        pos = nn.Softmax(dim=1)(pos)
        neg = nn.Softmax(dim=1)(logit_clip_score[:,cfg.class_num:])
        #loss_aux = loss_aux - (nn.functional.kl_div(pos,neg,reduction="batchmean"))*0.5

        m_base_sft = base_softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-m_base_sft * torch.log(m_base_sft + cfg.LCFD.EPSILON))
      
        with torch.no_grad():
            score_bank[tar_idx] = base_softmax_out.detach().clone()

        optimizer.zero_grad()
        opt_C.zero_grad()
        opt_aux.zero_grad()
        total_loss = classifier_loss + loss_aux   - cfg.DIFF_ALIGN.GENT_PAR * gentropy_loss 
        total_loss.backward()

        optimizer.step()
        opt_aux.step()
        opt_C.step()

        if iter_num <  (max_iter/2):
            with torch.no_grad():
                #opt_clip.step()
                promt_learning_rate = 0.002
                mypromt.positive_prompt_learner.ctx  -=  mypromt.positive_prompt_learner.ctx.grad * promt_learning_rate
                mypromt.negative_prompt_learner.ctx  -=  mypromt.negative_prompt_learner.ctx.grad * promt_learning_rate
        else:
            with torch.no_grad():
                #opt_clip.step()
                promt_learning_rate = 0.0001
                mypromt.positive_prompt_learner.ctx  -=  mypromt.positive_prompt_learner.ctx.grad * promt_learning_rate
                mypromt.negative_prompt_learner.ctx  -=  mypromt.negative_prompt_learner.ctx.grad * promt_learning_rate
        


        if iter_num % interval_iter == 0 or iter_num == max_iter or iter_num == 1:
            netF.eval()
            netB.eval()
            netC.eval()
            netC_aux.eval()
            if cfg.SETTING.DATASET == 'VISDA-C':
                acc_s_te, neg_acc,acc_list = cal_acc(dset_loaders['test'], netF, netB, netC,cfg, True)
                acc_s_te_aux, _ = cal_acc(dset_loaders['test'], netF, netB, netC_aux,cfg, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%,{:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter,
                                                                                     acc_s_te,acc_s_te_aux,
                                                                                     total_loss) + '\n' + acc_list

               
               # wandb.log({"accuracy": acc_s_te, "loss ": total_loss})
            else:
                acc_s_te, _,_ = cal_acc(dset_loaders['test'], netF, netB, netC,cfg, False)
                acc_s_te_aux,acc_neg, _ = cal_acc(dset_loaders['test'], netF, netB, netC_aux,cfg, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = NETC:{:.2f}%,AUX_POS:{:.2f}%,AUX_NEG:{:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter,
                                                                                     acc_s_te,acc_s_te_aux,acc_neg,
                                                                                     total_loss)

             #   wandb.log({"accuracy": acc_s_te, "loss ": total_loss})
            logger.info(log_str)
            netF.train()
            netB.train()
            netC.train()
            netC_aux.train()
            

    if cfg.ISSAVE:
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_classify_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C__classify" + cfg.savename + ".pt"))

    return netF, netB, netC


def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\\n".format(arg, content)
    return s

def clip_pre_text(cfg):
    #M = cfg.class_num_hidden
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.DIFF_ALIGN.CTX_INIT.replace("_", " ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # 把text变为数字信息

    return tokenized_prompts


def clip_text(model, text_features, inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits

@torch.no_grad()
def init_negative_with_mean(mypromt):
    """
    Initialize negative prompts as mean of all other positive prompts.
    This creates per-class ctx of shape (n_cls, n_ctx, ctx_dim) and replaces
    mypromt.negative_prompt_learner.ctx (and ctx_init_state) accordingly.
    """

    # device / dtype
    device = mypromt.clip_model.visual.conv1.weight.device
    dtype = mypromt.positive_prompt_learner.dtype

    # 1) 得到 positive text embeddings: shape (C, D)
    pos_prompts = mypromt.positive_prompt_learner()               # (C, L, D)
    pos_tok = mypromt.positive_prompt_learner.tokenized_prompts
    pos_emb = mypromt.text_encoder(pos_prompts, pos_tok)          # (C, D)
    pos_emb = pos_emb.to(device)                                  # ensure device
    C, D = pos_emb.shape
    n_ctx = mypromt.negative_prompt_learner.n_ctx  # number of ctx tokens (int)

    # 2) 构造每类的 negative prototype：mean of others -> (C, D)
    neg_protos = []
    for i in range(C):
        if C == 1:
            # corner case: only one class -> use negative of itself (rare)
            proto = -pos_emb[i:i+1]
        else:
            idx = [j for j in range(C) if j != i]
            proto = pos_emb[idx].mean(dim=0, keepdim=True)  # (1, D)
        neg_protos.append(proto)
    neg_protos = torch.cat(neg_protos, dim=0)               # (C, D)
    neg_protos = neg_protos / (neg_protos.norm(dim=-1, keepdim=True) + 1e-12)

    # 3) 构造 new_ctx：shape (n_cls, n_ctx, D)
    # 我们在每个 token 上复制 proto 向量并加一点小噪声以便可训练
    new_ctx = torch.zeros((C, n_ctx, D), device=device, dtype=dtype)
    for i in range(C):
        for t in range(n_ctx):
            # small scale to avoid覆盖原语义并留有可训练空间
            new_ctx[i, t] = neg_protos[i] * 0.01 + torch.randn(D, device=device, dtype=dtype) * 0.001

    # 4) set parameter and init_state consistently
    # PromptLearner expects ctx to be either (n_ctx, D) or (n_cls, n_ctx, D).
    # We will set it to (n_cls, n_ctx, D) so forward uses it directly.
    # However original PromptLearner used ctx shape (n_ctx, D) — replacing with (n_cls, n_ctx, D) is fine.
    neg_pl = mypromt.negative_prompt_learner

    # If existing ctx is a Parameter, delete/replace it safely:
    try:
        # assign new nn.Parameter
        neg_pl.ctx = nn.Parameter(new_ctx)
    except Exception as e:
        # fallback: set attribute then wrap as Parameter
        neg_pl.__dict__['ctx'] = nn.Parameter(new_ctx)

    # update ctx_init_state (used by reset)
    neg_pl.ctx_init_state = new_ctx.detach().clone()

    # Ensure dtype/device consistent
    neg_pl.ctx = neg_pl.ctx.to(device=device, dtype=dtype)

    print(f"[NEG INIT] negative ctx initialized to per-class means: new ctx shape = {neg_pl.ctx.shape}")
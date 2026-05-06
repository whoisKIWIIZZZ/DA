import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
#from src.utils import loss,prompt_tuning,IID_losses
from src.utils import loss,IID_losses

from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import  ImageList_idx
from sklearn.metrics import confusion_matrix
import clip
from src.utils.utils import *
from src.utils.prompt_tuning_diff_new_optim_old import *
logger = logging.getLogger(__name__)

import wandb
import copy

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

def image_test_50(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=[0.26862954, 0.26130258, 0.27577711])
                            
    return transforms.Compose([
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])

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
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
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
    decay = (1 + gamma * iter_num / max_iter) ** (-power)  # 0<decay<1，且越来越小
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        #param_group['momentum'] = 0.9
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def lr_scheduler2(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        #param_group['momentum'] = 0.9
        param_group['momentum'] = 0.5
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, netF, netB, netC, flag=False):
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

def cal_satistical_info(loader, netF, netB, netC, clip_softmax_score, flag=False):
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
    
        base_softmax_score = nn.Softmax(dim=1)(all_output)
        _, base_predict = torch.max(base_softmax_score, 1)
        _, clip_predict = torch.max(clip_softmax_score.cpu(), 1)
        base_sign = (torch.squeeze(base_predict).float() == all_label)
        clip_sign = (torch.squeeze(clip_predict).float() == all_label)
        #base是对的，但clip是错的
        base_corr_clip_wrong = (base_sign & (~clip_sign))
        #base是错的，但clip是对的
        base_wrong_clip_corr = ((~base_sign) & clip_sign)
        data_size = all_label.size()[0]
        
        base_corr_clip_wrong_num = torch.sum(base_corr_clip_wrong).item()
        base_wrong_clip_corr_num = torch.sum(base_wrong_clip_corr).item()
        #wandb.log({"base_corr_clip_wrong":base_corr_clip_wrong_num/data_size,"base_wrong_clip_corr":base_wrong_clip_corr_num/data_size})
        log_str = 'The base_corr_clip_wrong= {:.2f}%;The base_wrong_clip_corr={:.2f}%'.format(100*base_corr_clip_wrong_num/data_size,100*base_wrong_clip_corr_num/data_size)
        logger.info(log_str)

        #base正确但clip错误的base和clip的softmax的平均值和第二类的差距
        B_True_C_False_base_soft = base_softmax_score[base_corr_clip_wrong]
        B_True_C_False_clip_soft = clip_softmax_score[base_corr_clip_wrong]
        groud_truth_index_BTCF = all_label[base_corr_clip_wrong]
        CFBT_base_mean, CFBT_base_var = evaluesoftmax_goodpredictiom(B_True_C_False_base_soft)
        CFBT_clip_mean, CFBT_clip_var = evaluesoftmax_badpredictiom(B_True_C_False_clip_soft,groud_truth_index_BTCF)
        #wandb.log({" CFBT_base_mean":CFBT_base_mean,"CFBT_base_var":CFBT_base_var,"CFBT_clip_mean": CFBT_clip_mean,"CFBT_clip_var":CFBT_clip_var})
        log_str = 'The CFBT_base_mean= {}; The CFBT_base_var={};The CFBT_clip_mean={};The CFBT_clip_var={}'.format(CFBT_base_mean, CFBT_base_var, CFBT_clip_mean, CFBT_clip_var)
        logger.info(log_str)

        #clip正确base错误softmax的平均值和第二类的差距
        B_False_C_True_base_soft = base_softmax_score[base_wrong_clip_corr]
        B_False_C_True_clip_soft = clip_softmax_score[base_wrong_clip_corr]
        groud_truth_index_BFCT = all_label[base_wrong_clip_corr]
        BFCT_base_mean, BFCT_base_var = evaluesoftmax_goodpredictiom(B_False_C_True_clip_soft)
        BFCT_clip_mean, BFCT_clip_var = evaluesoftmax_badpredictiom(B_False_C_True_base_soft, groud_truth_index_BFCT)
        #wandb.log({" BFCT_base_mean":BFCT_base_mean,"BFCT_base_var":BFCT_base_var,"BFCT_clip_mean": BFCT_clip_mean,"BFCT_clip_var":BFCT_clip_var})
        log_str = 'The BFCT_base_mean= {}; The BFCT_base_var={};The BFCT_clip_mean={};The BFCT_clip_var={}'.format(BFCT_base_mean, BFCT_base_var, BFCT_clip_mean, BFCT_clip_var)
        logger.info(log_str)

        #base正确clip正确，与第二类的差值
        base_True_clip_True = (base_sign & clip_sign)
        B_True_C_True_base_soft = base_softmax_score[base_True_clip_True]
        B_True_C_True_clip_soft = clip_softmax_score[base_True_clip_True]
        groud_truth_index_BTCT = all_label[base_True_clip_True]
        BTCT_base_mean, BTCT_base_var = evaluesoftmax_goodpredictiom(B_True_C_True_base_soft)
        BTCT_clip_mean, BTCT_clip_var = evaluesoftmax_goodpredictiom(B_True_C_True_clip_soft)
        #wandb.log({" BTCT_base_mean":BTCT_base_mean,"BTCT_base_var":BTCT_base_var,"BTCT_clip_mean": BTCT_clip_mean,"BTCT_clip_var":BTCT_clip_var})
        log_str = 'The BTCT_base_mean= {}; The BTCT_base_var={};The BTCT_clip_mean={};The BTCT_clip_var={}'.format(BTCT_base_mean, BTCT_base_var, BTCT_clip_mean, BTCT_clip_var)
        logger.info(log_str)

        #base错误clip错误，分别与groundtruth的softmax上差异
        base_False_clip_False = (~base_sign & ~clip_sign)
        B_False_C_False_base_soft = base_softmax_score[base_False_clip_False]
        B_False_C_False_clip_soft = clip_softmax_score[base_False_clip_False]
        ground_truth_index_BFCF = all_label[base_False_clip_False]
        BFCF_base_mean, BFCF_base_var = evaluesoftmax_badpredictiom(B_False_C_False_base_soft, ground_truth_index_BFCF)
        BFCF_clip_mean, BFCF_clip_var = evaluesoftmax_badpredictiom(B_False_C_False_clip_soft, ground_truth_index_BFCF)
        #wandb.log({" BFCF_base_mean":BFCF_base_mean,"BFCF_base_var":BFCF_base_var,"BFCF_clip_mean": BFCF_clip_mean,"BFCF_clip_var":BFCF_clip_var})
        log_str = 'The BFCF_base_mean= {}; The BFCF_base_var={};The BFCF_clip_mean={};The BFCF_clip_var={}'.format(BFCF_base_mean, BFCF_base_var, BFCF_clip_mean, BFCF_clip_var)
        logger.info(log_str)
        


#输出变量
def evaluesoftmax_goodpredictiom(softmaxscore):
    acc , predict=torch.topk(softmaxscore, k = 2, dim = 1)
    distance = acc[:,0]-acc[:,1]
    return torch.mean(distance), torch.var(distance)

def evaluesoftmax_badpredictiom(softmaxscore,labelgroundtruth): #411*65
    size = labelgroundtruth.shape[0]
    tmp = torch.zeros(size,dtype=torch.float32,device='cpu')
    for i in range(size):
        tmp[i] =  softmaxscore[i,labelgroundtruth[i].int()]

    #tmp_index = torch.cat(torch.arange(size),labelgroundtruth,dim=1)
    #tmp_index = torch.cat(torch.arange(size),labelgroundtruth,dim=1)
    distance = torch.max(softmaxscore, dim=1)[0].cpu() - tmp
    return torch.mean(distance), torch.var(distance)




def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(cfg):
    print("using the new opt file")
    is_clip_form_begin = False

    #is_clip_train = True #指的是clip是长训练（单独优化多个epoch，还是只优化一个epcoh）
    
    is_c_learn = False #这里指的是target最后一层是否冻结，False就是冻结(优先选False)

    is_merge_stratg_use = True #加入衰减策略，保持高精度模型对低精度模型的指导(优先选False,还可true)!!!!!!!!!!!!!!!!!!!!

    is_test_merge = False #是否把原信号衰减接入到clip信号获取中(几乎没用，不管)
    is_new_merge_str = False #是否用对抗训练
    
    abulations=0 #衰减系数

    beta_pre_1=1  #原是1
    beta_pre_2=0.1  #原是0.2

    print("is_clip_form_begin:",is_clip_form_begin,"is_c_learn",is_c_learn,"is_merge_stratg_use",is_merge_stratg_use,"beta_pre_1:",beta_pre_1,"beta_pre_2:",beta_pre_2)
    #clip模型参数初始化
    clip_model, preprocess,_ = clip.load(cfg.DIFO.ARCH) #vit-B32
    clip_model.float()
    #引入clip后，自构造text的输入
    text_inputs = clip_pre_text(cfg) #65*77
    #训练测试数据导入
    dset_loaders = data_load(cfg)
    
    ## set base network
    print("using the arch:",cfg.MODEL.ARCH,"+++++++++++++++++++++++++++++")
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features , bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    #新加，用于指定训练文件夹,无需使用了
    #cfg.output_dir_src="/data1/users/wushuang/source-free-clip/source"
    #提取用source数据训练出的模型 
    modelpath = cfg.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    
    param_group = []
    #参数存储样本和学习率
    #让netC不更新
    for k, v in netC.named_parameters():
        v.requires_grad = False 
        param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1*0.5}]

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


    #optimizer = optim.SGD(param_group)
    optimizer = optim.SGD(param_group)
    
    optimizer = op_copy(optimizer)

    loader = dset_loaders["target"]
    num_sample=len(loader.dataset)
    #score_bank包含所有样本
    score_bank = torch.randn(num_sample, cfg.class_num).cuda() #4365目标样本,65类
    netF.eval()
    netB.eval()
    netC.eval()
    #获得score_bank，记录resnet的过往值
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            indx = data[-1]  #把对应的score_bank里面的内容更新
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            outputs = netC(output)
            outputs = nn.Softmax(dim=1)(outputs)
            score_bank[indx] = outputs.detach().clone() #batch*类别,为64*65,为原网络在经历softmax后得到的值的历史记录

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"]) * 2 #循环次数*每个数据集包含的batch个数
    interval_iter = max_iter // cfg.TEST.INTERVAL * 5 #for comment
    #interval_iter = max_iter//3 // cfg.TEST.INTERVAL #for vista
    iter_num = 0
    text_features = None

    
    #构建Promt learner
    is_feature_or_prompt_learn = False #学习text_freature为true，false为promt learning
    if is_feature_or_prompt_learn:   #这个不用了，效果不好
        mypromt = promote_learning_diff_feature_learning(cfg, clip_model)
    else:
        mypromt = promote_learning_diff_promote_learning(cfg, clip_model)

    if is_new_merge_str:
        data = np.random.exponential(0.1, size=cfg.class_num)
        #data = np.random.normal( 0, 2, size=cfg.class_num)

        #data = np.random.exponential(0.2, size=cfg.class_num) #多数实验结果
        #data[data>1]=1
        data = torch.tensor(np.expand_dims(data, axis=0))
        data.requires_grad = True
        #data = data.cuda()
        #data = data.equires_grad()cuda()
        #data.requires_grad = True



    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"]) 
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue
        #就是这一段，在把vil模型固定的同时，使得vil的输入特征变成可以学习的
        if iter_num % interval_iter == 0 and cfg.DIFO.CLS_PAR > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            #confi-img是图像位置，confi_dis是得到的两模型sofamax后相加的结果，这里是想得到clip+resnet50的总的推断结果
            #注意到，这里是基于测试数据的。同时也是用于测试clip精度的
            test_clip_all_output = CLIP_model_eval(dset_loaders['test'], text_inputs, text_features, clip_model)
            cal_acc_js(dset_loaders['test'],netF,netB,netC,test_clip_all_output)
            #分析模型重叠
            cal_satistical_info(dset_loaders['test'], netF, netB, netC, test_clip_all_output, flag=False)
            netF.train()
            netB.train()


        if is_feature_or_prompt_learn:
            logit_clip_score, text_features = mypromt(inputs_test)
            clip_all_output = nn.Softmax(dim=1)(logit_clip_score) 
            #optimizerclip = optim.SGD(mypromt.parameters(), 1.0,weight_decay=5e-4,momentum=0.9,nesterov=False) #optim.SGD(mypromt.parameters(), 1e-2,weight_decay=5e-4,momentum=0.9,nesterov=False)
        else:
            """  try:
                logit_clip_score, text_features = mypromt(inputs_test, trainable_param)
            except: """

            logit_clip_score, text_features = mypromt(inputs_test)
            #trainable_param = deepcopy(mypromt.prompt_learner.ctx)

            if 'RN' in cfg.DIFO.ARCH :
                prompt_lr = cfg.OPTIM.LR*0.1
                data_transform = image_test_50()
            else : #走这一边
                data_transform = image_test()
                prompt_lr = cfg.OPTIM.LR

            try:
                opt_clip.zero_grad()
            except:
                prompt_parameters = [mypromt.prompt_learner.ctx]
                opt_clip = torch.optim.SGD(prompt_parameters, prompt_lr,weight_decay=5e-4,momentum=0.9,nesterov=False)
                opt_clip.zero_grad()
                if mypromt.prompt_learner.ctx.grad is not None:
                    mypromt.prompt_learner.ctx.grad.data.zero_() 

                 
        if is_feature_or_prompt_learn:
            mypromt.text_features.requires_grad = False
        
        clip_all_output = nn.Softmax(dim=1)(logit_clip_score) 

        iter_num += 1
        if iter_num<2000: #600,2000最好
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        else:
            lr_scheduler2(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.cuda()
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        
        
        ln_sam = softmax_out.shape[0]
        #data = basedata[0:ln_sam]
        if is_new_merge_str:
            #data = data.view(1,-1)
            if data.grad is not None:
                data.grad.data.zero_() 
        else:
            if is_merge_stratg_use:
                # data = np.random.exponential(scale=1*(np.exp(iter_num/max_iter)/(np.exp(iter_num/max_iter)
                #                                     +np.exp(max_iter/(iter_num+ 1e-6)))) + beta_pre_2, size=ln_sam)
                #data[data>1]=1
                abulations = beta_pre_1*iter_num/max_iter + beta_pre_2
                data = np.random.exponential(scale = abulations, size=ln_sam) #0.5
                data[data>1]=1
            else:
                data = np.random.exponential(scale=0.1, size=ln_sam)
            #data = np.random.exponential(scale=1*iter_num/max_iter+0.1, size=ln_sam) #0.5
        
            data = np.expand_dims(data, axis=1)
            data = torch.from_numpy(data)
            data = data.numpy()  

        K = softmax_out.size(1)
        _, predict = torch.max(score_bank[tar_idx], 1)  #从bank来的
        _, clip_predict = torch.max(clip_all_output, 1) #从test clip来
        


        predict_one = np.eye(K)[predict.cpu()]
        clip_one = np.eye(K)[clip_predict.cpu()]

        if is_new_merge_str:
            predict_one = torch.from_numpy(predict_one)
            clip_one = torch.from_numpy(clip_one)
            predict_mix = data*predict_one + (1-data)*clip_one #64*65
            predict_mix = predict_mix.cuda()
        else:
            predict_mix = data*predict_one + (1-data)*clip_one
            predict_mix = torch.from_numpy(predict_mix).cuda()

        if cfg.DIFO.CLS_PAR > 0:
            targets = predict_mix
            #loss_soft = ( - targets * (logit_clip_score+outputs_test)/2).sum(dim=1)
            loss_soft = ( - targets * outputs_test).sum(dim=1)
            #loss_soft = ( - targets * predict).sum(dim=1)
            classifier_loss = loss_soft.mean()
            classifier_loss *= cfg.DIFO.CLS_PAR
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        #iic_loss = IID_losses.IID_loss(softmax_out, clip_all_output[tar_idx]) #
        iic_loss = IID_losses.IID_loss_old(softmax_out, clip_all_output)*1.5#64*65,predict_mix换其他
        if iter_num <  (max_iter/2):
            classifier_loss = classifier_loss  + cfg.DIFO.IIC_PAR * iic_loss * 1 #1.5
        else:
            classifier_loss = classifier_loss  + cfg.DIFO.IIC_PAR * iic_loss * 1

        msoftmax = softmax_out.mean(dim=0)

        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.LCFD.EPSILON))
        classifier_loss = classifier_loss - cfg.DIFO.GENT_PAR * gentropy_loss
        with torch.no_grad():
            score_bank[tar_idx] = softmax_out.detach().clone()

        """ if is_clip_train:
            optimizer_clip.zero_grad() """

        optimizer.zero_grad()
        if is_feature_or_prompt_learn:
            if text_features.grad is not None:
                with torch.no_grad():
                    text_features.grad.data.zero_() 

            
        classifier_loss.backward()    
        """ if is_clip_train:
            optimizer_clip.step()
            torch.save(clip_model.prompt_learner.state_dict(),"prompt_model.pt") """
            

        optimizer.step()
        if is_feature_or_prompt_learn:
            with torch.no_grad():
                feature_learning_rate = 0.00001
                text_features -= text_features.grad * feature_learning_rate
        else:
            if iter_num <  (max_iter/2):
                with torch.no_grad():
                    #opt_clip.step()
                    promt_learning_rate = 0.002
                    mypromt.prompt_learner.ctx  -=  mypromt.prompt_learner.ctx.grad * promt_learning_rate
            else:
                 with torch.no_grad():
                    #opt_clip.step()
                    promt_learning_rate = 0.0001
                    mypromt.prompt_learner.ctx  -=  mypromt.prompt_learner.ctx.grad * promt_learning_rate
        #最后2.7几比较好
        if is_new_merge_str:
            with torch.no_grad():
                if iter_num <  (max_iter):
                    if data.grad is not None:
                        if iter_num <  (max_iter/2):
                            data_learning_rate = 0.01     #0.0001
                        else:
                            data_learning_rate = 0.001   
                        data_tmp = torch.tensor(data)
                        data  +=  data.grad * data_learning_rate
                        is_new_stratagem_num =  '0'
                        # match is_new_stratagem_num: #注，0就是啥都不加，不限制
                        #     case '1':
                        #         data[data>1]=1
                        #         data[data<0]=0
                        #         data = data * (torch.mean(data_tmp)/torch.mean(data))
                        #     case '2':
                        #         data = data * (torch.mean(data_tmp)/torch.mean(data))
                        #         data[data>1]=1
                        #         data[data<0]=0
                        #     case '3':
                        #         data[data>1]=1
                        #         data[data<0]=0
                        #     case '4':
                        #         data  +=  data.grad * data_learning_rate
                        #     case '5':
                        #         data = data * (torch.mean(data_tmp)/torch.mean(data))
                        #     case '6':
                        #         data = data * 1.0001
                        #     #case "7":

                        data.requires_grad = True
                else:
                    if data.grad is not None:
                        data_learning_rate = 0.005     #0.0001
                        data  -=  data.grad * data_learning_rate
                        data = data * 1.002
                        data[data>1]=1
                        data[data<0]=0
                        data.requires_grad = True
                if iter_num % 5 == 0:
                    print("The mean of data is:{:.6f}".format(torch.mean(data)))
            #mypromt.prompt_learner.ctx = trainable_param
 
        #text_features = mypromt.named_parameters()

        """  if iter_num % 5 == 0:
            print("The mean of data is:{:.6f}".format(np.mean(data)))
        """
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            #cal_acc_js(dset_loaders['test'],netF,netB,netC,clip_all_output)
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list
                #wandb.log({"accuracy":acc_s_te,"loss ":classifier_loss})
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss)
                #wandb.log({"accuracy":acc_s_te,"loss ":classifier_loss})
            logger.info(log_str)
            netF.train()
            netB.train()

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


def obtain_label(loader, netF, netB, netC,text_inputs,text_features,clip_model,efficient,is_merge_stratg_use):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda() 
            feas = netB(netF(inputs)) 
            outputs = netC(feas)  #output是resnet50网络
            if is_merge_stratg_use: #如果用混合标签，那么这个是true，但实际效果不好,优先false
                if (text_features!=None):
                    #clip_score = clip_text(clip_model,text_features,inputs)
                    efficient = efficient*2
                    if efficient>0.7:
                        efficient=0.7
                    clip_score = (0.3+ efficient) * clip_text(clip_model,text_features,inputs)+ (0.7-efficient) * clip_model(inputs, text_inputs)[0]
                else :
                    clip_score,_ = clip_model(inputs, text_inputs) 
            else:
                if (text_features!=None):
                    clip_score = clip_text(clip_model,text_features,inputs)
                else :
                    clip_score,_ = clip_model(inputs, text_inputs) 

            clip_score = clip_score.cpu()
            if start_test: #第一次，clip网络
                all_output = outputs.float().cpu()
                all_clip_score = clip_score.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)
                
    clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()
    _, predict_clip = torch.max(clip_all_output, 1)  
    accuracy_clip = torch.sum(torch.squeeze(predict_clip).float() == all_label).item() / float(all_label.size()[0]) #ccuracy_clip精度


    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    all_mix_output = (all_output+clip_all_output)/2  #clip和target模型的混合输出

    confi_dis = all_mix_output.detach()
    confi_imag = loader.dataset.imgs
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) #accuracy精度
    log_str = 'Accuracy = {:.2f}% -> CLIP_Accuracy  = {:.2f}%'.format(accuracy * 100, accuracy_clip * 100)
    #wandb.log({"Basenetwork accuracy test": accuracy, "CLIP_Accuracy test": accuracy_clip})
    logging.info(log_str)
    return confi_imag,confi_dis,clip_all_output

def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.DIFO.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda() #把text变为数字信息
    return tokenized_prompts

def clip_text(model,text_features,inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp().cpu()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits

def cal_acc_js(loader, netF, netB, netC, clip_softmax):
    netF.eval()
    netB.eval()
    netC.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs, labels = data[0].cuda(), data[1]

            # target model
            logits_t = netC(netB(netF(inputs)))
            probs_t = F.softmax(logits_t, dim=1)

            # ViL model
            logits_v = clip_softmax
            probs_v = F.softmax(logits_v, dim=1)

            if start_test:
                all_probs_t = probs_t.cpu()
                all_probs_v = probs_v.cpu()
                all_labels = labels
                start_test = False
            else:
                all_probs_t = torch.cat((all_probs_t, probs_t.cpu()), 0)
                all_probs_v = torch.cat((all_probs_v, probs_v.cpu()), 0)
                all_labels = torch.cat((all_labels, labels), 0)

    # target predictions
    _, pred = torch.max(all_probs_t, dim=1)

    correct_mask = (pred == all_labels)
    incorrect_mask = ~correct_mask

    # JS divergence per sample
    js_all = js_divergence(all_probs_t, probs_v.cpu())

    js_correct = js_all[correct_mask].mean().item()
    js_incorrect = js_all[incorrect_mask].mean().item()

    log_str = "js_c:{};js_nc:{};correct_mask:{};ncorrect_mask:{}".format(js_correct,js_incorrect,correct_mask.sum().item(),incorrect_mask.sum().item())
    logging.info(log_str)


def js_divergence(p, q, eps=1e-8):
    """
    p, q: probability distributions, shape [N, C]
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    js = 0.5 * (
        F.kl_div(m.log(), p, reduction='none').sum(dim=1) +
        F.kl_div(m.log(), q, reduction='none').sum(dim=1)
    )
    return js

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import itertools

# --- 关键导入 ---
from src.utils import loss, IID_losses
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList_idx,ImageList_idx_aug_fix
from sklearn.metrics import confusion_matrix
from clip.custom_clip import get_coop # 用于加载带有Prompt Learner的CLIP模型
from src.utils.utils import *

logger = logging.getLogger(__name__)

# --- 数据加载和辅助函数 (与原文件基本一致) ---
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
    dsets["test"] = ImageList_idx_aug_fix(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
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

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
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

# --- 主要修改: 联合训练函数 ---

def train_target(cfg):
    # 1. ========== 模型和数据加载 ==========
    dset_loaders = data_load(cfg)
    if not hasattr(cfg, 'classname') or not cfg.classname:
        name_file_path = os.path.join(cfg.FOLDER, cfg.SETTING.DATASET, 'classname.txt')
        with open(name_file_path) as f:
            classnames_from_file = [line.strip() for line in f.readlines()]
        cfg.classname = classnames_from_file
        print("Successfully loaded classnames from file.")
    # 加载目标模型 (ResNet等)
    if cfg.MODEL.ARCH.startswith('res'):
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH.startswith('vgg'):
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    # 加载预训练的源模型权重
    modelpath = cfg.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    # 加载带Prompt Learner的CLIP模型
    classnames = [name.replace("_", " ") for name in cfg.classname]
    clip_model_prompt = get_coop(cfg.DIFO.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.DIFO.N_CTX, cfg.DIFO.CTX_INIT).cuda()
    clip_model_prompt.reset_classnames(classnames, cfg.DIFO.ARCH) # 设置类名

    # 2. ========== 定义投影层和可训练参数 ==========

    # 定义从CLIP特征空间到更大latent space的投影层
    clip_feature_dim = 512  # ViT-B/32 and RN50 use 512
    if cfg.DIFO.ARCH == "ViT-L/14" or cfg.DIFO.ARCH == "ViT-L/14@336px":
        clip_feature_dim = 768

    larger_latent_dim = 1024 # 您期望的latent space维度

    # 为图像和文本特征创建独立的投影层
    image_projection = nn.Linear(clip_feature_dim, larger_latent_dim).cuda()
    text_projection = nn.Linear(clip_feature_dim, larger_latent_dim).cuda()

    # 冻结不需要训练的参数
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    for name, param in clip_model_prompt.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False) # 只训练prompt_learner

    # 3. ========== 创建统一的优化器 ==========

    # 将所有需要训练的参数组合在一起
    param_group = [
        {'params': netF.parameters(), 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1},
        {'params': netB.parameters(), 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2},
        {'params': clip_model_prompt.prompt_learner.parameters(), 'lr': cfg.OPTIM.LR * 0.1}, # prompt通常用较小的学习率
        {'params': image_projection.parameters(), 'lr': cfg.OPTIM.LR},
        {'params': text_projection.parameters(), 'lr': cfg.OPTIM.LR},
    ]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    # 4. ========== 联合训练循环 ==========
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0

    # 初始化score_bank用于伪标签
    score_bank = torch.randn(len(dset_loaders["target"].dataset), cfg.class_num).cuda()

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test[0].cuda()

        # --- a. 目标模型前向传播 ---
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        # --- b. CLIP & Prompt 前向传播 ---
        with torch.no_grad():
            image_features_clip = clip_model_prompt.encode_image(inputs_test)
            image_features_clip_norm = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)

        text_features_clip = clip_model_prompt.prompt_learner()
        text_features_clip_norm = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)

        logits_clip = 100. * image_features_clip_norm @ text_features_clip_norm.t()
        clip_output_softmax = nn.Softmax(dim=1)(logits_clip)

        # --- c. 投影到更大的Latent Space ---
        mapped_image_feat = image_projection(image_features_clip.float())
        mapped_text_feat = text_projection(text_features_clip.float())

        # --- d. 计算各项损失 ---

        # d.1) 伪标签分类损失 (与原版类似)
        with torch.no_grad():
            K = softmax_out.size(1)
            _, predict_bank = torch.max(score_bank[tar_idx], 1)
            _, predict_clip = torch.max(clip_output_softmax, 1)

            predict_one = torch.nn.functional.one_hot(predict_bank, K).float().cuda()
            clip_one = torch.nn.functional.one_hot(predict_clip, K).float().cuda()

            # 混合伪标签
            mix_ratio = 0.5
            mixed_targets = mix_ratio * predict_one + (1 - mix_ratio) * clip_one

        loss_soft = (- mixed_targets * torch.log_softmax(outputs_test, dim=1)).sum(dim=1)
        classifier_loss = loss_soft.mean() * cfg.DIFO.CLS_PAR

        # d.2) IIC损失 (在两个模型的softmax输出之间)
        iic_loss = IID_losses.IID_loss(softmax_out, clip_output_softmax) * cfg.DIFO.IIC_PAR

        # d.3) G-Entropy损失 (与原版一致)
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.LCFD.EPSILON)) * cfg.DIFO.GENT_PAR

        # d.4) 新增: 投影空间对齐损失 (Projection Alignment Loss)
        # 使用CLIP的伪标签来指导对齐
        mapped_image_feat_norm = mapped_image_feat / mapped_image_feat.norm(dim=1, keepdim=True)
        mapped_text_feat_norm = mapped_text_feat / mapped_text_feat.norm(dim=1, keepdim=True)

        positive_text_feat = mapped_text_feat_norm[predict_clip] # 根据伪标签选取对应的文本特征
        cos_sim = (mapped_image_feat_norm * positive_text_feat).sum(1)
        projection_loss = -cos_sim.mean() * 0.1 # 假设在config中添加PROJ_PAR权重

        # --- e. 合并总损失 ---
        total_loss = classifier_loss + iic_loss - gentropy_loss + projection_loss

        # 更新score_bank
        with torch.no_grad():
            score_bank[tar_idx] = softmax_out.detach().clone()

        # --- f. 优化器步骤 ---
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            clip_model_prompt.eval() # 评估时也切换模式

            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = f'Task: {cfg.name}, Iter:{iter_num}/{max_iter}; Accuracy = {acc_s_te:.2f}%; Total Loss={total_loss.item():.4f}' + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = f'Task: {cfg.name}, Iter:{iter_num}/{max_iter}; Accuracy = {acc_s_te:.2f}%; Total Loss={total_loss.item():.4f}'

            logger.info(log_str)

            netF.train()
            netB.train()
            clip_model_prompt.train() # 训练时切换回train模式
            # 仅prompt learner部分可训练
            for name, param in clip_model_prompt.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)

    if cfg.ISSAVE:
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))
        # 也可以保存prompt和投影层
        torch.save(clip_model_prompt.prompt_learner.state_dict(), osp.join(cfg.output_dir, "prompt_learner_" + cfg.savename + ".pt"))
        torch.save(image_projection.state_dict(), osp.join(cfg.output_dir, "image_proj_" + cfg.savename + ".pt"))
        torch.save(text_projection.state_dict(), osp.join(cfg.output_dir, "text_proj_" + cfg.savename + ".pt"))

    return netF, netB, netC

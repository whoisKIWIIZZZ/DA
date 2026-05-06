import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def image_test(resize_size=256, crop_size=224):
    normalize_trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize_trans
    ])

def load_models_generic(cfg, path_F, path_B):
    from src.models import network # 确保在函数内或顶部引用
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()
        
    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, 
                                   bottleneck_dim=cfg.bottleneck).cuda()
    
    netF.load_state_dict(torch.load(path_F))
    netB.load_state_dict(torch.load(path_B))
    netF.eval()
    netB.eval()
    return netF, netB

def extract_all_features(loader, netF, netB):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data[0].cuda()
            labels = data[1]
            if inputs.size(0) == 1: continue
            # P2 效果通常提取自 bottleneck 的输出
            features = netB(netF(inputs))
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

def run_visualization(cfg):
    save_dir = cfg.output_dir
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    from src.data.data_list import ImageList_idx
    txt_test = open(cfg.test_dset_path).readlines()
    test_loader = DataLoader(ImageList_idx(txt_test, transform=image_test()), 
                             batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    
    base_root = "/data/home/scv8780/run/sfda-my/SNE-MODELS" 
    model_configs = {
        "A": ("source_F.pt", "source_B.pt"),
        "C": ("source_F.pt", "source_B.pt"),
        "difo": ("target_F_difo.pt", "target_B_difo.pt"),
        "ProDe": ("target_F_ProDe.pt", "target_B_ProDe.pt"),
        "diff_align": ("target_F_diff_align_AC_acc=73.77_2025-11-24 14:42:43.115645.pt", "target_B_diff_align_AC_acc=73.77_2025-11-24 14:42:43.115645.pt")
    }

    for folder_name, (f_name, b_name) in model_configs.items():
        path_F, path_B = osp.join(base_root, folder_name, f_name), osp.join(base_root, folder_name, b_name)
        if not osp.exists(path_F): continue

        netF, netB = load_models_generic(cfg, path_F, path_B)
        raw_feats, raw_labels = extract_all_features(test_loader, netF, netB)
        
        # --- 核心改进：为了达到 P2 的几何紧凑感 ---
        # 1. 必须做 L2 归一化，将特征强制约束在超球面上
        raw_feats = normalize(raw_feats, norm='l2')

        for target_cls in [10, 65]:
            mask = raw_labels < target_cls
            if not mask.any(): continue
            x, y = raw_feats[mask], raw_labels[mask]

            # 2. 调整 t-SNE 参数以获得 P2 的“放射状”和“高凝聚力”
            print(f"Calculating t-SNE for {folder_name}...")
            tsne = TSNE(
                n_components=2,
                perplexity=15,             # 减小困惑度：让点更倾向于寻找局部邻居，形成紧凑小簇
                early_exaggeration=20,     # 增大早期夸张系数：让簇在早期分得更开，避免挤在中心
                learning_rate=200,         # 适当的学习率
                n_iter=2500,               # 增加迭代次数确保充分聚拢
                init='pca',                # 保持全局大框架
                random_state=42
            )
            x_tsne = tsne.fit_transform(x)

            # 3. 绘图（完全模仿 P2 风格）
            plt.figure(figsize=(12, 10))
            # 使用较小的点 (s=12) 配合稍微高一点的透明度 (alpha=0.9)
            scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, 
                                 cmap='gist_rainbow', s=12, alpha=0.9)
            
            # P2 风格：没有 colorbar，标题在正上方，纯白背景，无坐标轴
            #plt.title(f"t-SNE Visualization ({target_cls} Classes)", fontsize=16)
            plt.xticks([])
            plt.yticks([])
            # 调整布局，去除边框
            plt.tight_layout()
            
            save_path = osp.join(save_dir, f"{folder_name}_{target_cls}_visual.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")
    # Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import logging
from pickle import TRUE
import torch
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode
import os.path as osp

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()    #类似字典，自定义了新的函数
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Setting - see README.md for more information
# Data directory
_C.DATA_DIR = "/Datasets" #自己修改数据存放路径

# Weight directory
_C.CKPT_DIR = "./Models" #权重存放点

# GPU id
_C.GPU_ID = '0'
# Output directory
_C.SAVE_DIR = "./output_Models"

_C.ISSAVE = True

# Test on source after target adaptation (GSFDA: evaluate source accuracy)
_C.GSFDA = False
# Path to a specific checkpoint
_C.CKPT_PATH = ""

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Optional description of a config
_C.DESC = ""

_C.DA = "uda"

_C.FOLDER = './data/'

_C.NUM_WORKERS = 4

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Some of the available models can be found here:
# Torchvision: https://pytorch.org/vision/0.14/models.html
# timm: https://github.com/huggingface/pytorch-image-models/tree/v0.6.13
# RobustBench: https://github.com/RobustBench/robustbench
_C.MODEL.ARCH = 'resnet101'

_C.MODEL.METHOD = "lcfd"

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

_C.MODEL.EPISODIC = False

_C.MODEL.WEIGHTS = 'IMAGENET1K_V1'
# ----------------------------- SETTING options -------------------------- #
_C.SETTING = CfgNode()

# Dataset for evaluation
_C.SETTING.DATASET = 'office-home'

# The index of source domain
_C.SETTING.S = 0 
# The index of Target domain
_C.SETTING.T = 1

#Seed
_C.SETTING.SEED = 2021

#Sorce model directory
_C.SETTING.OUTPUT_SRC = 'weight_512/seed2021'

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Choices: Adam, SGD
_C.OPTIM.METHOD = "SGD"

# Learning rate
_C.OPTIM.LR = 1e-3

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 5e-4

_C.OPTIM.LR_DECAY1 = 0.1

_C.OPTIM.LR_DECAY2 = 1

_C.OPTIM.LR_DECAY3 = 0.01

# ------------------------------- Test options ------------------------- #
_C.TEST = CfgNode()


# Batch size
_C.TEST.BATCH_SIZE = 64

# Max epoch 
_C.TEST.MAX_EPOCH = 15

# Interval
_C.TEST.INTERVAL = 15

# --------------------------------- SOURCE options ---------------------------- #
_C.SOURCE = CfgNode()

_C.SOURCE.EPSILON = 1e-5

_C.SOURCE.SMOOTH = 0.1

_C.SOURCE.TRTE = 'val'
_C.SOURCE.TRAIN_FRACTION = 1.0
# --------------------------------- NRC options --------------------------- #
_C.NRC = CfgNode()

_C.NRC.K = 5

_C.NRC.KK = 4

_C.NRC.EPSILON = 1e-5

# --------------------------------- SHOT options ---------------------------- #
_C.SHOT = CfgNode()

_C.SHOT.CLS_PAR = 0.3
_C.SHOT.ENT = True
_C.SHOT.GENT = True
_C.SHOT.EPSILON = 1e-5
_C.SHOT.ENT_PAR = 1.0
_C.SHOT.THRESHOLD = 0.0
_C.SHOT.DISTANCE = 'cosine'# ["cosine", "euclidean"]
# --------------------------------- SCLM options ---------------------------- #
_C.SCLM = CfgNode()

_C.SCLM.CLS_PAR = 0.3
_C.SCLM.ENT = True
_C.SCLM.GENT = True
_C.SCLM.EPSILON = 1e-5
_C.SCLM.CLS_SNT = 0.1
_C.SCLM.ENT_PAR = 1.0
_C.SCLM.NEW_ENT_PAR = 0.3
_C.SCLM.DISTANCE = 'cosine'# ["cosine", "euclidean"]
_C.SCLM.THRESHOLD = 0.0
_C.SCLM.INITC_PAR = 0.3
_C.SCLM.CONFI_PAR = 0.3
# --------------------------------- GKD options ---------------------------- #
_C.GKD = CfgNode()

_C.GKD.CLS_PAR = 0.3
_C.GKD.ENT = True
_C.GKD.GENT = True
_C.GKD.EPSILON = 1e-5
_C.GKD.ENT_PAR = 1.0
_C.GKD.THRESHOLD = 0.0
_C.GKD.DISTANCE = 'cosine'# ["cosine", "euclidean"]
# --------------------------------- TPDS options ---------------------------- #
_C.TPDS = CfgNode()

_C.TPDS.EPSILON = 1e-5
_C.TPDS.THRESHOLD = 0.0
_C.TPDS.DISTANCE = 'cosine'# ["cosine", "euclidean"]

# --------------------------------- COWA options ----------------------------- #
_C.COWA = CfgNode()

_C.COWA.ALPHA = 0.2
_C.COWA.WARM = 0.0
_C.COWA.COEFF = 'JMDS' #['LPG', 'JMDS', 'PPL','NO']
_C.COWA.EPSILON = 1e-5
_C.COWA.EPSILON2 = 1e-6
_C.COWA.DISTANCE = 'cosine'# ["cosine", "euclidean"]
_C.COWA.PICKLE = False
# --------------------------------- PLUE options --------------------- #
_C.PLUE = CfgNode()

_C.PLUE.TEMPORAL_LENGTH = 5
_C.PLUE.LABEL_REFINEMENT = True
_C.PLUE.CTR = True
_C.PLUE.EPSILON = 1e-5
_C.PLUE.NEG_L = True
_C.PLUE.REWEIGHTING = True
# _C.PLUE.QUEUE_SIZE = 16384
_C.PLUE.NUM_NEIGHBORS = 10
# ---------------------------------ADACONTRAST  options --------------------- #
_C.ADACONTRAST = CfgNode()

_C.ADACONTRAST.CONTRAST_TYPE = "class_aware"
_C.ADACONTRAST.CE_TYPE = "standard" # ["standard", "symmetric", "smoothed", "soft"]
_C.ADACONTRAST.ALPHA = 1.0  # lambda for classification loss
_C.ADACONTRAST.BETA = 1.0   # lambda for instance loss
_C.ADACONTRAST.ETA = 1.0    # lambda for diversity loss
_C.ADACONTRAST.OPTIM_COS = True
_C.ADACONTRAST.OPTIM_EXP = False
_C.ADACONTRAST.FULL_PROGRESS = 0
_C.ADACONTRAST.SCHEDULE = [10,20]
_C.ADACONTRAST.GAMMA = 0.2
_C.ADACONTRAST.DIST_TYPE = "cosine" # ["cosine", "euclidean"]
_C.ADACONTRAST.CE_SUP_TYPE = "weak_strong" # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.ADACONTRAST.REFINE_METHOD = "nearest_neighbors"
_C.ADACONTRAST.NUM_NEIGHBORS = 10

# --------------------------------- LCFD options ----------------------------- #
_C.LCFD = CfgNode()

_C.LCFD.CLS_PAR = 0.4
_C.LCFD.LOSS_FUNC = 'sce' #['l1',''l2','kl','sce']
_C.LCFD.ENT = True
_C.LCFD.GENT = True
_C.LCFD.EPSILON = 1e-5
_C.LCFD.GENT_PAR = 1.0
_C.LCFD.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.LCFD.N_CTX = 4 
_C.LCFD.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.LCFD.TTA_STEPS = 1
# --------------------------------- DIFO options ----------------------------- #
_C.DIFO = CfgNode()
_C.DIFO.TRAIN_FRACTION = 1.0

_C.DIFO.CLS_PAR = 0.4
_C.DIFO.ENT = True
_C.DIFO.GENT = True
_C.DIFO.EPSILON = 1e-5
_C.DIFO.GENT_PAR = 1.0
_C.DIFO.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.DIFO.N_CTX = 4 
_C.DIFO.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.DIFO.TTA_STEPS = 1
_C.DIFO.IIC_PAR = 1.0
_C.DIFO.LOAD = None
_C.DIFO.MERGE_USE_OBSERVATION_PRIOR = 0
_C.DIFO.MERGE_INITIAL_TARGET_WEIGHT = 0.1
_C.DIFO.MERGE_PRIOR_DECAY = 2.0
_C.DIFO.MERGE_CONFIDENCE_WEIGHT = 0.2
_C.DIFO.MERGE_STABILITY_WEIGHT = 0.8
_C.DIFO.MERGE_GATE_TEMPERATURE = 0.25
_C.DIFO.MERGE_ROBUST_TEMPERATURE = 5.0
_C.DIFO.MERGE_MIN_WEIGHT = 0.05
# ---- adaptive-alpha ablation switches (see src/utils/adaptive_alpha.py) ----
_C.DIFO.MERGE_USE_CONFIDENCE = 1     # 0=可靠性只用稳定性
_C.DIFO.MERGE_USE_STABILITY = 1      # 0=可靠性只用置信度
_C.DIFO.MERGE_USE_PRIOR_TERM = 1     # 0=去掉可靠性先验(只剩loss gap)
_C.DIFO.MERGE_USE_LOSS_GAP = 1       # 0=去掉loss gap(只剩可靠性先验)
_C.DIFO.MERGE_PRIOR_DIRECT = 0       # 1=关掉loss gap后直接用prior不做温度收缩
_C.DIFO.MERGE_FIXED_ALPHA = -1.0     # [0,1]=固定alpha(0.5等权/1.0只用target/0.0只用clip)
_C.DIFO.MERGE_LOSS_GAP_CLAMP = 5.0   # loss gap的截断宽度
_C.DIFO.MERGE_CLIP_USE_STABILITY = 0 # 1=clip侧可靠性也加入clip_stability
# --------------------------------- DIFFADD options ----------------------------- #
_C.DIFFADD = CfgNode()

#使用时最好把DIFFADD和DIFO一致化
_C.DIFFADD.CLS_PAR = 0.4
_C.DIFFADD.ENT = True
_C.DIFFADD.GENT = True
_C.DIFFADD.EPSILON = 1e-5
_C.DIFFADD.GENT_PAR = 1.0
_C.DIFFADD.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.DIFFADD.N_CTX = 4 
_C.DIFFADD.ARCH = 'RN50' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.DIFFADD.TTA_STEPS = 1
_C.DIFFADD.IIC_PAR = 1.0
_C.DIFFADD.LOAD = None

# --------------------------------- DIFF_ALIGN options ----------------------------- #
_C.DIFF_ALIGN = CfgNode()

_C.DIFF_ALIGN.CLS_PAR = 0.4
_C.DIFF_ALIGN.ENT = True
_C.DIFF_ALIGN.GENT = True
_C.DIFF_ALIGN.EPSILON = 1e-5
_C.DIFF_ALIGN.GENT_PAR = 1.0
_C.DIFF_ALIGN.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.DIFF_ALIGN.N_CTX = 4 
_C.DIFF_ALIGN.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.DIFF_ALIGN.TTA_STEPS = 1
_C.DIFF_ALIGN.IIC_PAR = 1.0
_C.DIFF_ALIGN.LOAD = None
_C.DIFF_ALIGN.NUM_BG = 5
_C.DIFF_ALIGN.NUM_SUBCLASS = 5
_C.DIFF_ALIGN.NUM_CLASS_HIDDEN = 10
_C.DIFF_ALIGN.USE_NEG = True
_C.DIFF_ALIGN.LENT_PAR = 0.05
_C.DIFF_ALIGN.MERGE_ADAPTIVE = False  # 旧版随机融合开关，minmax 融合不读取
_C.DIFF_ALIGN.MERGE_INITIAL_TARGET_WEIGHT = 0.1  # 每个样本首次出现时优先 CLIP
_C.DIFF_ALIGN.MERGE_PRIOR_DECAY = 2.0  # 初始偏置按样本观测次数衰减
_C.DIFF_ALIGN.MERGE_CONFIDENCE_WEIGHT = 0.2
_C.DIFF_ALIGN.MERGE_STABILITY_WEIGHT = 0.8
_C.DIFF_ALIGN.MERGE_GATE_TEMPERATURE = 0.25
_C.DIFF_ALIGN.MERGE_ROBUST_TEMPERATURE = 5.0  # 越大越接近可靠性 prior
_C.DIFF_ALIGN.MERGE_MIN_WEIGHT = 0.05
_C.DIFF_ALIGN.MERGE_LOW_CONFIDENCE = 0.3  # 仅用于日志诊断
_C.DIFF_ALIGN.MERGE_TARGET_EMA_DECAY = 0.9
# ---- adaptive-alpha ablation switches (see src/utils/adaptive_alpha.py) ----
_C.DIFF_ALIGN.MERGE_USE_CONFIDENCE = 1
_C.DIFF_ALIGN.MERGE_USE_STABILITY = 1
_C.DIFF_ALIGN.MERGE_USE_PRIOR_TERM = 1
_C.DIFF_ALIGN.MERGE_USE_LOSS_GAP = 1
_C.DIFF_ALIGN.MERGE_PRIOR_DIRECT = 0
_C.DIFF_ALIGN.MERGE_FIXED_ALPHA = -1.0
_C.DIFF_ALIGN.MERGE_LOSS_GAP_CLAMP = 5.0
_C.DIFF_ALIGN.MERGE_CLIP_USE_STABILITY = 0

# --------------------------------- BRIDGE options ----------------------------- #
_C.BRIDGE = CfgNode()
_C.BRIDGE.ENABLE = False              # 是否启用bridge实验checkpoint保存
_C.BRIDGE.CP_INTERVAL = 1             # 每隔N个interval保存一次checkpoint
_C.BRIDGE.SAVE_DIR = 'bridge_ckpts'   # checkpoint子目录名
_C.BRIDGE.SAVE_MODEL = True           # 是否同时保存模型权重(供NTK分析)
_C.BRIDGE.NTK_SAMPLES = 10            # NTK计算每类采样数

# --------------------------------- ProDe options ----------------------------- #
_C.ProDe = CfgNode()
_C.ProDe.TRAIN_FRACTION = 1.0
_C.ProDe.TUNE_TEXT_FEATURES = 0
_C.ProDe.TEXT_FEATURE_LR_MULT = 0.1
_C.ProDe.TEXT_FEATURE_ANCHOR_WEIGHT = 0.1
_C.ProDe.TEXT_FEATURE_LOAD = ''

_C.ProDe.ENT = True
_C.ProDe.GENT = True
_C.ProDe.EPSILON = 1e-5
_C.ProDe.GENT_PAR = 0.1
_C.ProDe.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.ProDe.N_CTX = 4 
_C.ProDe.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.ProDe.TTA_STEPS = 1
_C.ProDe.IIC_PAR = 1.3
_C.ProDe.REP_PAR = 0.1
_C.ProDe.REP_TOPK = 5
_C.ProDe.LOAD = None
_C.ProDe.PROXY_WEIGHT= 0.3
_C.ProDe.USE_LEARNED_NEG = True
_C.ProDe.NEG_ALPHA = 0.5
_C.ProDe.OVER_LAMBDA = 1.0
_C.ProDe.OVER_FACTOR = 3
_C.ProDe.USE_ADAPTIVE_ALPHA = 0
_C.ProDe.MERGE_USE_OBSERVATION_PRIOR = 0
_C.ProDe.MERGE_INITIAL_TARGET_WEIGHT = 0.1
_C.ProDe.MERGE_PRIOR_DECAY = 2.0
_C.ProDe.MERGE_CONFIDENCE_WEIGHT = 0.2
_C.ProDe.MERGE_STABILITY_WEIGHT = 0.8
_C.ProDe.MERGE_GATE_TEMPERATURE = 0.25
_C.ProDe.MERGE_ROBUST_TEMPERATURE = 5.0
_C.ProDe.MERGE_MIN_WEIGHT = 0.05
# ---- adaptive-alpha ablation switches (see src/utils/adaptive_alpha.py) ----
_C.ProDe.MERGE_USE_CONFIDENCE = 1
_C.ProDe.MERGE_USE_STABILITY = 1
_C.ProDe.MERGE_USE_PRIOR_TERM = 1
_C.ProDe.MERGE_USE_LOSS_GAP = 1
_C.ProDe.MERGE_PRIOR_DIRECT = 0
_C.ProDe.MERGE_FIXED_ALPHA = -1.0
_C.ProDe.MERGE_LOSS_GAP_CLAMP = 5.0
_C.ProDe.MERGE_CLIP_USE_STABILITY = 0

_C.LEARNABLE_NEG = CfgNode()
_C.LEARNABLE_NEG.TRAIN = True
# --------------------------------- DIFO++ options ----------------------------- #
_C.DIFOPLUS = CfgNode()

_C.DIFOPLUS.CLS_PAR = 0.4
_C.DIFOPLUS.ENT = True
_C.DIFOPLUS.GENT = True
_C.DIFOPLUS.EPSILON = 1e-5
_C.DIFOPLUS.GENT_PAR = 1.0
_C.DIFOPLUS.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.DIFOPLUS.N_CTX = 4 
_C.DIFOPLUS.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.DIFOPLUS.TTA_STEPS = 1
_C.DIFOPLUS.IIC_PAR = 1.0
_C.DIFOPLUS.LOAD = None
_C.DIFOPLUS.LENT_PAR = 0.05
# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args():
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--cfg", dest="cfg_file",default="cfgs/imagenet_a/sclm.yaml", type=str,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    args = parser.parse_args()
    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.bottleneck = 512
    if cfg.SETTING.DATASET == 'office-home':
        cfg.domain = ['Art', 'Clipart', 'Product', 'RealWorld']
        cfg.class_num = 65 
        cfg.name_file = './data/office-home/classname.txt'
        cfg.class_num_hidden=30
    if cfg.SETTING.DATASET == 'VISDA-C':
        cfg.domain = ['train', 'validation']
        cfg.class_num = 12
        cfg.class_num_hidden=0 #10
        cfg.name_file = './data/VISDA-C/classname.txt'
    if cfg.SETTING.DATASET == 'office':
        cfg.domain = ['amazon', 'dslr', 'webcam']
        cfg.name_file = './data/office/classname.txt'
        cfg.class_num = 31
        cfg.class_num_hidden=10
    if cfg.SETTING.DATASET == 'imagenet_a':
        cfg.domain = ['target']
        cfg.class_num = 200
        cfg.bottleneck = 2048
        cfg.class_num_hidden = 30
    if cfg.SETTING.DATASET == 'imagenet_r':
        cfg.domain = ['target']
        cfg.class_num = 200
        cfg.bottleneck = 2048
        cfg.class_num_hidden = 30
    if cfg.SETTING.DATASET == 'imagenet_k':
        cfg.domain = ['target']
        cfg.class_num = 1000
        cfg.bottleneck = 2048
        cfg.class_num_hidden = 100
    if cfg.SETTING.DATASET == 'imagenet_v':
        cfg.domain = ['target']
        cfg.class_num = 1000
        cfg.bottleneck = 2048
        cfg.class_num_hidden = 100
    if cfg.SETTING.DATASET == 'domainnet126':
        cfg.domain = ["clipart", "painting", "real", "sketch"]
        cfg.name_file = './data/domainnet126/classname.txt'
        cfg.class_num = 126
        cfg.bottleneck = 256
        cfg.class_num_hidden = 30

    cfg.output_dir_src = os.path.join(cfg.CKPT_DIR,cfg.SETTING.OUTPUT_SRC,cfg.DA,cfg.SETTING.DATASET,cfg.domain[cfg.SETTING.S][0].upper())
    #cfg.output_dir_src = os.path.join()
    
    cfg.output_dir = os.path.join(cfg.SAVE_DIR,cfg.DA,cfg.SETTING.DATASET,cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper(),cfg.MODEL.METHOD)
    cfg.name = cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper()
    cfg.name_src = cfg.domain[cfg.SETTING.S][0].upper()
    g_pathmgr.mkdirs(cfg.output_dir)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.output_dir, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])


    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda
               ]
    logger.info("PyTorch Version: torch={}, cuda={}".format(*version))
    #logger.info(cfg)


def complete_data_dir_path(root, dataset_name):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_a": "imagenet-a",
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "domainnet126": "DomainNet-126", # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "office31": "office-31",
               "visda": "visda-2017",
               "cifar10": "",  # do not change the following values
               "cifar10_c": "",
               "cifar100": "",
               "cifar100_c": "",
               "imagenet_v": "imagenetv2-matched-frequency-format-val"
               }
    return os.path.join(root, mapping[dataset_name])


def get_domain_sequence(ckpt_path):
    assert ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt')
    domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
    mapping = {"real": ["clipart", "painting", "sketch"],
               "clipart": ["sketch", "real", "painting"],
               "painting": ["real", "sketch", "clipart"],
               "sketch": ["painting", "clipart", "real"],
               }
    return mapping[domain]


def adaptation_method_lookup(adaptation):
    lookup_table = {"source": "Norm",
                    "norm_test": "Norm",
                    "norm_alpha": "Norm",
                    "norm_ema": "Norm",
                    "ttaug": "TTAug",
                    "memo": "MEMO",
                    "lame": "LAME",
                    "tent": "Tent",
                    "eata": "EATA",
                    "sar": "SAR",
                    "adacontrast": "AdaContrast",
                    "cotta": "CoTTA",
                    "rotta": "RoTTA",
                    "gtta": "GTTA",
                    "rmt": "RMT",
                    "roid": "ROID",
                    "proib": "Proib"
                    }
    assert adaptation in lookup_table.keys(), \
        f"Adaptation method '{adaptation}' is not supported! Choose from: {list(lookup_table.keys())}"
    return lookup_table[adaptation]

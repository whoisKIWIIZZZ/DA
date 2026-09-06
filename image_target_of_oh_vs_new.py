import os
import numpy as np
import torch
import random
import time
import src.methods.oh.nrc as NRC
import src.methods.oh.shot as SHOT
import src.methods.oh.sclm as SCLM
import src.methods.oh.cowa as COWA
import src.methods.oh.gkd as GKD
import src.methods.oh.tpds as TPDS
import src.methods.oh.nrc_vs as Nrc_vs
import src.methods.oh.lcfd as LCFD
import src.methods.oh.difo as DIFO
import src.methods.oh.difo_50 as DIFO_50
#import src.methods.oh.diffadd  as DIFFADD
#import src.methods.oh.diffadd_50  as DIFFADD_50
import src.methods.oh.diffadd_new_opt_old  as DIFFOPT
#import src.methods.oh.diffadd_new_opt_50  as DIFFOPT_50
import src.methods.oh.ProDe as ProDe
# import src.methods.oh.ProDe_diff as ProDe_diff
import src.methods.oh.diff_new_opt_new_50 as DIFF_ALIGN_50
import src.methods.oh.diff_new_opt_new as DIFF_ALIGN
import src.methods.oh.diff_new_opt_new_plus as DIFF_ALIGN_PLUS
import src.methods.oh.draw_visda as DRAW_VISDA
import src.methods.oh.source_clip as SOURCE_CLIP
import src.methods.oh.source_neg as SOURCE_NEG
import src.methods.oh.plue as PLUE
import src.methods.oh.adacontrast as ADACONTRAST
import src.methods.oh.source as SOURCE
import src.methods.oh.zeroshot_neg_test as ZSNEG
import src.utils.plot as PLOT
import src.methods.oh.difo_plus as DIFO_PLUS
import src.methods.oh.difo_plus_partial as DIFO_PLUS_PARTIAL
import src.methods.oh.difo_plus_open as DIFO_PLUS_OPEN
# import src.methods.oh.ProDe_partial as PRODE_PARTIAL
# import src.methods.oh.ProDe_open as PRODE_OPEN
from conf import cfg, load_cfg_from_args
from setproctitle import setproctitle
import warnings
 

import wandb

if __name__ == "__main__":

    
    os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
    # wandb.login(key="c6fbc8b5e3ec04517513fcec32d3d3192582e830")
    # wandb.init(
    # # set the wandb project where this run will be logged
    # project="sfda",

    # # track hyperparameters and run metadata
    # config={
    # "architecture": "vit",
    # "dataset": "office-100",
    # })
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    setproctitle("sfda-clip")
    load_cfg_from_args()

    print("+++++++++++++++++++IB+++++++++++++++++++++")
    print("cfg.MODEL.METHOD:{}".format(cfg.MODEL.METHOD))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID #在conf.py中使用    cfg.type = cfg.domain
    cfg.t_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
    cfg.test_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
    cfg.s_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.S] + '_list.txt'
    cfg.savename = cfg.MODEL.METHOD
    start =time.time()
    torch.manual_seed(cfg.SETTING.SEED)
    torch.cuda.manual_seed(cfg.SETTING.SEED)
    np.random.seed(cfg.SETTING.SEED)
    random.seed(cfg.SETTING.SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    if cfg.SETTING.DATASET == 'office-home':
        if cfg.DA == 'pda':
            cfg.class_num = 65
            cfg.src_classes = [i for i in range(65)]
            cfg.tar_classes = [i for i in range(25)]
        elif cfg.DA == 'open':
            cfg.class_num = 26
            cfg.src_classes = [i for i in range(25)]
            cfg.tar_classes = [i for i in range(65)]
        # pda/open use the closed-set (uda) source model
        if cfg.DA in ('pda', 'open'):
            cfg.output_dir_src = cfg.output_dir_src.replace('/' + cfg.DA + '/', '/uda/')

    if cfg.MODEL.METHOD == "nrc":
        print("using nrc method")
        acc = NRC.train_target(cfg)

    elif cfg.MODEL.METHOD == "shot":
        print("using shot method")
        acc = SHOT.train_target(cfg)

    elif cfg.MODEL.METHOD == "sclm":
        print("using sclm method")
        acc = SCLM.train_target(cfg)

    elif cfg.MODEL.METHOD == "cowa":
        print("using cowa method")
        cfg.prefix = '{}_alpha{}_lr{}_epoch{}_interval{}_seed{}_warm{}'.format(
            cfg.COWA.COEFF, cfg.COWA.ALPHA, cfg.OPTIM.LR, cfg.TEST.MAX_EPOCH, cfg.TEST.INTERVAL, cfg.SETTING.SEED, cfg.COWA.WARM
        )
        acc = COWA.train_target(cfg)

    elif cfg.MODEL.METHOD == "gkd":
        print("using gkd method")
        acc = GKD.train_target(cfg)

    elif cfg.MODEL.METHOD == "tpds":
        print("using tpds method")
        acc = TPDS.train_target(cfg)

    elif cfg.MODEL.METHOD == "nrc_vs":
        print("using nrc method")
        acc = Nrc_vs.train_target(cfg)

    elif cfg.MODEL.METHOD == "lcfd":
        print("using lcfd method")
        acc = LCFD.train_target(cfg)

    elif cfg.MODEL.METHOD == "difo":
        print("using difo method")
        if 'RN' in cfg.DIFO.ARCH:
            acc = DIFO_50.train_target(cfg)
        else:
            acc = DIFO.train_target(cfg)

    elif cfg.MODEL.METHOD == "diffadd": #ours
        print("using diffadd method")
        # if 'RN' in cfg.DIFFADD.ARCH:
        #     print("using diffopt_res50")
        #     acc = DIFFOPT_50.train_target(cfg)
        #     print("11111")
        # else:
        #     #acc = DIFFADD.train_target(cfg)
        #     print("using diffopt_res101")
        #     acc = DIFFOPT.train_target(cfg)
        #     print("22222")
        print("using diffopt_res101")
        acc = DIFFOPT.train_target(cfg)
        print("22222")
    
    elif cfg.MODEL.METHOD == "ProDe":
        print("using ProDe method")
        acc = ProDe.train_target(cfg)

    elif cfg.MODEL.METHOD == "ProDe_diff":
        print("using ProDe+diff method (with negative classes)")
        acc = ProDe_diff.train_target(cfg)

    elif cfg.MODEL.METHOD == "plue":
        print("using plue method")
        acc = PLUE.train_target(cfg)

    elif cfg.MODEL.METHOD == "adacontrast":
        print("using adacontrast method")
        acc = ADACONTRAST.train_target(cfg)

    elif cfg.MODEL.METHOD == "source":
        print("training source model")
        acc = SOURCE.train_source(cfg)

    elif cfg.MODEL.METHOD == "diff_align":
        print("using DIFF_ALIGN method.")
        if 'RN' in cfg.DIFF_ALIGN.ARCH:
            print("RN")
            acc = DIFF_ALIGN_50.train_target(cfg)
        else:
            acc = DIFF_ALIGN_PLUS.train_target(cfg)
        # print("==================not use CLS===================================")
        # acc = DIFF_ALIGN.train_target(cfg,beta_pre_1 = 1,beta_pre_2 =0.1,use_iid = True,use_cls=False,use_clip = True)
        # print("==================not use IID===================================")
        # acc = DIFF_ALIGN.train_target(cfg,beta_pre_1 = 1,beta_pre_2 =0.1,use_iid = False,use_cls=True,use_clip = False)
        # print("==================not use PROMPT TUNING===================================")
        # acc = DIFF_ALIGN.train_target(cfg,beta_pre_1 = 1,beta_pre_2 =0.1,use_iid = True,use_cls=True,use_clip = False)
        # print("==================not use AAS===================================")
        # acc = DIFF_ALIGN.train_target(cfg,beta_pre_1 = 0,beta_pre_2 =0.1,use_iid = True,use_cls=True,use_clip = True)       
    elif cfg.MODEL.METHOD == "source_neg":
        print("train with negative prompts")
        acc = SOURCE_NEG.train_neg_proxy(cfg)
    elif cfg.MODEL.METHOD == "source_clip":
        print("train the neg clip")
        acc = SOURCE_CLIP.train_clip_prompt_dual(cfg)
    elif cfg.MODEL.METHOD == "plot":
        print("fkyyl")
        PLOT.run_visualization(cfg)
    elif cfg.MODEL.METHOD == "draw":
        print("yes")
        acc = DRAW_VISDA.train_target(cfg)
    elif cfg.MODEL.METHOD == "difo_plus":
        print("using difo_plus method")
        acc = DIFO_PLUS.train_target(cfg)

    elif cfg.MODEL.METHOD == "difo_plus_partial":
        print("using difo_plus_partial method")
        acc = DIFO_PLUS_PARTIAL.train_target(cfg)

    elif cfg.MODEL.METHOD == "difo_plus_open":
        print("using difo_plus_open method")
        acc = DIFO_PLUS_OPEN.train_target(cfg)

    elif cfg.MODEL.METHOD == "ProDe_partial":
        print("using ProDe_partial method")
        acc = PRODE_PARTIAL.train_target(cfg)

    elif cfg.MODEL.METHOD == "ProDe_open":
        print("using ProDe_open method")
        acc = PRODE_OPEN.train_target(cfg)

    elif cfg.MODEL.METHOD == "zeroshot_neg":
        print("testing zero-shot negative prompts")
        ZSNEG.evaluate_zeroshot_neg(cfg)

    elif cfg.MODEL.METHOD == "semantic_prior":
        print("testing semantic prior (class_sim propagation)")
        ZSNEG.evaluate_semantic_prior(cfg)

    elif cfg.MODEL.METHOD == "learnable_neg":
        print("training learnable neg embed on S, testing on T")
        ZSNEG.evaluate_learnable_neg(cfg)

    end = time.time()
    all_time = 'Running time: %s Seconds'%(round(end-start, 2))
    print(all_time)

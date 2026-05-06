from copy import deepcopy
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from src.utils import IID_losses
from clip.custom_clip import get_coop
from data.datautils_domain import  build_dataset
from data.cls_to_names import *
from data.domain_datasets import domain_datasets

from data.fewshot_datasets import fewshot_datasets
from data.imagnet_prompts import imagenet_classes

from src.utils.utils import *
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip import load, tokenize
import clip

_tokenizer = _Tokenizer()
DOWNLOAD_ROOT='/data/wangyanbo/shouzhuoyi/sfda-my/clip-module'

logger = logging.getLogger(__name__)
import wandb
import copy


#用于输出clip模型在loader数据集下的精度
def CLIP_model_eval(loader, text_inputs, text_features, clip_model): # 传入C类数量
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        
        # --- 1. Accumulate Scores and Labels ---
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1] # 假设 labels 的范围是 [0, C-1] (正类标签)
            inputs = inputs.cuda() 

            if (text_features!=None):
                # 假设 clip_text 返回的 clip_score 形状为 (batch_size, 2*C)
                clip_score = clip_text(clip_model, text_features, inputs) 
            else :
                # 假设 clip_model(inputs, text_inputs)[0] 形状为 (batch_size, 2*C)
                clip_score, _ = clip_model(inputs, text_inputs) 
                
            clip_score = clip_score.cpu()
           # print(clip_score.shape)
            #print(text_features.shape)
            if start_test:
                all_clip_score = clip_score.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_label = torch.cat((all_label, labels.float()), 0)
                all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)

        N = all_label.size()[0]
        
        C=65
        # --- 2. Split Scores ---
        # 正类分数: 前 C 个维度 (0 to C-1)
        positive_scores = all_clip_score[:, :C]
        
        # 负类分数: 后 C 个维度 (C to 2C-1)
        negative_scores = all_clip_score[:, C:]

        
        # --- 3. Evaluate Positive Classes (Max Score == Ground Truth) ---
        #print(f"Total samples (N): {N}, Positive Classes (C): {C}")
        
        # 应用 Softmax
        positive_all_output = nn.Softmax(dim=1)(positive_scores).cpu()
        
        # 找到正类分数中的最高分对应的索引
        _, predict_positive = torch.max(positive_all_output, 1)  
        
        # 计算正类准确率：如果预测的最高分索引 == 实际标签，则正确
        accuracy_positive = torch.sum(torch.squeeze(predict_positive).float() == all_label).item() / float(N)
        log_str_pos = 'Positive Classes Accuracy (Max Score is GT) = {:.2f}%'.format(accuracy_positive * 100)
        #logging.info(log_str_pos)

        
        # --- 4. Evaluate Negative Classes (Min Score == Ground Truth) ---
        
        # 负类评估逻辑：最高分数对应的是负类，如果 '最低分' 对应的是 'ground truth'，则算对
        # 1. 对负类分数取负号：因为我们要找最低分，取负号后找最高分就等价于找原始的最低分。
        # 2. **注意：** 负类评估需要确保负类的索引与 all_label 对齐。
        #    如果 all_label 只有 0 到 C-1，并且我们期望负类得分中最低的那个索引 (0 to C-1) 
        #    与 all_label 匹配，这意味着：样本 $i$ 的标签是 $\text{GT}_i$。 
        #    如果 $\text{negative\_scores}[i]$ 在索引 $\text{GT}_i$ 处的值是**最小**的，则算对。
        
        # 取负号，将寻找 Min 转化为寻找 Max
        negative_softmax = nn.Softmax(dim=1)(negative_scores)
        inverted_negative_scores = negative_scores
        
        # 找到 inverted_negative_scores 中的最高分对应的索引
        _, predict_negative_min = torch.min(inverted_negative_scores, 1) # 此时 predict_negative_min 对应原始 negative_scores 中的最低分索引 (0 to C-1)
        
        # 计算负类准确率：如果预测的最低分索引 == 实际标签，则正确
        accuracy_negative = torch.sum(torch.squeeze(predict_negative_min).float() == all_label).item() / float(N)
        log_str_neg = 'Negative Classes Accuracy (Min Score is GT) = {:.2f}%'.format(accuracy_negative * 100)
        #logging.info(log_str_neg)

        # --- 5. Return Results ---
        
        # 由于 Softmax 是对 2C 维度的分数进行的，所以这里返回 clip_all_output (2C 维度)
        clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()
        
        # 打印总体的 Max 准确率 (作为参考)
        _, predict_clip_max = torch.max(clip_all_output[:, :C], 1)  
        accuracy_clip_max = torch.sum(torch.squeeze(predict_clip_max).float() == all_label).item() / float(N)
        
        #logging.info('-' * 50)
        logging.info(log_str_pos)
        logging.info(log_str_neg)
        #logging.info(f"Overall Max-Score Accuracy (using only first C classes): {accuracy_clip_max * 100:.2f}%")
        #logging.info('-' * 50)
        
        # 也可以返回分开的 Softmax 输出，这里返回完整的 2C 维度 Softmax
        return positive_all_output,negative_softmax

#用于输出混合标签,并构造计算图(主要针对text_feature)     
class promote_learning_diff_feature_learning(torch.nn.Module):
    def __init__(self, cfg, clip_model, text_features =None):
        super(promote_learning_diff_feature_learning,self).__init__()
        self.clip_model = clip_model
        self.cfg = cfg
        self.text_features = nn.Parameter(None,requires_grad= True).cuda()


    def forward(self, img_inputs):   
        with torch.no_grad():
            #print(self.text_features.shape[0])
            if self.text_features.shape[0] == 0:
                _, text_features = self.text_feature_gen(self.cfg, self.clip_model, img_inputs, "offfice-home") #这个office-home不一定准确，但不管
            img_inputs = img_inputs.cuda()
            image_features = self.clip_model.encode_image(img_inputs)
            logit_scale = self.clip_model.logit_scale.data
            logit_scale = logit_scale.exp().cuda()
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if self.text_features.shape[0] == 0:
            self.text_features = text_features
            self.text_features.requires_grad = True
        logits = logit_scale * image_features @ self.text_features.t()
        #logits = logit_scale *torch.cosine_similarity(image_features, self.text_features, dim =0)
        return logits, self.text_features

    def text_feature_gen(self, cfg, clip_model, img_inputs, test_set, ctx_init = "a_phota_of_a", n_ctx = 4):
        img_inputs = img_inputs.cuda()
        #get textfeature
        learned_cls = False
        ctx_position = 'end'

        text_features = []
        if test_set in fewshot_datasets:
            classnames = eval("{}_classes".format(test_set.lower()))
        elif test_set == 'bongard':
            if learned_cls:
                classnames = ['X', 'X']
            else:
                classnames = ['True', 'False']
        else:
            classnames = imagenet_classes #office-home等走这个判断
        
        # torch.save(prompts, 'learned_prompts.pt')
        # torch.save(prompts, 'learned_tokenized.pt')
        #注册prompt
        prompt_learner = PromptLearner(clip_model, classnames, None, n_ctx, ctx_init, ctx_position, learned_cls)
        prompt_learner.reset_classnames(cfg.classname, cfg.DIFO.ARCH)
        prompts = prompt_learner() #65*77*512
        text_encoder = TextEncoder(clip_model)

        tokenized_prompts = prompt_learner.tokenized_prompts #65*77
        t_features = text_encoder(prompts, tokenized_prompts) 
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)[0,:,:]

        with torch.no_grad():
            image_features = clip_model.visual(img_inputs.type(clip_model.visual.conv1.weight.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = clip_model.logit_scale.data.exp()

        logits = logit_scale * image_features @ text_features.t()
        return logits,text_features
  
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

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

#PromptLearner
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

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init: #一般是true
            # use given words to initialize context vectors
            #print("Initializing the contect with given words: [{}]".format(ctx_init))
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
            prompt = tokenize(ctx_init).to(self.device) #1*77
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) #1*77*512
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] #权重4*512
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        #print(f'Initial context: "{prompt_prefix}"')
        #print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D) 4*512，512是权重clip
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls: #clip模型走这一条
            classnames = [name.replace("_", " ") for name in classnames] #1000
            name_lens = [len(_tokenizer.encode(name)) for name in classnames] #1000
            prompts = [prompt_prefix + " " + name + "." for name in classnames] #1000
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized 4*512

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device) #1000*77
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) #1000*77*512

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames
        self.clip_model = clip_model

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_prompt(self,args,ent_interval,adv_dic,pesu_label):
        batch_ent = torch.sum(-pesu_label * torch.log(pesu_label + args.epsilon), dim=1).mean(0)
        batch_ent = batch_ent.numpy().tolist()
        # for i in range(len(adv_dic)):
        if (batch_ent>=ent_interval[0] and batch_ent<ent_interval[1]):
            newctx_con = adv_dic[0]
            print(newctx_con)
        elif (batch_ent>=ent_interval[1] and batch_ent<ent_interval[2]):
            newctx_con = adv_dic[1]
            print(newctx_con)
        elif (batch_ent>=ent_interval[2] and batch_ent<ent_interval[3]):
            newctx_con = adv_dic[2]
            print(newctx_con)
        elif (batch_ent>=ent_interval[3] and batch_ent<ent_interval[4]):
            newctx_con = adv_dic[3]
            print(newctx_con)
        elif(batch_ent>=ent_interval[4]):
            newctx_con = adv_dic[3]
            print(newctx_con)
        elif(batch_ent<=ent_interval[0]):
            newctx_con = adv_dic[0]
            print(newctx_con)
        # List_rd = classname
        # space = " "
        # indices = indices.squeeze(0).cpu().numpy().tolist()
        # newctx = [List_rd[i] for i in indices]
        # newctx_con = space.join(newctx)
        n_ctx = len(newctx_con.split(" "))
        prompt = tokenize(newctx_con).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        # self.ctx_init_state = torch.cat((ctx_vectors,self.ctx_init_state[n_ctx:self.ctx_init_state.shape[0],:].float()), 0)
        ctx_init_state_temp = torch.cat((self.ctx_init_state[0:2,:].float(),ctx_vectors),0)
        self.ctx_init_state = torch.cat((ctx_init_state_temp,self.ctx_init_state[3:,:].float()),0)
        # print(self.ctx_init_state.shape)
        # self.ctx_init_state = torch.cat((ctx_vectors,self.ctx_init_state[n_ctx:self.ctx_init_state.shape[0],:].float()), 0)
        # self.n_ctx = self.ctx_init_state.shape[0]
        # print(newctx_con)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx  #4*512
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix #65*1*512
        suffix = self.token_suffix #65*72*512
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else: #这里进入
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts #65*77*512

def clip_text(model, text_features, inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp().cpu()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits

#用于输出混合标签,并构造计算图(主要针对text_feature)     
class promote_learning_diff_promote_learning(torch.nn.Module):
    def __init__(self, cfg, clip_model):
        super(promote_learning_diff_promote_learning, self).__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.device = clip_model.visual.conv1.weight.device

        classnames = cfg.classname
        ctx_position = 'end'
        learned_cls = False

        self.positive_prompt_learner = PromptLearner(
            cfg, classnames, clip_model, None, n_ctx=4, ctx_init="a_photo_of_a", 
            ctx_position=ctx_position, learned_cls=learned_cls
        )

        if cfg.class_num_hidden > 0:
            self.negative_prompt_learner = PromptLearner(
                cfg, classnames, clip_model, None, n_ctx=4, ctx_init="", 
                ctx_position=ctx_position, learned_cls=learned_cls
            )
        else:
            self.negative_prompt_learner = None

        self.text_encoder = TextEncoder(self.clip_model)

    def forward(self, img_inputs):

        image_features = self.clip_model.encode_image(img_inputs.type(self.clip_model.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        positive_prompts = self.positive_prompt_learner()
        positive_tokenized = self.positive_prompt_learner.tokenized_prompts
        positive_text_features = self.text_encoder(positive_prompts, positive_tokenized)

        if self.negative_prompt_learner is not None:
            negative_prompts = self.negative_prompt_learner()
            negative_tokenized = self.negative_prompt_learner.tokenized_prompts
            negative_text_features = self.text_encoder(negative_prompts, negative_tokenized)

            all_text_features = [positive_text_features, negative_text_features]
            text_features_all = torch.cat(all_text_features, dim=0)
        else:
            text_features_all = positive_text_features
        
        text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features_all.t()
        
        return logits, text_features_all


   
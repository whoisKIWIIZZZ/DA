# TIMELINE
## 5.11 
写了`diff_new_opt_new_plus`,融合了DIFO++的逻辑.

preprocessing work!!之前的精度都基于把错误的预处理数据送给了CLIP,这导致了精度上不去(DIFO就存在问题,但是后续修复了)

ACcase上,DIFO++的代码跑出来比他汇报的要好的多(73.2汇报,但跑出来是峰值74.0,结尾74.0),将他们的创新点直接加进了我们的方法,结果是(峰值75.2,结尾74.6).也许优化一下可以得到更好的结果.

我们的方法其实真正涨点的还是把CLIP和target联合训练了,后面提的权重矩阵其实还是对涨点没啥直截了当的提升(或者说,这些操作都是试图修复联合训练可能产生的问题).

> 所以负类和权重矩阵不重要?根据之前的认识,负类权重矩阵只是给IID loss权重降低,才能在AC case优化的这么好.IID loss的权重能在CASE里面自适应吗?也许用source的知识可以定一个(IID loss有没有用的)标准,然后自适应调整权重?

> DIFO++仍然没有解决在初始准确率高的时候ProDe更厉害的问题.同理,我们也没解决.

未来也许能进一步的方向:
- netC的初始化.Kaiming初始化 vs 利用source语义的初始化有无区别?
- 负类prompt的初始化.如果负类的not本身看起来不是一个重要的问题,有没有别的初始化方法?
- 关于权重矩阵.difo++的思路是处理难样本,权重矩阵能不能进一步被设计,不难的时候就高权重,用来拟合ProDe?



## 5.16
- ProDe中logits bank存的是softmax,所以论文那个创新点完全是错的,他把CLIP直接变成了`CLIP+target`然后softmax然后IID loss.先不说这俩是不是在一个数量级,这么做有意义吗?我把那个减法改成了加法,效果完全一样
```
 outputs=nn.Softmax(dim=1)(outputs)
logtis_bank[indx] = outputs.detach().clone()
```
- 发现CLIP否定prompt达不到一个好语义;尝试并成功在source域上训练了一个负类别的CLIP代理模型,本质就是一个小的MLP,接受CLIP图像head的特征然后输出一个65维度的分数,受到https://arxiv.org/pdf/2511.19147 这一篇的启发.
- DIFO++也是没用的.


## 5.19
## 完整思路回顾
研究ProDe,发现核心在CLIP + source_model 直接相加,而且不带任何加权的相加.观察到 ensemble 有效，想找到更系统的方式利用 CLIP 的能力。

---

### 第一阶段：训练 Negative Proxy（source_neg.py）

**动机**：CLIP 特征空间 domain-invariant，能否训练一个 proxy 输出"这个样本不属于某类"的先验。

**尝试**：
- v1：BCE + 100x 权重补偿 → 退化成普通分类器
- v2：Margin ranking loss + Prototype contrastive + Domain consistency 正则

**死因**：`NegProxyLinear` 是在 CLIP 特征上的线性层，和 `clip_score`（也是 CLIP 特征的线性变换）**完全共线**，没有引入任何增量信息。

---

### 第二阶段：Text-guided Prior

**动机**：既然线性层共线，能否用 CLIP 文本空间的**非线性结构**挖掘 negative prior？

**思路**：
```
class_sim = text_feat @ text_feat.T    # 类间语义相似度矩阵
neg_prior  = softmax(clip_score) @ class_sim   # 二阶语义传播
corr       = neg_prior - p             # 正确类为负，错误类为正
```

**理论上**：corr 是二阶信息，和 clip_score 不完全共线。

**实验结果**：
```
fixes 0.0% of CLIP's 1538 mistakes
```
完全无效。

---

### 第三阶段：量纲分析和数值问题

**发现**：
```
term_b(clip_score): mean=21  std=2.75
term_c(residual):   mean=-95 std=76    ← 爆炸
```

尝试 log 变换 + logit_scale 缩放 → 数值不稳定，std 达到 60+，因为 log 放大了接近 0 的概率的数值误差。

**结论**：log 变换这条路不通，直接用概率空间的 corr 更稳定。

---

### 第四阶段：诊断 Ensemble 的真正瓶颈

**分析**：
```
both_right:        48%
clip_only_right:   12%
src_only_right:    14%
both_wrong:        26%   ← 真正的瓶颈
```

关键发现：
```
ensemble 修了 57/1134 both_wrong（5%）
但破坏了 181+136=317 个原本对的样本
→ ensemble 是净负收益的修正策略
```

```
diff_error top-5 union = 62%
→ 答案就在候选集里，是排序问题不是知识问题
```

```
oracle（选对的模型）= 74~77%
当前 ens = 68~72%
→ 5% 的稳定 gap 来自融合策略不对
```

---

### 第五阶段：路由器

**动机**：能否学一个置信度路由器，判断每个样本该信哪个模型？

**尝试**：margin 路由、entropy 路由、agreement+margin、agreement+entropy

**结果**：
```
margin=71.8%  entropy=72.2%  ens=71.9%
```
路由和直接相加几乎一样。**死因**：两个模型错误时都过度自信，margin/entropy 无法区分"该信谁"，不确定性估计在 hard case 上失效。

---

### 第六阶段：量级对齐（进行中）

**动机**：
```
clip_score: mean=21  std=2.75
src_logits: mean=-0  std=1.23
```
两者量级不对等，直接相加等于 clip 天然权重更大，ensemble 本身就是有偏的。

**方向**：z-score / minmax / softmax 概率空间对齐后再相加，看是否能填补部分 oracle gap。

---

## 核心教训

| 阶段 | 死因 |
|------|------|
| Neg Proxy | 和 clip_score 共线 |
| Text Semantic Prior | 文本拓扑≠视觉混淆 |
| Log变换 | 数值不稳定 |
| 路由器 | 模型过度自信，不确定性失效 |

**根本矛盾**：所有在 CLIP 特征空间内的操作都无法引入真正的增量信息。真正的瓶颈是 both_wrong 的 26% 样本，这些样本两个模型都过度自信地错了，任何融合策略都救不了——除非引入**完全不同来源的第三个信号**。
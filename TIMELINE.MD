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


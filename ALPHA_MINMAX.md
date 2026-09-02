# 动态 Alpha:minmax


训练使用 target 模型与 CLIP 的混合伪标签：

$$
q_i^{mix}=\alpha_i q_i^T+(1-\alpha_i)q_i^C.
$$

其中 $\alpha_i$ 是 target 权重.容易知道,通常训练初期 CLIP 更可靠,后期适应后的 target 更可靠.原方法使用随训练进度增大的随机权重,想用一个minmax问题替换他.

目标是学习逐样本动态权重：

- 初期主要依赖 CLIP；
- 后期逐渐转向 target；
- 根据数据结构调整权重；
- 避免 alpha 塌缩到 0 或 1.
## 式子

记当前 target 学生预测为 $p_\theta(x_i)$.$q_i^T$ 是 target 的历史输出（自蒸馏参考）,$q_i^C$ 是固定的 CLIP 教师输出：

$$
\ell_i^T=\operatorname{CE}(q_i^T,p_\theta(x_i)),\qquad
\ell_i^C=\operatorname{CE}(q_i^C,p_\theta(x_i)).
$$

这里的CE是逐样本的软标签交叉熵，不是 IID/mutual-information 损失;CLIP侧不用CE的原因是,当前IID loss继承了DIFO_PLUS,在[indices_greater]上计算,并不是逐样本的loss,还不如重新算一个.不过这个loss的**形式是不太重要的**.

考虑这样选择alpha,Ber表示伯努利分布

$$
\min_\theta\min_{\alpha_i\in(0,1)}
\left[
\alpha_i\ell_i^T+(1-\alpha_i)\ell_i^C
+\rho D_{KL}\big(\operatorname{Ber}(\alpha_i)\Vert\operatorname{Ber}(\pi_i)\big)
\right].
$$

其中：

- $\pi_i$ 是一个先验,没想好最终形式,但我打算从target、clip的logits出发,提取出一些物理量出来.这是alpha的学习目标.
- 外层最小化使用混合伪标签更新 target 模型.

因此该方法是两阶段交替优化：先固定模型求 alpha,再固定 alpha 更新 target 模型.

至于当前的优化是min-min问题而不是minmax,如果把alpha替换成1-alpha很容易把min-min问题转化成min-max的,所以感觉这个也不重要,重要的是解决pi的设计

## 闭式解

Bernoulli KL 为：

$$
D_{KL}=\alpha_i\log\frac{\alpha_i}{\pi_i}
+(1-\alpha_i)\log\frac{1-\alpha_i}{1-\pi_i}.
$$


参考出处：[KL divergence between Bernoulli distributions](https://statproofbook.github.io/P/bern-kl.html).的公式2.


$$
\begin{aligned}
L(\alpha_i)
={}&\alpha_i\ell_i^T+(1-\alpha_i)\ell_i^C\\
&+\rho\left[
\alpha_i\log\frac{\alpha_i}{\pi_i}
+(1-\alpha_i)\log\frac{1-\alpha_i}{1-\pi_i}
\right].
\end{aligned}
$$



$$
\frac{\partial L}{\partial\alpha_i}
=\ell_i^T-\ell_i^C
+\rho\left[
\log\frac{\alpha_i}{1-\alpha_i}
-\log\frac{\pi_i}{1-\pi_i}
\right].
$$

利用

$$
\log\frac{\alpha_i}{1-\alpha_i}=\operatorname{logit}(\alpha_i),
\qquad
\log\frac{\pi_i}{1-\pi_i}=\operatorname{logit}(\pi_i),
$$

令导数为零并整理,得到：

$$
\operatorname{logit}(\alpha_i)
=\operatorname{logit}(\pi_i)
+\frac{\ell_i^C-\ell_i^T}{\rho}.
$$

两边取 sigmoid,得到闭式解：

$$

\alpha_i^*=\sigma\left(
\operatorname{logit}(\pi_i)
+\frac{\ell_i^C-\ell_i^T}{\rho}
\right)

$$

所以实现难度只是在pi那一块.

## pi如何实现
一个简单的思路是,$\pi_i = \Gamma_{target} - \Gamma_{CLIP}$,其中$\Gamma$是某种统计量.
利用score bank,考虑[JS散度](https://www.bohrium.com/sciencepedia/feynman/keyword/jensen_shannon_divergence),
$$
\Gamma  = \frac15\big(1-\frac{H(p_\theta (x_i))}{\log N}\big) + \frac45 (1 - \frac{\text{JSD}(p_\theta (x_i),q)}{\log 2})
$$
q是score bank,p表示模型forward一次拿到的logits,使用JSD是考虑到KL有方向性.第一项是衡量这个东西是不是one-hot分布.第二项是考虑模型的预测有没有发生剧烈的变动.

注意:
- 针对CLIP,约束前者并没有太大的意义,在实现上,$\Gamma_{CLIP}$并没有第一项;;
- 如果$\pi_i = \Gamma_{target} - \Gamma_{CLIP}$这么定义,很难在一开始的时候保证$\alpha$ 的权重小.所以粗暴地加了一个$-e^{-n/2}$,n表示模型在这个样本上被更新的次数,例如第一次、第二次我们还是更相信CLIP而不是source的logits.之前那个acc截图的实现如下.
$$
\pi_i = \Gamma_{target} - \Gamma_{CLIP} - e^{-n_i/2}
$$
- officehome的某些case,删掉$-e^{-n/2}$没啥影响.但是office31这个数据集有很大的问题,也许是lr的问题,但更多的问题是我们必须长时间信任source,这是与调度想得到的结果背道而驰的(所以DIFO的结果是最好的,比ProDe都好,就因为他是0.1随机取样).officehome有一些大准确率的case删掉后也会掉点这样,但大体来说还好.
- 必须保持15epoch训练.之前汇报过到A域的掉精度问题,发现是简单的欠训练导致的,但是其他case不能double到30epoch,不然会出问题.
- 想起来了

<!-- 
这样 alpha 只能在可靠性先验附近进行小幅修正,不能因单个 batch 的损失差异而剧烈跳变.

## 4. 结构先验

仅比较 CLIP 与 target 的绝对置信度不可靠,因为二者的 logit 尺度不同.当前实现采用“全局 curriculum + 逐样本结构残差”.

原随机策略的期望作为安全锚点：

$$
s_t=0.1+\frac{t}{T},\qquad
\mu_t=s_t(1-e^{-1/s_t}).
$$

结构分数由五类相对信号组成：

$$
r_i=
0.20r_i^{entropy}
+0.15r_i^{margin}
+0.10r_i^{flip}
+0.35r_i^{knn}
+0.20r_i^{prototype}.
$$

分别表示：教师内归一化熵变化、Top-2 margin、标签翻转率、冻结特征空间的邻域一致性,以及类内紧凑性与类间原型间隔.

最终先验为：

$$
\boxed{
\pi_i=\sigma\left(
\operatorname{logit}(\mu_t)
+\delta_{max}\tanh(r_i)
\right)
}
$$

结构残差只在两教师分歧时生效；证据落在熵 margin 内时降低其强度.该设计保留已验证有效的 CLIP→target 全局趋势,同时允许逐样本自适应. -->

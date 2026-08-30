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

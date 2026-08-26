# 定理 A 的真实实验验证：从线性权重到可观测预测空间量

> 目标：在真实 CLIP + ResNet + 真实数据集上验证 §3（定理 A 系列）的机制。
> 训练阶段无标签；实验分析阶段拥有 target 标签 $Y$（但不存在可恢复的 $w_\star$）。
> 核心结论：**不需要 $w_\star$**——理论中 $w_\star$ 只通过真值残差起作用，
> 而残差在带标评测集的预测函数空间中有精确的可观测对应物。

---

## 0. 为什么必须迁移到预测函数空间

真实 CLIP 与 ResNet：参数维数不同、特征空间不同、不存在共享的 $w\in\mathbb R^d$，
故参数空间量 $g=w_{\rm tgt}-w_c$、$u=w_\star-w_c$ **原则上不可观测**。
但二者在同一批样本上输出同维度类别概率，可在样本×类别输出空间 $\mathbb R^{N\times C}$ 中定义对应物。

**关键事实（桥梁的合法性）**：理论的 $\Sigma_t$-内积在函数空间恰是经验内积——
线性情形下 $\langle w,w'\rangle_{\widehat\Sigma}=\frac1N(Xw)^\top(Xw')$ 是恒等式。
因此预测空间不是近似替代，而是 $\Sigma_t$-几何的**原生居所**。

---

## 1. 核心定义（理论—实验词典）

评测集 $\mathcal D=\{(x_n,y_n)\}_{n=1}^N$，$C$ 类，one-hot 标签矩阵 $Y\in\mathbb R^{N\times C}$。
第 $k$ 个 checkpoint 记录概率输出 $P_c(k),\,P_{\rm tgt}(k)\in\mathbb R^{N\times C}$。

| 理论量（参数空间） | 经验对应（预测空间） | 可观测性 |
|---|---|---|
| $g(t)=\Pi_S(w_{\rm tgt}-w_c)$ | $G_k=P_{\rm tgt}(k)-P_c(k)$ | ✅ 无需标签 |
| $u(t)=\Pi_S(w_\star-w_c)$ | $U_k=Y-P_c(k)$ | ✅ 需标签 |
| $\zeta=\Pi_S(w_\star-w_s)$ | $Z=Y-P_{\rm tgt}(0)=Y-P_s$ | ✅ 需标签 |
| $\langle\cdot,\cdot\rangle_{\Sigma_t}$ | $\langle X,W\rangle_{\mathcal D}=\frac1N\mathrm{tr}(X^\top W)$ | ✅ |
| $L_t(w)$ | Brier score $\frac1N\|Y-P\|_F^2$ | ✅ 需标签 |
| $u_0=g_0+\zeta$（§3.6 重参数化） | $U_0=G_0+Z$（**精确恒等**） | ✅ |

注意：$P$ 各行和为 1 ⟹ $G,U,Z$ 各行和为 0，
一切模态分析应显式限制在该 $(C-1)$ 维行子空间内。

---

## 2. 第 0 层验证：精确几何恒等式（无模态、无 $w_\star$、零近似）

Brier loss 下，由 $Y-P_{\rm tgt}=U-G$：

$$\Delta L_{\rm Brier}(k)=\frac1N\|G_k\|_F^2-\frac2N\langle U_k,G_k\rangle_F$$

这是**严格恒等式**（与命题 prop:loss-gap 逐字对应），可声称 exact empirical verification。

**实验**：每个 checkpoint 画三条曲线
- gap cost $Q_k=\frac1N\|G_k\|_F^2$；
- alignment benefit $R_k=\frac2N\langle U_k,G_k\rangle_F$；
- $\Delta_k=Q_k-R_k$（应逐点等于 Brier gap，作为 sanity check）。

**理论预测**：反超发生 $\iff R_k>Q_k$。这已直接验证 §2 的反超几何判据（rem:crossover-geometry）。

**为什么用 Brier 而非 accuracy/CE**：理论是平方损失，只有 Brier 使恒等式严格成立；
概率输出在同一单纯形上、免疫 logit 温度差异。
accuracy/CE 照常报告，但与理论量无代数等价（Brier 改善不保证 accuracy 变化）。

---

## 3. 第 0.5 层验证：信任半径的经验版（联动 §3.6，草稿遗漏的强预测）

$U_0=G_0+Z$ 精确复现定理 thm:A-radius 的重参数化 $u_0=g_0+\zeta$，且

$$\zeta_s^{\rm emp}=\|Z\|_{\mathcal D}=\sqrt{\mathrm{Brier}(P_s)}\qquad(\text{完全可观测}).$$

可直接检验（两边都可算，无需任何模态拟合）：
- **初始领先判据**（thm:A-radius(iii)）：$\|Z\|_{\mathcal D}<\tfrac12\|G_0\|_{\mathcal D}\ \Rightarrow\ \Delta L(0)<0$；
- **可证伪预测**：人为退化 source（欠训练/标签噪声）抬升 $\|Z\|_{\mathcal D}$，
  当其越过半径时反超应消失；增大 $\eta_c$ 半径应收缩。

---

## 4. 模态基的两种构造

### 方法 A：轨迹驱动模态（主实验，推荐）

对 $z_k=\mathrm{vec}(G_k)$：先 PCA 降维 $z_k\approx V_r q_k$，
再拟合**对称半正定生成元**（与理论中 $A$ 的 $\Sigma_t$-自伴正定性一致）：

$$\min_{A_{\rm emp}\succeq0}\ \sum_k\Bigl\|\tfrac{q_{k+1}-q_k}{\Delta t}+A_{\rm emp}\,q_k\Bigr\|^2,$$

对 $A_{\rm emp}$ 谱分解得经验模态 $\{e_i\}$ 与衰减率 $\{\rho_i^{\rm emp}\}$。

> ⚠️ 不建议 $A_{\rm emp}=-\frac1{\Delta t}\log P_r$（DMD 原始形式）：
> $P_r$ 非对称时特征值可为复/负，矩阵对数病态。PSD 约束拟合应为**首选**。

- 优点：无需 Jacobian，架构任意，只存预测，直接读出各模态衰减率；
- 局限：模态是估计量；只能恢复轨迹实际激活的低维子空间；
  强非线性时单一常算子拟合差（见 §7 注意事项：须固定 $\alpha$）。

### 方法 B：NTK 共识算子（小规模机制实验）

局部线性化下 prediction gap 近似满足

$$\dot G\approx-\bigl(\gamma\beta\,J_{\rm tgt}J_{\rm tgt}^\top+\eta_c\,J_cJ_c^\top\bigr)G
=: -A_{\rm NTK}\,G,$$

两个核均作用在 $\mathbb R^{NC}$ 上故可相加——这是理论算子
$A=\gamma\beta\Pi_S\mathcal K\Sigma_t+\eta_cC$ 的切线核对应物
（$J_{\rm tgt}J_{\rm tgt}^\top\leftrightarrow\mathcal K$-预条件，$J_cJ_c^\top\leftrightarrow\Pi_{R_c}$-受限耦合）。

实现：不显式构造 $JJ^\top$；用 JVP+VJP（$(JJ^\top)v=J(J^\top v)$）配合
Lanczos / randomized eigensolver 取前 20–50 个本征模态。
规模控制：每类 20–50 样本、top-$r$ 类别输出。

### 推荐分工

- 主表/多数据集：方法 A（轨迹模态）；
- 机制深挖（小规模）：方法 B，检验 (a) gap 是否主要沿前几模态衰减、
  (b) 预测的 $\rho_i$ 与实际轨迹是否吻合、(c) $a_i$ 符号是否预测各模态优势。

---

## 5. 逐模量与定理 A 的经验检验

给定模态基 $\{e_i\}$（$\langle\cdot,\cdot\rangle_{\mathcal D}$-标准正交）：

$$g_i(k)=\langle G_k,e_i\rangle_{\mathcal D},\quad
u_i(k)=\langle U_k,e_i\rangle_{\mathcal D},\quad
a_i=u_i(0)\,g_i(0).$$

有限 $\eta_c$ 量（方法 B 下）：$C_{\rm emp}=J_cJ_c^\top$，

$$\nu_i=\langle e_i,C_{\rm emp}e_i\rangle,\quad
\rho_i=\langle e_i,A_{\rm NTK}e_i\rangle,\quad
\delta_i=\frac{\eta_c\nu_i}{\rho_i}.$$

**定理 A 的经验判据与 margin**：

$$m_i:=a_i-\min\{\delta_i,\tfrac12\}\,g_i(0)^2,$$

检验：$m_i>0$ 的模态是否产生负贡献 $f_i(k)=g_i(k)^2-2u_i(k)g_i(k)<0$；
$m_i<0$ 的模态是否不产生；margin 大小是否与反超深度/持续时间相关。

**模态贡献分解**（任意截断秩 $r$）：

$$\Delta L_{\rm Brier}(k)=\sum_{i\le r}f_i(k)+R_{\rm residual}(k),$$

报告：各主模态 $f_i(k)$ 曲线、正负贡献堆叠面积图、
前 $r$ 模态解释比例（分母接近 0 时改用重构误差比较，避免除零）。

---

## 6. $u_i$ 用初始 CLIP 还是当前 CLIP？

两者都记录，对应不同验证目标：
- **定理条件**（初值量）：$a_i=u_i(0)g_i(0)$，用 $U_0=Y-P_c(0)$；
- **精确逐时刻分解**（有限 $\eta_c$ 真实系统）：$f_i(k)$ 用 $U_k=Y-P_c(k)$
  （CLIP 在训练中移动，正是 §3 三项分解中"CLIP 位移修正"的来源）。

---

## 7. 实验流程 checklist 与注意事项

1. **数据划分**：adaptation set（只用输入）/ analysis set（机制验证）/ test set（最终 accuracy）。
   声明模板："Target labels are never accessed by the adaptation algorithm and are
   used only for post-hoc mechanism verification."
2. **固定 $\alpha$**（重要，草稿遗漏）：定理 A 假设 $\alpha_t\equiv\alpha$（$A$ 为常算子）。
   机制实验必须跑固定 $\alpha$ 版本；时变调度实验属于 §5 的验证，不要混用。
3. **checkpoint 采样**：对数间隔（指数衰减早期信息密集，均匀采样浪费预算）。
4. **概率校准**：CLIP 固定/记录 logit scale；两模型输出均过 softmax 后再比较。
5. **每个 checkpoint 保存**：$P_c,P_{\rm tgt}$、accuracy、CE、Brier、
   disagreement、$\|G_k\|_{\mathcal D}$。
6. **单纯形约束**：模态分析前把 $G,U,Z$ 投影到行和为零子空间。

---

## 8. 表述规范：能声称什么

| 层级 | 内容 | 可声称的措辞 |
|---|---|---|
| 1 | Brier 恒等式 + 反超几何判据 $R_k>Q_k$ | **exact empirical verification** |
| 1.5 | 信任半径经验版（$\|Z\|$ vs 可观测半径） | direct test of the observable-radius prediction |
| 2 | 轨迹模态 + $a_i$ 符号 + 负 $f_i$ 导致反超 | empirical evidence **consistent with** the modal mechanism |
| 3 | NTK 算子 + $a_i>\delta_ig_i(0)^2$ 与逐模 crossover 对应 | **local** empirical verification **under the tangent-kernel approximation** |

关于 $w_\star$ 的正确表述（禁止写"从标签恢复了 $w_\star$"——非线性网络中不可辨识）：

> 在线性模型中 $w_\star$ 仅通过真值预测残差起作用。对真实多分类网络，
> 我们不尝试恢复参数空间真值，而是在带标评测集的预测函数空间中以
> $U=Y-P_c$ 作为其经验对应量。

---

## 9. 一句话总结

$$\boxed{G=P_{\rm tgt}-P_c,\qquad U=Y-P_c,\qquad Z=Y-P_s}$$

配任一经验模态基（轨迹 PSD 拟合或 NTK 共识算子），
$g_i=\langle G,e_i\rangle_{\mathcal D}$、$u_i=\langle U,e_i\rangle_{\mathcal D}$、$a_i=u_ig_i$
即为定理 A 全部机制变量的可观测版本；标签 $Y$ 已提供 $w_\star$
在有限样本预测空间中的全部作用，无需（也不可能）恢复 $w_\star$ 本身。

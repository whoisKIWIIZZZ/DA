import numpy as np
import matplotlib.pyplot as plt

# 参数
scale = 1
ln_sam = 200_000          # 抽样次数
rng = np.random.default_rng(42)  # 固定随机种子，方便复现

# 从指数分布抽样
samples = rng.exponential(scale=scale, size=ln_sam)

# 设置显示范围。scale=0.1 时，大部分值都在 0~0.8 内
x_max = 0.8
bins = np.linspace(0, x_max, 101)  # 100 个区间

# 手动计算概率密度直方图，避免长尾把前面压扁
counts, bin_edges = np.histogram(samples, bins=bins)
bin_width = bin_edges[1] - bin_edges[0]
pdf_empirical = counts / (ln_sam * bin_width)  # 经验概率密度

# 理论概率密度：f(x) = (1/scale) * exp(-x/scale)
x = np.linspace(0, x_max, 500)
pdf_theory = (1.0 / scale) * np.exp(-x / scale)

# 画图
plt.figure(figsize=(8, 5))

# 抽样直方图
plt.bar(
    bin_edges[:-1],
    pdf_empirical,
    width=bin_width,
    align='edge',
    alpha=0.6,
    color='skyblue',
    edgecolor='white',
    label='抽样直方图'
)

# 理论曲线
plt.plot(
    x,
    pdf_theory,
    'r-',
    linewidth=2,
    label=f'理论 PDF, scale={scale}'
)

plt.xlabel('数值 x')
plt.ylabel('概率密度')
plt.title(f'指数分布抽样：scale={scale}, n={ln_sam}')
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, x_max)
plt.tight_layout()
plt.savefig('./analysis/fk.pdf')
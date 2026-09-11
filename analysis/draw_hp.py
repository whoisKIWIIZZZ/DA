import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

# =========================
# 1. 从日志读取实验结果
# =========================
LOG_DIR = Path(__file__).resolve().parent / "AAProDeee"
FILE_PATTERN = re.compile(r"ProDe_hp_cs_(\d{2})(\d{2})_T(\d+)\.log$")
RESULT_PATTERN = re.compile(
    r"Iter:\s*(\d+)\s*/\s*(\d+)\s*;\s*Accuracy\s*=\s*([\d.]+)%"
)


def load_results(log_dir):
    results = {}
    for log_path in sorted(log_dir.glob("ProDe_hp_cs_*_T*.log")):
        name_match = FILE_PATTERN.fullmatch(log_path.name)
        if name_match is None:
            continue

        alpha_part, complement_part, rho_part = map(int, name_match.groups())
        if alpha_part + complement_part != 10:
            raise ValueError(f"文件名中的参数不满足 X + (10-X) = 10: {log_path.name}")

        final_accuracy = None
        for iteration, total, accuracy in RESULT_PATTERN.findall(
            log_path.read_text(encoding="utf-8", errors="replace")
        ):
            if iteration == total:
                final_accuracy = float(accuracy)

        if final_accuracy is None:
            raise ValueError(f"日志中没有找到最终迭代的 Accuracy: {log_path}")

        parameters = (alpha_part / 10, rho_part)
        if parameters in results:
            raise ValueError(f"参数 alpha={parameters[0]}, rho={parameters[1]} 重复")
        results[parameters] = final_accuracy

    if not results:
        raise ValueError(f"没有在 {log_dir} 中找到符合命名规则的日志")

    alpha = np.array(sorted({parameters[0] for parameters in results}))
    rho = np.array(sorted({parameters[1] for parameters in results}))
    missing = [
        (alpha_value, rho_value)
        for alpha_value in alpha
        for rho_value in rho
        if (alpha_value, rho_value) not in results
    ]
    if missing:
        raise ValueError(f"实验结果网格不完整，缺少参数: {missing}")

    accuracy = np.array(
        [[results[(alpha_value, rho_value)] for rho_value in rho]
         for alpha_value in alpha]
    )
    return alpha, rho, accuracy


alpha, rho, Z = load_results(LOG_DIR)
B, A = np.meshgrid(rho, alpha)

# =========================
# 2. 创建 3D 图
# =========================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 底部等高线所在的 z 坐标
z_offset = np.floor(Z.min()) - 1

# =========================
# 3. 绘制 3D 曲面
# =========================
surf = ax.plot_surface(
    B,
    A,
    Z,
    cmap="viridis",
    edgecolor="none",
    linewidth=0,
    antialiased=True,
    alpha=0.95
)

# =========================
# 4. 底部等高线投影
# =========================
contour = ax.contourf(
    B,
    A,
    Z,
    zdir="z",
    offset=z_offset,
    levels=8,
    cmap="viridis",
    alpha=0.35
)

# =========================
# 5. 坐标轴设置
# =========================
ax.set_xlabel(
    r"$\rho$",
    fontsize=28,
    labelpad=18
)

ax.set_ylabel(
    r"$\alpha$",
    fontsize=28,
    labelpad=18
)

ax.set_zlabel(
    "Accuracy (%)",
    fontsize=22,
    labelpad=15
)

ax.set_xlim(rho.max(), rho.min())
ax.set_ylim(alpha.min(), alpha.max())
ax.set_zlim(z_offset, np.ceil(Z.max()) + 0.5)

# 刻度
ax.set_xticks(rho)
ax.set_yticks(alpha)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=16
)

# =========================
# 6. 调整观察角度
# =========================
ax.view_init(
    elev=25,
    azim=40
)

# =========================
# 7. Colorbar
# =========================
cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.62,
    aspect=12,
    pad=0.10
)

cbar.ax.tick_params(labelsize=16)

# =========================
# 8. 美化
# =========================
ax.grid(True)

plt.tight_layout()

# 保存高清论文图片
plt.savefig(
    Path(__file__).resolve().parent/"hp.pdf",
    bbox_inches="tight"
)

plt.show()

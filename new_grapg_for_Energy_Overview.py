import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

# === Save path (set your target here) ===
save_path = "static/assets/images/download_8_3.png"  # ← Change this path as needed

# === Delete old file if it exists ===
if os.path.exists(save_path):
    os.remove(save_path)

# === Data (kWh) ===
labels = ['Imported', 'Consumption', 'Predicted', 'Produced']
values = [350, 340, 390, 200]
insta_colors = ['#734ff7', '#8667f8', '#9a80f9', '#ad98fa']

# === Setup Figure and Axes ===
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#E6F5E9')   # Outside background
ax.set_facecolor('#E6F5E9')          # Plot background

# === Create placeholder bars ===
bars = ax.bar(range(len(values)), values, width=0.8, color=insta_colors)

# === Replace bars with rounded patches ===
new_patches = []
for bar in bars:
    bb = bar.get_bbox()
    color = bar.get_facecolor()
    patch = FancyBboxPatch(
        (bb.xmin, bb.ymin),
        abs(bb.width), abs(bb.height),
        boxstyle="round,pad=-0.05,rounding_size=0.50",
        ec="none", fc=color,
        mutation_aspect=0.5
    )
    bar.remove()
    new_patches.append(patch)

for patch in new_patches:
    ax.add_patch(patch)

# === Add text labels ===
for i, (x, val) in enumerate(zip(range(len(values)), values)):
    ax.text(
        x, val + 10,
        f"{val} kWh",
        ha='center', va='bottom',
        fontsize=15
    )

# === Label X-axis ===
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=13, fontfamily='sans-serif')

# === Final Styling ===
ax.set_title("Energy Overview (kWh)", fontsize=16)
ax.set_ylim(2, max(values) + 120)
ax.set_yticks([])
ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)

# === Layout and Save ===
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved plot to {save_path}")

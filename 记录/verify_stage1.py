import matplotlib.pyplot as plt

# ---------------------- 1. Prepare Data ----------------------
# Average score data
model_names = ["RGB Baseline", "RGB+Polar Model"]
avg_scores = [4.48, 4.7]

# Score distribution data
score_bins = ["1-3", "4-6", "7-8", "9-10"]
rgb_counts = [162, 91, 55, 44]
polar_rgb_counts = [149, 99, 56, 48]
bar_width = 0.35  # Width of grouped bars
x = range(len(score_bins))

# ---------------------- 2. Set Global Style ----------------------
plt.rcParams["font.size"] = 11

# ---------------------- 3. Create Canvas and Dual Plot Layout ----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# ---------------------- Plot 1: Average Score Comparison ----------------------
bars1 = ax1.bar(model_names, avg_scores, color=["#1f77b4", "#ff7f0e"], width=0.5)
ax1.set_title("Comparison of Model Average Scores", fontsize=14, fontweight="bold", pad=15)
ax1.set_ylabel("Average Score", fontsize=12)
ax1.set_ylim(0, 5.5)  # Fix y-axis range for clearer comparison
ax1.grid(axis="y", linestyle="--", alpha=0.7)

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f"{height}", ha="center", va="bottom", fontweight="bold")

# ---------------------- Plot 2: Score Distribution Comparison ----------------------
bars2_1 = ax2.bar([i - bar_width/2 for i in x], rgb_counts, width=bar_width, label="RGB Baseline", color="#1f77b4")
bars2_2 = ax2.bar([i + bar_width/2 for i in x], polar_rgb_counts, width=bar_width, label="RGB+Polar Model", color="#ff7f0e")

ax2.set_title("Comparison of Sample Distribution by Score Range", fontsize=14, fontweight="bold", pad=15)
ax2.set_xlabel("Score Range", fontsize=12)
ax2.set_ylabel("Number of Samples", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(score_bins)
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.7)

# Add value labels on top of bars
for bar in bars2_1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f"{height}", ha="center", va="bottom", fontsize=10)
for bar in bars2_2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f"{height}", ha="center", va="bottom", fontsize=10)

# ---------------------- 4. Adjust Layout and Show/Save ----------------------
plt.tight_layout(pad=3)
plt.show()
# Uncomment the line below to save the figure
# plt.savefig("model_performance_comparison.png", dpi=300, bbox_inches="tight")
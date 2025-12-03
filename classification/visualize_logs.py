import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 18,     # Title of each graph
    'axes.labelsize': 16,     # X and Y axis labels
    'xtick.labelsize': 14,    # Numbers on X axis
    'ytick.labelsize': 14,    # Numbers on Y axis
    'legend.fontsize': 14,    # Legend text
    'figure.titlesize': 20    # Overall figure title
})

# ==========================================
# 1. INPUT DATA STRUCTURE
# ==========================================
data = {
    "metrics": ["Hit@1", "Hit@3", "Hit@5"],
    "models": {
        "Baseline (Multi-step LLM Inference)": {
            "Hit@1": 0.4667,
            "Hit@3": 0.5133,
            "Hit@5": 0.5133,
            "Precision": 0.3348,
            "Recall": 0.0690,
            "F1": 0.1049,
        },
        "Base GraphCodeBERT": {
            "Hit@1": 0.0100,
            "Hit@3": 0.0233,
            "Hit@5": 0.0400,
            "Precision": 0.0080,
            "Recall": 0.0400,
            "F1": 0.0133,
        },
        "GraphCodeBERT FineTuned (2 epochs)": {
            "Hit@1": 0.1867,
            "Hit@3": 0.3233,
            "Hit@5": 0.4067,
            "Precision": 0.1867,
            "Recall": 0.1867,
            "F1": 0.1867,
        },
    },
}

gemini_data = {
    "metrics": ["Hit@1", "Hit@3", "Hit@5"],
    "models": {
        "Baseline (Multi-step LLM Inference)": {
            "Hit@1": 0.1233,
            "Hit@3": 0.1233,
            "Hit@5": 0.1233,
            "Precision": 0.1233,
            "Recall": 0.0191,
            "F1": 0.0312,
        },
        "Base GraphCodeBERT": {
            "Hit@1": 0.0233,
            "Hit@3": 0.0733,
            "Hit@5": 0.0967,
            "Precision": 0.0256,
            "Recall": 0.0733,
            "F1": 0.0377,
        },
        "GraphCodeBERT FineTuned (2 epochs)": {
            "Hit@1": 0.1933,
            "Hit@3": 0.3200,
            "Hit@5": 0.4000,
            "Precision": 0.1933,
            "Recall": 0.1933,
            "F1": 0.1933,
        },
    },
}


# ==========================================
# 2. PLOTTING LOGIC
# ==========================================
def create_comparison_charts():
    # Define the order of models to appear on the legend/graph
    model_order = [
        "Baseline (Multi-step LLM Inference)",
        "Base GraphCodeBERT",
        "GraphCodeBERT FineTuned (2 epochs)",
    ]
    # Colors: Dark Blue, Red, Green
    colors = ["#2c3e50", "#e74c3c", "#27ae60"]

    # Setup the figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    width = 0.25  # Thinner bars to fit 3 side-by-side

    # --- CHART 1: HIT RATES ---
    labels = data["metrics"]
    x = np.arange(len(labels))

    for i, model_name in enumerate(model_order):
        # Calculate position:
        # i=0 (Baseline) -> x - 0.25
        # i=1 (Base)     -> x
        # i=2 (Tuned)    -> x + 0.25
        position = x + (i - 1) * width

        # Get values and convert to percentages (0-100)
        vals = [data["models"][model_name][m] * 100 for m in labels]

        rects = ax1.bar(position, vals, width, label=model_name, color=colors[i])
        ax1.bar_label(rects, padding=3, fmt="%.1f%%", fontsize=9)

    # Styling Chart 1
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Localization Accuracy (Hit Rate)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_ylim(0, 100)

    # --- CHART 2: IR METRICS ---
    ir_labels = ["Precision", "Recall", "F1"]
    x_ir = np.arange(len(ir_labels))

    for i, model_name in enumerate(model_order):
        position = x_ir + (i - 1) * width
        vals = [data["models"][model_name][m] * 100 for m in ir_labels]

        rects = ax2.bar(position, vals, width, label=model_name, color=colors[i])
        ax2.bar_label(rects, padding=3, fmt="%.1f%%", fontsize=9)

    # Styling Chart 2
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Information Retrieval Metrics")
    ax2.set_xticks(x_ir)
    ax2.set_xticklabels(ir_labels)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_ylim(0, 45)

    # Final Layout Adjustments
    plt.tight_layout()
    plt.savefig("benchmark_comparison_v2.png")
    print("Graph saved as benchmark_comparison_v2.png")
    plt.show()


training_data = {
    "steps": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "epochs": [
        0.1983,
        0.3966,
        0.5949,
        0.7933,
        0.9916,
        1.1884,
        1.3867,
        1.5850,
        1.7833,
        1.9817,
    ],
    "loss": [
        4.0681,
        2.4532,
        2.3908,
        2.4027,
        5.7012,
        2.1844,
        16.1541,
        4.2594,
        2.103,
        2.0968,
    ],
}


def create_training_loss_chart():
    steps = training_data["steps"]
    losses = training_data["loss"]
    epochs = training_data["epochs"]

    plt.figure(figsize=(10, 6))

    # 1. Plot the raw data (The "Spiky" Line)
    plt.plot(
        steps,
        losses,
        marker="o",
        color="#e74c3c",
        label="Batch Loss",
        linewidth=2,
        alpha=0.7,
    )

    # 2. Add annotations for the huge spikes (Contextualizing the noise)
    # The spike at 700 is massive, so we highlight it.
    max_loss_idx = losses.index(max(losses))
    plt.annotate(
        f"Spike: {max(losses)}",
        xy=(steps[max_loss_idx], losses[max_loss_idx]),
        xytext=(steps[max_loss_idx] + 50, losses[max_loss_idx]),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    # 3. Create a smoothed trend line (Moving Average) to show actual convergence
    # This helps show that despite the spike at 700, the model IS learning.
    window_size = 3
    if len(losses) >= window_size:
        smoothed = np.convolve(losses, np.ones(window_size) / window_size, mode="valid")
        # Adjust x-axis for valid convolution
        smooth_steps = steps[window_size - 1 :]
        plt.plot(
            smooth_steps,
            smoothed,
            color="#2c3e50",
            linestyle="--",
            linewidth=2,
            label="Smoothed Trend",
        )

    # 4. Styling
    plt.title("Fine-Tuning Stability (Loss over Time)")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss (MultipleNegativesRankingLoss)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    # 5. Add Epoch markers on top x-axis
    # This lets you see "Oh, the spike happened early in Epoch 2"
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(steps)
    ax2.set_xticklabels([f"Ep {e:.1f}" for e in epochs], rotation=45, fontsize=8)
    ax2.set_xlabel("Epoch Progress")

    plt.tight_layout()
    plt.savefig("training_loss_curve.png")
    print("Graph saved as training_loss_curve.png")
    plt.show()


# ==========================================
# 4. PLOTTING LOGIC
# ==========================================
def create_gemini_comparison_charts():
    # Define the order of models to appear on the legend/graph
    model_order = [
        "Baseline (Multi-step LLM Inference)",
        "Base GraphCodeBERT",
        "GraphCodeBERT FineTuned (2 epochs)",
    ]
    # Colors: Dark Blue, Red, Green
    colors = ["#2c3e50", "#e74c3c", "#27ae60"]

    # Setup the figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    width = 0.25  # Thinner bars to fit 3 side-by-side

    # --- CHART 1: HIT RATES ---
    labels = gemini_data["metrics"]
    x = np.arange(len(labels))

    for i, model_name in enumerate(model_order):
        # Calculate position:
        # i=0 (Baseline) -> x - 0.25
        # i=1 (Base)     -> x
        # i=2 (Tuned)    -> x + 0.25
        position = x + (i - 1) * width

        # Get values and convert to percentages (0-100)
        vals = [gemini_data["models"][model_name][m] * 100 for m in labels]

        rects = ax1.bar(position, vals, width, label=model_name, color=colors[i])
        ax1.bar_label(rects, padding=3, fmt="%.1f%%", fontsize=9)

    # Styling Chart 1
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Localization Accuracy (Hit Rate)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_ylim(0, 100)

    # --- CHART 2: IR METRICS ---
    ir_labels = ["Precision", "Recall", "F1"]
    x_ir = np.arange(len(ir_labels))

    for i, model_name in enumerate(model_order):
        position = x_ir + (i - 1) * width
        vals = [gemini_data["models"][model_name][m] * 100 for m in ir_labels]

        rects = ax2.bar(position, vals, width, label=model_name, color=colors[i])
        ax2.bar_label(rects, padding=3, fmt="%.1f%%", fontsize=9)

    # Styling Chart 2
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Information Retrieval Metrics")
    ax2.set_xticks(x_ir)
    ax2.set_xticklabels(ir_labels)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_ylim(0, 45)

    # Final Layout Adjustments
    plt.tight_layout()
    plt.savefig("benchmark_comparison_v2_gemini.png")
    print("Graph saved as benchmark_comparison_v2_gemini.png")
    plt.show()


if __name__ == "__main__":
    create_comparison_charts()
    create_gemini_comparison_charts()
    create_training_loss_chart()  # <--- The new function

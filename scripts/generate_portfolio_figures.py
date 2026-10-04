from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "portfolio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND = "#17233c"
EDGE = "#1f2937"
TEXT = "#111827"
WHITE = "#ffffff"
MUTED = "#cbd5e1"


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    facecolor,
    edgecolor=EDGE,
    textcolor=TEXT,
    fontsize=17,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )

    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=textcolor,
        wrap=True,
    )


def add_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color=WHITE,
    linewidth=2.6,
    linestyle="-",
):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            mutation_scale=17,
            shrinkA=0,
            shrinkB=0,
        ),
    )


def add_elbow_arrow(
    ax,
    points,
    color=WHITE,
    linewidth=2.4,
    linestyle="-",
):
    """Draw an orthogonal connector ending with an arrowhead."""

    for (x1, y1), (x2, y2) in zip(
        points[:-2],
        points[1:-1],
    ):
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )

    (x1, y1), (x2, y2) = points[-2], points[-1]

    add_arrow(
        ax,
        x1,
        y1,
        x2,
        y2,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )


def save_figure(fig, filename):
    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.25,
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# IMAGE 1 — PROJECT OVERVIEW
# ============================================================


def generate_project_overview():
    fig, ax = plt.subplots(figsize=(11, 14))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    ax.text(
        0.5,
        0.965,
        "Enterprise Knowledge Intelligence Platform",
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
        color=WHITE,
    )

    ax.text(
        0.5,
        0.925,
        "Project Overview Flow",
        ha="center",
        va="center",
        fontsize=15,
        color=MUTED,
    )

    # ---------------------------------------------------------
    # MAIN PROJECT LAYERS
    # ---------------------------------------------------------

    box_x = 0.20
    box_width = 0.60
    box_height = 0.075

    y_positions = [
        0.82,
        0.685,
        0.55,
        0.415,
        0.28,
        0.145,
    ]

    labels = [
        "1. Enterprise Documents",
        "2. Document Processing",
        "3. Semantic Retrieval",
        "4. Agentic RAG",
        "5. Grounded LLM Response",
        "6. API & Deployment",
    ]

    colors = [
        "#BFDBFE",
        "#BBF7D0",
        "#FEF3C7",
        "#FED7AA",
        "#DDD6FE",
        "#CFFAFE",
    ]

    for y, label, color in zip(
        y_positions,
        labels,
        colors,
    ):
        add_box(
            ax,
            box_x,
            y,
            box_width,
            box_height,
            label,
            facecolor=color,
        )

    # ---------------------------------------------------------
    # FLOW ARROWS
    # ---------------------------------------------------------

    center_x = 0.5

    for i in range(len(y_positions) - 1):
        add_arrow(
            ax,
            center_x,
            y_positions[i] - 0.008,
            center_x,
            y_positions[i + 1] + box_height + 0.008,
            linewidth=3,
        )

    # ---------------------------------------------------------
    # FOOTER
    # ---------------------------------------------------------

    ax.text(
        0.5,
        0.065,
        (
            "RAG  •  FAISS  •  LangGraph  •  OpenAI  •  "
            "FastAPI  •  Docker  •  CI/CD"
        ),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#e2e8f0",
    )

    save_figure(
        fig,
        "01_project_overview.png",
    )


# ============================================================
# IMAGE 2 — ENTERPRISE RAG ARCHITECTURE
# ============================================================


def generate_rag_architecture():
    fig, ax = plt.subplots(figsize=(12, 20))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 36)
    ax.axis("off")

    background = "#17233c"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    # =========================================================
    # HELPERS
    # =========================================================

    def box(
        x,
        y,
        w,
        h,
        text,
        color,
        fontsize=12,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=2,
            edgecolor="#1f2937",
            facecolor=color,
        )

        ax.add_patch(patch)

        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#111827",
            wrap=True,
        )

    def arrow(
        x1,
        y1,
        x2,
        y2,
        color="white",
        linewidth=2.4,
    ):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                linewidth=linewidth,
                mutation_scale=17,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    def line(
        x1,
        y1,
        x2,
        y2,
        color="white",
        linewidth=2.3,
    ):
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
        )

    def elbow_arrow(
        points,
        color="white",
        linewidth=2.3,
    ):
        """
        Draw orthogonal connector segments.
        The final segment contains the arrowhead.
        """

        for i in range(len(points) - 2):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            line(
                x1,
                y1,
                x2,
                y2,
                color=color,
                linewidth=linewidth,
            )

        x1, y1 = points[-2]
        x2, y2 = points[-1]

        arrow(
            x1,
            y1,
            x2,
            y2,
            color=color,
            linewidth=linewidth,
        )

    # =========================================================
    # TITLE
    # =========================================================

    ax.text(
        6,
        35.25,
        "Enterprise RAG Architecture",
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
        color="white",
    )

    ax.text(
        6,
        34.55,
        (
            "Enterprise Knowledge Intelligence Platform\n"
            "Retrieval • Evidence Validation • Agentic AI • Deployment"
        ),
        ha="center",
        va="center",
        fontsize=12.5,
        color="#cbd5e1",
        linespacing=1.5,
    )

    # =========================================================
    # 1. INDEXING PIPELINE
    # =========================================================

    ax.text(
        0.6,
        33.35,
        "1  INDEXING PIPELINE",
        fontsize=13,
        fontweight="bold",
        color="#93c5fd",
    )

    main_x = 3.6
    main_w = 4.8
    main_h = 1.15
    center_x = 6

    index_positions = [
        31.65,
        29.85,
        28.05,
        26.25,
    ]

    index_labels = [
        "Enterprise Documents",
        "Section-Aware Chunking",
        "Sentence Transformer Embeddings",
        "FAISS Vector Index",
    ]

    index_colors = [
        "#BFDBFE",
        "#BBF7D0",
        "#FEF3C7",
        "#DDD6FE",
    ]

    for y, label, color in zip(
        index_positions,
        index_labels,
        index_colors,
    ):
        box(
            main_x,
            y,
            main_w,
            main_h,
            label,
            color,
            fontsize=12.5,
        )

    for i in range(len(index_positions) - 1):
        arrow(
            center_x,
            index_positions[i],
            center_x,
            index_positions[i + 1] + main_h,
        )

    # =========================================================
    # 2. QUERY & RETRIEVAL
    # =========================================================

    ax.text(
        0.6,
        25.15,
        "2  QUERY & RETRIEVAL",
        fontsize=13,
        fontweight="bold",
        color="#6ee7b7",
    )

    retrieval_positions = [
        23.55,
        21.85,
        20.15,
        18.45,
        16.75,
        15.05,
        13.35,
    ]

    retrieval_labels = [
        "User Question",
        "Query Embedding",
        "Dense Retrieval — Top 10",
        "Cross-Encoder Reranking",
        "Top 3 Evidence Chunks",
        "Context Builder",
        "Evidence Quality Gate",
    ]

    retrieval_colors = [
        "#CFFAFE",
        "#BAE6FD",
        "#FEF3C7",
        "#FED7AA",
        "#DDD6FE",
        "#FDE68A",
        "#BBF7D0",
    ]

    for y, label, color in zip(
        retrieval_positions,
        retrieval_labels,
        retrieval_colors,
    ):
        box(
            main_x,
            y,
            main_w,
            main_h,
            label,
            color,
            fontsize=12,
        )

    for i in range(len(retrieval_positions) - 1):
        arrow(
            center_x,
            retrieval_positions[i],
            center_x,
            retrieval_positions[i + 1] + main_h,
        )

    # =========================================================
    # FAISS → DENSE RETRIEVAL
    # Dedicated right-side lane
    # =========================================================

    faiss_center_y = (
        index_positions[-1]
        + main_h / 2
    )

    dense_center_y = (
        retrieval_positions[2]
        + main_h / 2
    )

    right_lane_x = 10.4

    # Leave FAISS from right edge
    line(
        main_x + main_w,
        faiss_center_y,
        right_lane_x,
        faiss_center_y,
        color="#93c5fd",
    )

    # Move vertically in empty right lane
    line(
        right_lane_x,
        faiss_center_y,
        right_lane_x,
        dense_center_y,
        color="#93c5fd",
    )

    # Enter Dense Retrieval from right
    arrow(
        right_lane_x,
        dense_center_y,
        main_x + main_w,
        dense_center_y,
        color="#93c5fd",
    )

    ax.text(
        10.65,
        23.15,
        "Indexed\nKnowledge",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#bfdbfe",
    )

    # =========================================================
    # 3. AGENTIC RAG & GENERATION
    # =========================================================

    ax.text(
        0.6,
        12.25,
        "3  AGENTIC RAG & GENERATION",
        fontsize=13,
        fontweight="bold",
        color="#c4b5fd",
    )

    gate_bottom_y = retrieval_positions[-1]

    # Evidence gate center
    gate_center_x = center_x

    # ---------------------------------------------------------
    # BRANCH ROUTING LANE
    # ---------------------------------------------------------

    branch_y = 11.75

    line(
        gate_center_x,
        gate_bottom_y,
        gate_center_x,
        branch_y,
        color="#e2e8f0",
        linewidth=2.5,
    )

    # =========================================================
    # SUFFICIENT EVIDENCE PATH — LEFT
    # =========================================================

    sufficient_x = 0.9
    sufficient_w = 4.0
    sufficient_center_x = (
        sufficient_x
        + sufficient_w / 2
    )

    ax.text(
        sufficient_center_x,
        11.28,
        "SUFFICIENT EVIDENCE",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#86efac",
    )

    line(
        gate_center_x,
        branch_y,
        sufficient_center_x,
        branch_y,
        color="#86efac",
        linewidth=2.5,
    )

    sufficient_positions = [
        10.05,
        8.35,
        6.65,
        4.95,
    ]

    sufficient_labels = [
        "LangGraph Agentic Workflow",
        "OpenAI Grounded Generation",
        "Citation Validation",
        "Grounded Answer + Sources",
    ]

    sufficient_colors = [
        "#DDD6FE",
        "#FBCFE8",
        "#CFFAFE",
        "#BBF7D0",
    ]

    for y, label, color in zip(
        sufficient_positions,
        sufficient_labels,
        sufficient_colors,
    ):
        box(
            sufficient_x,
            y,
            sufficient_w,
            main_h,
            label,
            color,
            fontsize=11.2,
        )

    # Branch enters LangGraph from above
    arrow(
        sufficient_center_x,
        branch_y,
        sufficient_center_x,
        sufficient_positions[0] + main_h,
        color="#86efac",
        linewidth=2.5,
    )

    # Vertical sufficient path
    for i in range(len(sufficient_positions) - 1):
        arrow(
            sufficient_center_x,
            sufficient_positions[i],
            sufficient_center_x,
            sufficient_positions[i + 1] + main_h,
            color="#86efac",
            linewidth=2.3,
        )

    # =========================================================
    # INSUFFICIENT EVIDENCE PATH — RIGHT
    # =========================================================

    abstain_x = 7.1
    abstain_w = 4.0
    abstain_center_x = (
        abstain_x
        + abstain_w / 2
    )

    ax.text(
        abstain_center_x,
        11.28,
        "INSUFFICIENT EVIDENCE",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#fca5a5",
    )

    line(
        gate_center_x,
        branch_y,
        abstain_center_x,
        branch_y,
        color="#fca5a5",
        linewidth=2.5,
    )

    abstain_y = 10.05

    box(
        abstain_x,
        abstain_y,
        abstain_w,
        main_h,
        "Abstention Response",
        "#FECACA",
        fontsize=11.5,
    )

    arrow(
        abstain_center_x,
        branch_y,
        abstain_center_x,
        abstain_y + main_h,
        color="#fca5a5",
        linewidth=2.5,
    )

    # =========================================================
    # 4. SERVING & ENGINEERING
    # =========================================================

    ax.text(
        0.6,
        4.15,
        "4  SERVING & ENGINEERING",
        fontsize=13,
        fontweight="bold",
        color="#67e8f9",
    )

    fastapi_y = 2.85

    box(
        main_x,
        fastapi_y,
        main_w,
        main_h,
        "FastAPI REST API",
        "#CFFAFE",
        fontsize=12.5,
    )

    fastapi_top_y = (
        fastapi_y
        + main_h
    )

    # =========================================================
    # GROUNDED ANSWER → FASTAPI
    # Separate left routing lane
    # =========================================================

    grounded_bottom_y = sufficient_positions[-1]

    grounded_lane_y = 4.55

    line(
        sufficient_center_x,
        grounded_bottom_y,
        sufficient_center_x,
        grounded_lane_y,
        color="#86efac",
        linewidth=2.3,
    )

    line(
        sufficient_center_x,
        grounded_lane_y,
        center_x,
        grounded_lane_y,
        color="#86efac",
        linewidth=2.3,
    )

    arrow(
        center_x,
        grounded_lane_y,
        center_x,
        fastapi_top_y,
        color="#86efac",
        linewidth=2.3,
    )

    # =========================================================
    # ABSTENTION → FASTAPI
    # Dedicated far-right lane
    # =========================================================

    abstention_lane_x = 11.45

    abstain_bottom_y = abstain_y

    line(
        abstain_center_x,
        abstain_bottom_y,
        abstention_lane_x,
        abstain_bottom_y,
        color="#fca5a5",
        linewidth=2.3,
    )

    line(
        abstention_lane_x,
        abstain_bottom_y,
        abstention_lane_x,
        4.35,
        color="#fca5a5",
        linewidth=2.3,
    )

    line(
        abstention_lane_x,
        4.35,
        center_x,
        4.35,
        color="#fca5a5",
        linewidth=2.3,
    )

    arrow(
        center_x,
        4.35,
        center_x,
        fastapi_top_y,
        color="#fca5a5",
        linewidth=2.3,
    )

    # =========================================================
    # FASTAPI → DOCKER
    # =========================================================

    docker_y = 1.10

    box(
        main_x,
        docker_y,
        main_w,
        main_h,
        "Docker Container",
        "#BFDBFE",
        fontsize=12.5,
    )

    arrow(
        center_x,
        fastapi_y,
        center_x,
        docker_y + main_h,
        color="#93c5fd",
        linewidth=2.5,
    )

    # =========================================================
    # GITHUB ACTIONS — SEPARATE CI / QUALITY LAYER
    # =========================================================

    ci_x = 8.85
    ci_y = 1.10
    ci_w = 2.75
    ci_h = 1.15

    box(
        ci_x,
        ci_y,
        ci_w,
        ci_h,
        "GitHub Actions CI\nRuff + 71 Tests",
        "#E9D5FF",
        fontsize=10.5,
    )

    ax.text(
        ci_x + ci_w / 2,
        0.68,
        "CI / QUALITY LAYER",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#d8b4fe",
    )

    # No arrow from CI into runtime architecture.
    # GitHub Actions validates the codebase rather than
    # participating in inference requests.

    # =========================================================
    # FOOTER
    # =========================================================

    ax.text(
        6,
        0.25,
        (
            "35 Chunks  •  384-D Embeddings  •  "
            "Dense Top-10 + Cross-Encoder  •  "
            "71 Tests  •  Dockerized API"
        ),
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color="#e2e8f0",
    )

    # =========================================================
    # SAVE
    # =========================================================

    output_path = (
        OUTPUT_DIR
        / "02_rag_architecture.png"
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.30,
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print(f"Saved: {output_path}")
# ============================================================
# SCRIPT ENTRY POINT
# ============================================================
def generate_retrieval_evaluation():
    import numpy as np

    fig, ax = plt.subplots(figsize=(15, 9))

    background = "#17233c"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    strategies = [
        "BM25",
        "Hybrid RRF",
        "Dense Retrieval",
        "Dense +\nCross-Encoder",
    ]

    recall_at_1 = [
        0.3889,
        0.7222,
        0.8333,
        0.8889,
    ]

    mrr_at_5 = [
        0.5463,
        0.8167,
        0.8981,
        0.9444,
    ]

    recall_at_5 = [
        0.7778,
        1.0000,
        1.0000,
        1.0000,
    ]

    x = np.arange(len(strategies))

    bar_width = 0.22

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    fig.text(
        0.5,
        0.95,
        "Retrieval Strategy Evaluation",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="white",
    )

    fig.text(
        0.5,
        0.905,
        "Hard-Query Benchmark • Enterprise Knowledge Retrieval",
        ha="center",
        va="center",
        fontsize=15,
        color="#cbd5e1",
    )

    # ---------------------------------------------------------
    # BARS
    # ---------------------------------------------------------

    bars_recall1 = ax.bar(
        x - bar_width,
        recall_at_1,
        width=bar_width,
        label="Recall@1",
        color="#60A5FA",
    )

    bars_mrr = ax.bar(
        x,
        mrr_at_5,
        width=bar_width,
        label="MRR@5",
        color="#A78BFA",
    )

    bars_recall5 = ax.bar(
        x + bar_width,
        recall_at_5,
        width=bar_width,
        label="Recall@5",
        color="#34D399",
    )

    # ---------------------------------------------------------
    # VALUE LABELS
    # ---------------------------------------------------------

    def add_values(bars):
        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.022,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color="white",
            )

    add_values(bars_recall1)
    add_values(bars_mrr)
    add_values(bars_recall5)

    # ---------------------------------------------------------
    # AXIS FORMATTING
    # ---------------------------------------------------------

    ax.set_ylim(0, 1.13)

    ax.set_xticks(x)

    ax.set_xticklabels(
        strategies,
        fontsize=13,
        fontweight="bold",
        color="white",
    )

    ax.set_ylabel(
        "Retrieval Performance",
        fontsize=13,
        fontweight="bold",
        color="white",
    )

    ax.tick_params(
        axis="y",
        colors="#e2e8f0",
        labelsize=11,
    )

    ax.tick_params(
        axis="x",
        colors="white",
    )

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=1,
    )

    ax.set_axisbelow(True)

    # Remove unnecessary frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")

    # ---------------------------------------------------------
    # LEGEND
    # ---------------------------------------------------------

    legend = ax.legend(
        loc="upper left",
        frameon=False,
        ncol=3,
        fontsize=12,
    )

    for text in legend.get_texts():
        text.set_color("white")

    # ---------------------------------------------------------
    # BEST CONFIGURATION CALLOUT
    # ---------------------------------------------------------

    best_x = x[-1]

    ax.annotate(
        "Selected Production Configuration",
        xy=(
            best_x,
            recall_at_5[-1],
        ),
        xytext=(
            best_x - 0.1,
            1.095,
        ),
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#FDE68A",
        arrowprops=dict(
            arrowstyle="->",
            color="#FDE68A",
            linewidth=2,
        ),
    )

    # ---------------------------------------------------------
    # ENGINEERING INSIGHT BOX
    # ---------------------------------------------------------

    insight_text = (
        "Dense retrieval outperformed BM25 and Hybrid RRF on difficult semantic queries.\n"
        "Cross-encoder reranking improved Recall@1 from 0.833 → 0.889 "
        "and MRR@5 from 0.898 → 0.944."
    )

    ax.text(
        0.5,
        -0.17,
        insight_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#e2e8f0",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="#243552",
            edgecolor="#64748b",
            linewidth=1.5,
        ),
    )

    # ---------------------------------------------------------
    # FOOTER
    # ---------------------------------------------------------

    fig.text(
        0.5,
        0.025,
        (
            "Final Retrieval Pipeline: "
            "Dense Top-10 → Cross-Encoder Reranking → Top-3 Evidence"
        ),
        ha="center",
        fontsize=11.5,
        fontweight="bold",
        color="#93c5fd",
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "03_retrieval_evaluation.png"
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.35,
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

def generate_rag_benchmark_dashboard():
    fig, ax = plt.subplots(figsize=(18, 11.5))

    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11.5)
    ax.axis("off")

    background = "#17233c"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    # =========================================================
    # HELPERS
    # =========================================================

    def rounded_box(
        x,
        y,
        w,
        h,
        facecolor,
        edgecolor="#475569",
        linewidth=2,
        radius=0.16,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.03,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        ax.add_patch(patch)
        return patch

    def metric_card(
        x,
        y,
        w,
        h,
        title,
        value,
        subtitle,
        facecolor,
    ):
        rounded_box(
            x,
            y,
            w,
            h,
            facecolor=facecolor,
            edgecolor="#334155",
            linewidth=2,
            radius=0.16,
        )

        ax.text(
            x + w / 2,
            y + h * 0.70,
            value,
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color="#111827",
        )

        ax.text(
            x + w / 2,
            y + h * 0.42,
            title,
            ha="center",
            va="center",
            fontsize=12.2,
            fontweight="bold",
            color="#111827",
            linespacing=1.22,
        )

        ax.text(
            x + w / 2,
            y + h * 0.15,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.2,
            color="#475569",
            linespacing=1.15,
        )

    # =========================================================
    # TITLE
    # =========================================================

    ax.text(
        9,
        10.85,
        "End-to-End RAG Evaluation",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="white",
    )

    ax.text(
        9,
        10.38,
        "Grounded Answering • Citation Validation • Abstention",
        ha="center",
        va="center",
        fontsize=14.5,
        color="#cbd5e1",
    )

    # =========================================================
    # CURATED BENCHMARK STRIP
    # =========================================================

    strip_x = 0.8
    strip_y = 9.0
    strip_w = 16.4
    strip_h = 1.05

    rounded_box(
        strip_x,
        strip_y,
        strip_w,
        strip_h,
        facecolor="#243552",
        edgecolor="#475569",
        linewidth=1.8,
        radius=0.16,
    )

    # Left badge inside the strip
    badge_x = 1.0
    badge_y = 9.22
    badge_w = 2.4
    badge_h = 0.60

    rounded_box(
        badge_x,
        badge_y,
        badge_w,
        badge_h,
        facecolor="#1f2e4a",
        edgecolor="#4b5d78",
        linewidth=1.4,
        radius=0.14,
    )

    ax.text(
        badge_x + badge_w / 2,
        badge_y + badge_h / 2,
        "CURATED\nBENCHMARK",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#93c5fd",
        linespacing=1.1,
    )

    # Top strip items with more spacing
    ax.text(
        4.2,
        9.53,
        "10 Queries",
        ha="center",
        va="center",
        fontsize=17.5,
        fontweight="bold",
        color="white",
    )

    ax.text(
        8.1,
        9.53,
        "5 Supported",
        ha="center",
        va="center",
        fontsize=17.5,
        fontweight="bold",
        color="#86efac",
    )

    ax.text(
        12.0,
        9.53,
        "5 Unsupported",
        ha="center",
        va="center",
        fontsize=17.5,
        fontweight="bold",
        color="#fca5a5",
    )

    ax.text(
        15.4,
        9.53,
        "Enterprise Policies",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color="#ddd6fe",
    )

    # =========================================================
    # METRIC CARDS
    # =========================================================

    card_y = 5.95
    card_w = 3.55
    card_h = 2.15
    gap = 0.45

    x_positions = [
        0.8,
        0.8 + (card_w + gap),
        0.8 + 2 * (card_w + gap),
        0.8 + 3 * (card_w + gap),
    ]

    metric_card(
        x_positions[0],
        card_y,
        card_w,
        card_h,
        "Answer / Abstain\nDecision Accuracy",
        "100%",
        "10 / 10 decisions correct",
        "#BFDBFE",
    )

    metric_card(
        x_positions[1],
        card_y,
        card_w,
        card_h,
        "Citation\nAccuracy",
        "100%",
        "Supported answers cited correctly",
        "#DDD6FE",
    )

    metric_card(
        x_positions[2],
        card_y,
        card_w,
        card_h,
        "Abstention\nAccuracy",
        "100%",
        "Unsupported queries rejected",
        "#BBF7D0",
    )

    metric_card(
        x_positions[3],
        card_y,
        card_w,
        card_h,
        "Expected Keyword\nCoverage",
        "100%",
        "Mean coverage on supported queries",
        "#FEF3C7",
    )

    # =========================================================
    # EVALUATION BEHAVIOR
    # =========================================================

    ax.text(
        0.8,
        4.95,
        "Evaluation Behavior",
        fontsize=13,
        fontweight="bold",
        color="#67e8f9",
    )

    rounded_box(
        0.8,
        2.9,
        7.7,
        1.65,
        facecolor="#1f3d3b",
        edgecolor="#4ade80",
        linewidth=2,
        radius=0.16,
    )

    ax.text(
        1.35,
        4.0,
        "SUPPORTED QUERY",
        fontsize=10.5,
        fontweight="bold",
        color="#86efac",
        ha="left",
        va="center",
    )

    ax.text(
        1.35,
        3.45,
        (
            "Evidence found\n"
            "→ Generate grounded answer\n"
            "→ Validate citations"
        ),
        fontsize=11.2,
        color="white",
        ha="left",
        va="center",
        linespacing=1.5,
    )

    rounded_box(
        9.3,
        2.9,
        7.9,
        1.65,
        facecolor="#432734",
        edgecolor="#ef4444",
        linewidth=2,
        radius=0.16,
    )

    ax.text(
        9.85,
        4.0,
        "UNSUPPORTED QUERY",
        fontsize=10.5,
        fontweight="bold",
        color="#fca5a5",
        ha="left",
        va="center",
    )

    ax.text(
        9.85,
        3.45,
        (
            "Insufficient evidence\n"
            "→ Abstain\n"
            "→ Return no unsupported sources"
        ),
        fontsize=11.2,
        color="white",
        ha="left",
        va="center",
        linespacing=1.5,
    )

    # =========================================================
    # KEY RESULT PANEL
    # =========================================================

    rounded_box(
        1.4,
        1.05,
        15.2,
        1.0,
        facecolor="#243552",
        edgecolor="#64748b",
        linewidth=1.8,
        radius=0.16,
    )

    ax.text(
        9,
        1.55,
        (
            "100% performance on the curated 10-query enterprise benchmark\n"
            "— not a claim of universal accuracy"
        ),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#e2e8f0",
        linespacing=1.35,
    )

    # =========================================================
    # FOOTER
    # =========================================================

    ax.text(
        9,
        0.35,
        (
            "Evidence Gate  •  Grounded Generation  •  Citation Validation  •  "
            "Hallucination-Resistant Abstention"
        ),
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color="#93c5fd",
    )

    # =========================================================
    # SAVE
    # =========================================================

    output_path = OUTPUT_DIR / "04_rag_benchmark.png"

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.45,
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

def generate_api_deployment_workflow():
    fig, ax = plt.subplots(figsize=(18, 11))

    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    background = "#17233c"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    # =========================================================
    # HELPERS
    # =========================================================

    def rounded_box(
        x,
        y,
        w,
        h,
        text="",
        facecolor="#243552",
        edgecolor="#334155",
        linewidth=2,
        fontsize=12,
        textcolor="#111827",
        radius=0.16,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.03,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        ax.add_patch(patch)

        if text:
            ax.text(
                x + w / 2,
                y + h / 2,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=textcolor,
                wrap=True,
                linespacing=1.2,
            )

        return patch

    def arrow(
        x1,
        y1,
        x2,
        y2,
        color="white",
        linewidth=2.4,
    ):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                linewidth=linewidth,
                mutation_scale=16,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    def line(
        x1,
        y1,
        x2,
        y2,
        color="white",
        linewidth=2.2,
    ):
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
        )

    # =========================================================
    # TITLE
    # =========================================================

    ax.text(
        9,
        10.45,
        "API, Deployment & Engineering Quality",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="white",
    )

    ax.text(
        9,
        9.95,
        "FastAPI • Docker • GitHub Actions • Production-Style RAG Serving",
        ha="center",
        va="center",
        fontsize=14.5,
        color="#cbd5e1",
    )

    # =========================================================
    # 1. FASTAPI SERVING LAYER
    # =========================================================

    ax.text(
        0.8,
        9.1,
        "1  FASTAPI SERVING LAYER",
        fontsize=13,
        fontweight="bold",
        color="#67e8f9",
    )

    # Client / User
    rounded_box(
        0.9,
        7.85,
        2.3,
        1.0,
        "Client / User",
        "#BFDBFE",
        fontsize=13,
    )

    # FastAPI application
    rounded_box(
        3.95,
        7.65,
        3.6,
        1.35,
        "FastAPI Application",
        "#CFFAFE",
        fontsize=14,
    )

    # Client -> FastAPI
    arrow(
        3.2,
        8.35,
        3.95,
        8.35,
        color="#93c5fd",
    )

    # Endpoints container
    rounded_box(
        3.7,
        5.55,
        4.1,
        1.55,
        "",
        facecolor="#243552",
        edgecolor="#64748b",
        linewidth=1.8,
    )

    ax.text(
        5.75,
        6.8,
        "API Endpoints",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#e2e8f0",
    )

    # Endpoint pills
    rounded_box(
        4.0,
        6.0,
        1.0,
        0.62,
        "/health",
        "#E2E8F0",
        edgecolor="#94a3b8",
        linewidth=1.6,
        fontsize=10.5,
    )

    rounded_box(
        5.23,
        6.0,
        1.0,
        0.62,
        "/ready",
        "#DCFCE7",
        edgecolor="#86efac",
        linewidth=1.6,
        fontsize=10.5,
    )

    rounded_box(
        6.46,
        6.0,
        1.0,
        0.62,
        "/ask",
        "#FEF3C7",
        edgecolor="#facc15",
        linewidth=1.6,
        fontsize=10.5,
    )

    ax.text(
        4.5,
        5.72,
        "liveness",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#cbd5e1",
    )

    ax.text(
        5.73,
        5.72,
        "readiness",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#cbd5e1",
    )

    ax.text(
        6.96,
        5.72,
        "inference",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#cbd5e1",
    )

    # FastAPI -> endpoints
    arrow(
        5.75,
        7.65,
        5.75,
        7.1,
        color="#e2e8f0",
        linewidth=2.0,
    )

    # =========================================================
    # 2. RAG APPLICATION FLOW
    # =========================================================

    ax.text(
        8.8,
        9.1,
        "2  RAG APPLICATION FLOW",
        fontsize=13,
        fontweight="bold",
        color="#a78bfa",
    )

    # Main flow boxes
    rag_y = 7.7
    rag_h = 1.1
    rag_w = 2.1

    rag_x_positions = [8.8, 11.5, 14.2]
    rag_labels = [
        "RAG Service",
        "LangGraph\nRAG Agent",
        "Grounded JSON\nResponse",
    ]
    rag_colors = [
        "#DDD6FE",
        "#FBCFE8",
        "#BBF7D0",
    ]

    for x, label, color in zip(rag_x_positions, rag_labels, rag_colors):
        rounded_box(
            x,
            rag_y,
            rag_w,
            rag_h,
            label,
            color,
            fontsize=12,
        )

    # /ask -> RAG Service (important explicit connection)
    ask_center_x = 6.46 + 0.5
    ask_center_y = 6.0 + 0.31
    rag_service_left_x = rag_x_positions[0]
    rag_service_center_y = rag_y + rag_h / 2

    arrow(
        7.46,
        ask_center_y,
        rag_service_left_x,
        rag_service_center_y,
        color="#facc15",
        linewidth=2.4,
    )

    # RAG main arrows
    arrow(
        rag_x_positions[0] + rag_w,
        rag_y + rag_h / 2,
        rag_x_positions[1],
        rag_y + rag_h / 2,
        color="#c4b5fd",
    )

    arrow(
        rag_x_positions[1] + rag_w,
        rag_y + rag_h / 2,
        rag_x_positions[2],
        rag_y + rag_h / 2,
        color="#86efac",
    )

    # Internal transparent/dark pipeline panel
    rounded_box(
        8.9,
        5.45,
        7.5,
        1.2,
        "",
        facecolor="#243552",
        edgecolor="#64748b",
        linewidth=1.8,
    )

    ax.text(
        12.65,
        6.25,
        "Internal Retrieval Pipeline",
        ha="center",
        va="center",
        fontsize=11.3,
        fontweight="bold",
        color="#e2e8f0",
    )

    ax.text(
        12.65,
        5.8,
        (
            "Dense Retrieval  →  Reranking  →  Evidence Gate  →\n"
            "Grounded Generation  →  Citations"
        ),
        ha="center",
        va="center",
        fontsize=10.6,
        color="#cbd5e1",
        linespacing=1.35,
    )

    # Connect LangGraph down to the transparent panel
    langgraph_center_x = rag_x_positions[1] + rag_w / 2

    arrow(
        langgraph_center_x,
        rag_y,
        langgraph_center_x,
        6.65,
        color="#d8b4fe",
        linewidth=2.2,
    )

    # Connect panel up to final response
    grounded_center_x = rag_x_positions[2] + rag_w / 2

    arrow(
        grounded_center_x,
        6.65,
        grounded_center_x,
        rag_y,
        color="#86efac",
        linewidth=2.2,
    )

    # =========================================================
    # 3. DOCKERIZED DEPLOYMENT
    # =========================================================

    ax.text(
        0.8,
        4.95,
        "3  DOCKERIZED DEPLOYMENT",
        fontsize=13,
        fontweight="bold",
        color="#93c5fd",
    )

    container_x = 1.0
    container_y = 2.35
    container_w = 8.1
    container_h = 2.1

    rounded_box(
        container_x,
        container_y,
        container_w,
        container_h,
        "",
        "#243552",
        edgecolor="#60A5FA",
        linewidth=2.2,
    )

    ax.text(
        container_x + container_w / 2,
        container_y + container_h - 0.32,
        "Docker Container",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#bfdbfe",
    )

    rounded_box(
        1.35,
        2.72,
        2.1,
        1.0,
        "FastAPI App",
        "#CFFAFE",
        fontsize=12,
    )

    rounded_box(
        4.0,
        2.72,
        2.1,
        1.0,
        "Vector Store\n+ Metadata",
        "#DDD6FE",
        fontsize=12,
    )

    rounded_box(
        6.65,
        2.72,
        2.1,
        1.0,
        "Models /\nConfig",
        "#FEF3C7",
        fontsize=12,
    )

    # FastAPI application down to container
    arrow(
        5.75,
        5.55,
        5.75,
        4.45,
        color="#93c5fd",
    )

    # =========================================================
    # 4. CI / ENGINEERING QUALITY
    # =========================================================

    ax.text(
        9.2,
        5.15,
        "4  CI / ENGINEERING QUALITY",
        fontsize=13,
        fontweight="bold",
        color="#86efac",
    )

    rounded_box(
        10.2,
        3.95,
        6.8,
        1.15,
        "GitHub Actions CI Workflow",
        "#E9D5FF",
        fontsize=15,
    )

    rounded_box(
        10.35,
        2.3,
        2.0,
        1.0,
        "Ruff",
        "#FDE68A",
        fontsize=13,
    )

    rounded_box(
        12.6,
        2.3,
        2.0,
        1.0,
        "Pytest",
        "#BBF7D0",
        fontsize=13,
    )

    rounded_box(
        14.85,
        2.3,
        2.0,
        1.0,
        "71 Passed",
        "#BFDBFE",
        fontsize=13,
    )

    arrow(
        13.6,
        3.95,
        11.35,
        3.3,
        color="#d8b4fe",
        linewidth=2.1,
    )

    arrow(
        13.6,
        3.95,
        13.6,
        3.3,
        color="#d8b4fe",
        linewidth=2.1,
    )

    arrow(
        13.6,
        3.95,
        15.85,
        3.3,
        color="#d8b4fe",
        linewidth=2.1,
    )

    # =========================================================
    # HIGHLIGHTS PANEL
    # =========================================================

    rounded_box(
        1.0,
        0.75,
        16.0,
        0.95,
        (
            "Production Features: Health & Readiness Checks  •  "
            "Grounded JSON Answers  •  Dockerized API  •  Automated CI Validation"
        ),
        "#243552",
        edgecolor="#64748b",
        fontsize=11.1,
        textcolor="#e2e8f0",
    )

    # =========================================================
    # FOOTER
    # =========================================================

    ax.text(
        9,
        0.22,
        (
            "FastAPI  •  Docker  •  GitHub Actions  •  Ruff  •  "
            "Pytest  •  71 Tests Passed"
        ),
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color="#93c5fd",
    )

    # =========================================================
    # SAVE
    # =========================================================

    output_path = OUTPUT_DIR / "05_api_deployment_workflow.png"

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.35,
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_project_overview()
    generate_rag_architecture()
    generate_retrieval_evaluation()
    generate_rag_benchmark_dashboard()
    generate_api_deployment_workflow()
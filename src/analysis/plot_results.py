from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXCEL_FILE = (
    PROJECT_ROOT
    / "results"
    / "candidate_ranking.xlsx"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURE CLUSTERS
# ============================================================

CLUSTER_MAP = {

    # --------------------------------------------------------
    # 1. TEXTUAL AND SEMANTIC
    # --------------------------------------------------------

    "Text Levenshtein":
        "Textual and Semantic",

    "Text Jaccard":
        "Textual and Semantic",

    "Text Semantic Cosine":
        "Textual and Semantic",

    "Content-desc Semantic":
        "Textual and Semantic",

    "Content-desc Semantic Cosine":
        "Textual and Semantic",

    "Neighbor Text Jaccard":
        "Textual and Semantic",


    # --------------------------------------------------------
    # 2. DOM ATTRIBUTE AND ELEMENT IDENTITY
    # --------------------------------------------------------

    "Resource ID Exact":
        "DOM Attribute and Element Identity",

    "Resource ID Exact Match":
        "DOM Attribute and Element Identity",

    "Class Exact":
        "DOM Attribute and Element Identity",

    "Class Exact Match":
        "DOM Attribute and Element Identity",

    "DOM Attributes":
        "DOM Attribute and Element Identity",


    # --------------------------------------------------------
    # 3. STRUCTURAL FEATURES
    # --------------------------------------------------------

    "Tree Edit Distance":
        "Structural Features",

    "XPath Levenshtein":
        "Structural Features",


    # --------------------------------------------------------
    # 4. SPATIAL AND GEOMETRIC
    # --------------------------------------------------------

    "Position Euclidean":
        "Spatial and Geometric",

    "Size Euclidean":
        "Spatial and Geometric",

    "Bounding Box IoU":
        "Spatial and Geometric",
}


CLUSTER_ORDER = [
    "Textual and Semantic",
    "DOM Attribute and Element Identity",
    "Structural Features",
    "Spatial and Geometric",
]


CLUSTER_LABELS = [
    "Textual and\nSemantic",
    "DOM Attribute and\nElement Identity",
    "Structural\nFeatures",
    "Spatial and\nGeometric",
]


# ============================================================
# HELPERS
# ============================================================

def to_percentage(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = values.dropna()

    if (
        not valid.empty
        and valid.max() <= 1.5
    ):
        values = values * 100

    return values


def save_figure(filename):

    output_path = (
        FIGURE_DIR / filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_candidate_metric(
    dataframe,
    metric_order,
    column,
    ylabel,
    title,
    filename,
):

    plot_df = (
        dataframe
        .set_index("Metric")
        .reindex(metric_order)
        .reset_index()
    )

    values = (
        plot_df[column]
        .tolist()
    )

    x = np.arange(
        len(metric_order)
    )

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    bars = ax.bar(
        x,
        values
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        metric_order,
        rotation=55,
        ha="right",
        fontsize=9
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlabel(
        "Metric"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        0,
        105
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        values
    ):

        if pd.notna(value):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8
            )

    save_figure(
        filename
    )


# ============================================================
# CHECK EXCEL
# ============================================================

if not EXCEL_FILE.exists():

    raise FileNotFoundError(
        f"Excel file not found: "
        f"{EXCEL_FILE}"
    )


# ============================================================
# LOAD EXCEL SHEETS
# ============================================================

summary_df = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Summary"
)


classification_df = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Classification Summary"
)


classification_na_df = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Classification Summary NA"
)


per_app_df = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Per App Summary"
)


# ============================================================
# CONVERT RANKING VALUES TO PERCENTAGES
# ============================================================

for column in [
    "Coverage",
    "Strict Accuracy",
    "Top-set Success",
]:

    summary_df[column] = (
        to_percentage(
            summary_df[column]
        )
    )


# ============================================================
# ADD CLUSTER INFORMATION
# ============================================================

summary_df["Cluster"] = (
    summary_df["Metric"]
    .map(CLUSTER_MAP)
)


unmapped_metrics = (
    summary_df.loc[
        summary_df["Cluster"].isna(),
        "Metric"
    ]
    .tolist()
)


if unmapped_metrics:

    print(
        "\nWARNING - metrics without cluster:"
    )

    for metric in unmapped_metrics:
        print(
            f" - {metric}"
        )


cluster_df = (
    summary_df
    .dropna(
        subset=["Cluster"]
    )
    .copy()
)


# ============================================================
# 1. COVERAGE OF ALL 13 METRICS
# ============================================================

metric_names = (
    summary_df["Metric"]
    .tolist()
)


coverage_values = (
    summary_df["Coverage"]
    .tolist()
)


x = np.arange(
    len(metric_names)
)


fig, ax = plt.subplots(
    figsize=(15, 8)
)


bars = ax.bar(
    x,
    coverage_values
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    metric_names,
    rotation=55,
    ha="right",
    fontsize=9
)


ax.set_ylabel(
    "Coverage (%)"
)


ax.set_xlabel(
    "Metric"
)


ax.set_title(
    "Coverage of Evaluated Metrics"
)


ax.set_ylim(
    0,
    108
)


ax.grid(
    axis="y",
    alpha=0.25
)


for bar, value in zip(
    bars,
    coverage_values
):

    if pd.notna(value):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )


save_figure(
    "overall_ranking_performance.png"
)


# ============================================================
# 2. APPLICABLE CASES BY FEATURE CLUSTER
# ============================================================

cluster_applicable = (
    cluster_df
    .groupby("Cluster")[
        "Applicable"
    ]
    .mean()
    .reindex(
        CLUSTER_ORDER
    )
)


fig, ax = plt.subplots(
    figsize=(11, 7)
)


bars = ax.bar(
    CLUSTER_LABELS,
    cluster_applicable.values
)


ax.set_ylabel(
    "Mean Applicable Cases"
)


ax.set_xlabel(
    "Feature Cluster"
)


ax.set_title(
    "Applicable Cases by Feature Cluster"
)


ax.set_ylim(
    0,
    215
)


ax.grid(
    axis="y",
    alpha=0.25
)


for bar, value in zip(
    bars,
    cluster_applicable.values
):

    if pd.notna(value):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 3,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9
        )


save_figure(
    "applicable_by_feature_cluster.png"
)


# ============================================================
# 3. TOP-SET SUCCESS BY FEATURE CLUSTER
# ============================================================

cluster_topset = (
    cluster_df
    .groupby("Cluster")[
        "Top-set Success"
    ]
    .mean()
    .reindex(
        CLUSTER_ORDER
    )
)


fig, ax = plt.subplots(
    figsize=(11, 7)
)


bars = ax.bar(
    CLUSTER_LABELS,
    cluster_topset.values
)


ax.set_ylabel(
    "Mean Top-set Success (%)"
)


ax.set_xlabel(
    "Feature Cluster"
)


ax.set_title(
    "Top-set Success by Feature Cluster"
)


ax.set_ylim(
    0,
    105
)


ax.grid(
    axis="y",
    alpha=0.25
)


for bar, value in zip(
    bars,
    cluster_topset.values
):

    if pd.notna(value):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )


save_figure(
    "topset_by_feature_cluster.png"
)


# ============================================================
# ORIGINAL CANDIDATE-LEVEL DATA
# ============================================================

for column in [
    "Precision",
    "Recall",
    "F1-score",
]:

    classification_df[column] = (
        to_percentage(
            classification_df[column]
        )
    )


classification_df = (
    classification_df
    .sort_values(
        "F1-score",
        ascending=False,
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)


candidate_metric_order = (
    classification_df[
        "Metric"
    ]
    .tolist()
)


# ============================================================
# 4. CANDIDATE-LEVEL PRECISION
# ============================================================

plot_candidate_metric(
    classification_df,
    candidate_metric_order,
    "Precision",
    "Precision (%)",
    "Candidate-Level Precision",
    "candidate_level_precision.png",
)


# ============================================================
# 5. CANDIDATE-LEVEL RECALL
# ============================================================

plot_candidate_metric(
    classification_df,
    candidate_metric_order,
    "Recall",
    "Recall (%)",
    "Candidate-Level Recall",
    "candidate_level_recall.png",
)


# ============================================================
# 6. CANDIDATE-LEVEL F1-SCORE
# ============================================================

plot_candidate_metric(
    classification_df,
    candidate_metric_order,
    "F1-score",
    "F1-score (%)",
    "Candidate-Level F1-score",
    "candidate_level_f1.png",
)


# ============================================================
# CANDIDATE-LEVEL DATA INCLUDING NOT APPLICABLE
# ============================================================

for column in [
    "Precision",
    "Recall",
    "F1-score",
]:

    classification_na_df[column] = (
        to_percentage(
            classification_na_df[column]
        )
    )


classification_na_df = (
    classification_na_df
    .sort_values(
        "F1-score",
        ascending=False,
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)


candidate_na_metric_order = candidate_metric_order


# ============================================================
# 7. PRECISION INCLUDING NOT APPLICABLE
# ============================================================

plot_candidate_metric(
    classification_na_df,
    candidate_na_metric_order,
    "Precision",
    "Precision (%)",
    (
        "Candidate-Level Precision "
        "Including Not Applicable Cases"
    ),
    "candidate_level_precision_with_na.png",
)


# ============================================================
# 8. RECALL INCLUDING NOT APPLICABLE
# ============================================================

plot_candidate_metric(
    classification_na_df,
    candidate_na_metric_order,
    "Recall",
    "Recall (%)",
    (
        "Candidate-Level Recall "
        "Including Not Applicable Cases"
    ),
    "candidate_level_recall_with_na.png",
)


# ============================================================
# 9. F1-SCORE INCLUDING NOT APPLICABLE
# ============================================================

plot_candidate_metric(
    classification_na_df,
    candidate_na_metric_order,
    "F1-score",
    "F1-score (%)",
    (
        "Candidate-Level F1-score "
        "Including Not Applicable Cases"
    ),
    "candidate_level_f1_with_na.png",
)

# ============================================================
# COMPARISON: APPLICABLE F1 VS NA-AWARE F1 (VERTICAL)
# ============================================================

f1_comparison = pd.merge(
    classification_df[["Metric", "F1-score"]],
    classification_na_df[["Metric", "F1-score"]],
    on="Metric",
    how="inner",
    suffixes=("_applicable", "_na_aware")
)

# Keep the same order as the other figures
f1_comparison = (
    f1_comparison
    .set_index("Metric")
    .reindex(candidate_metric_order)
    .reset_index()
)

x = np.arange(len(f1_comparison))
bar_width = 0.36

fig, ax = plt.subplots(figsize=(16, 8))

bars_applicable = ax.bar(
    x - bar_width / 2,
    f1_comparison["F1-score_applicable"],
    bar_width,
    label="Applicable-case F1"
)

bars_na = ax.bar(
    x + bar_width / 2,
    f1_comparison["F1-score_na_aware"],
    bar_width,
    label="NA-aware F1"
)

ax.set_xticks(x)
ax.set_xticklabels(
    f1_comparison["Metric"],
    rotation=55,
    ha="right",
    fontsize=9
)

ax.set_ylabel("F1-score (%)")
ax.set_xlabel("Metric")

ax.set_title(
    "Effect of Not Applicable Cases on F1-score"
)

ax.set_ylim(0, 105)
ax.legend()

ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

# Display percentages above the bars
for bars in [bars_applicable, bars_na]:
    for bar in bars:
        value = bar.get_height()

        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90
            )

save_figure(
    "candidate_level_f1_comparison_na.png"
)

# ============================================================
# 10. COVERAGE VS STRICT ACCURACY
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 8)
)


ax.scatter(
    summary_df["Coverage"],
    summary_df["Strict Accuracy"],
    s=40
)


label_offsets = {

    "Text Semantic Cosine":
        (6, 6),

    "Text Levenshtein":
        (6, 4),

    "Text Jaccard":
        (6, -8),

    "Content-desc Semantic":
        (6, 5),

    "Content-desc Semantic Cosine":
        (6, 5),

    "Neighbor Text Jaccard":
        (6, 5),

    "Resource ID Exact":
        (6, 5),

    "Resource ID Exact Match":
        (6, 5),

    "Bounding Box IoU":
        (-85, 8),

    "Position Euclidean":
        (-85, 0),

    "XPath Levenshtein":
        (-85, -8),

    "Size Euclidean":
        (-85, -16),

    "Tree Edit Distance":
        (-95, 8),

    "Class Exact":
        (-95, 0),

    "Class Exact Match":
        (-95, 0),

    "DOM Attributes":
        (-95, -8),
}


for _, row in (
    summary_df.iterrows()
):

    coverage = (
        row["Coverage"]
    )

    accuracy = (
        row["Strict Accuracy"]
    )

    metric = (
        row["Metric"]
    )

    if (
        pd.isna(coverage)
        or pd.isna(accuracy)
    ):
        continue

    dx, dy = (
        label_offsets.get(
            metric,
            (6, 5)
        )
    )

    ax.annotate(
        metric,
        (
            coverage,
            accuracy
        ),
        xytext=(
            dx,
            dy
        ),
        textcoords="offset points",
        fontsize=8,
        annotation_clip=False
    )


ax.set_xlabel(
    "Coverage (%)"
)


ax.set_ylabel(
    "Strict Accuracy (%)"
)


ax.set_title(
    "Coverage vs Strict Accuracy"
)


ax.set_xlim(
    -5,
    105
)


ax.set_ylim(
    -5,
    105
)


ax.grid(
    alpha=0.25
)


save_figure(
    "coverage_vs_strict_accuracy.png"
)


# ============================================================
# 11. PER-APPLICATION F1 DISTRIBUTION
# ============================================================

per_app_df[
    "F1-score"
] = to_percentage(
    per_app_df[
        "F1-score"
    ]
)


valid_f1 = (
    per_app_df
    .dropna(
        subset=["F1-score"]
    )
    .copy()
)


# ============================================================
# PER-APPLICATION MEAN AND MEDIAN F1 - EXCEL OUTPUT
# ============================================================

per_app_stats = (
    valid_f1
    .groupby("Metric")["F1-score"]
    .agg(
        Mean_F1="mean",
        Median_F1="median",
        App_Count="count"
    )
    .reset_index()
)

# Sort by Median F1 (highest first)
per_app_stats = per_app_stats.sort_values(
    by="Median_F1",
    ascending=False
)

# Round values to two decimal places
per_app_stats["Mean_F1"] = (
    per_app_stats["Mean_F1"].round(2)
)

per_app_stats["Median_F1"] = (
    per_app_stats["Median_F1"].round(2)
)

# Save to Excel
excel_output = (
    PROJECT_ROOT
    / "results"
    / "per_application_mean_median.xlsx"
)

per_app_stats.to_excel(
    excel_output,
    index=False
)

print("\nPER-APPLICATION F1 STATISTICS:")
print(per_app_stats.to_string(index=False))

print(f"\nExcel saved to: {excel_output}")


metric_order = (
    valid_f1
    .groupby("Metric")[
        "F1-score"
    ]
    .median()
    .sort_values(
        ascending=False
    )
    .index
    .tolist()
)


boxplot_data = []


for metric in metric_order:

    values = (
        valid_f1.loc[
            valid_f1[
                "Metric"
            ] == metric,
            "F1-score"
        ]
        .dropna()
        .tolist()
    )

    boxplot_data.append(
        values
    )


fig_height = max(
    8,
    len(metric_order)
    * 0.55
)


fig, ax = plt.subplots(
    figsize=(
        13,
        fig_height
    )
)


ax.boxplot(
    boxplot_data,
    vert=False,
    showmeans=True,
    widths=0.55
)


positions = list(
    range(
        1,
        len(metric_order) + 1
    )
)


ax.set_yticks(
    positions
)


ax.set_yticklabels(
    metric_order,
    fontsize=9
)


ax.invert_yaxis()


ax.set_xlabel(
    "Per-application F1-score (%)"
)


ax.set_ylabel(
    "Metric"
)


ax.set_title(
    "Distribution of F1-score Across Applications"
)


ax.set_xlim(
    -5,
    105
)


ax.grid(
    axis="x",
    alpha=0.25
)


fig.text(
    0.5,
    0.01,
    (
        "Box = middle 50% of application F1-scores; "
        "whiskers = non-outlier range; "
        "triangle = mean; "
        "line inside box = median."
    ),
    ha="center",
    fontsize=8
)


save_figure(
    "per_application_f1_distribution.png"
)


# ============================================================
# DONE
# ============================================================

print(
    "\n=========================================="
)

print(
    "ALL FIGURES GENERATED SUCCESSFULLY"
)

print(
    "=========================================="
)

print(
    f"\nFigures saved in:\n{FIGURE_DIR}"
)
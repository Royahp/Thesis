from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXCEL_FILE = PROJECT_ROOT / "results" / "candidate_ranking.xlsx"

FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_GROUND_TRUTH = 201


def normalize_column_name(name):
    return " ".join(str(name).strip().lower().split())


def find_sheet_by_columns(all_sheets, required_columns, excluded_columns=None):

    required = {
        normalize_column_name(column)
        for column in required_columns
    }

    excluded = {
        normalize_column_name(column)
        for column in (excluded_columns or [])
    }

    for sheet_name, df in all_sheets.items():

        available = {
            normalize_column_name(column)
            for column in df.columns
        }

        if required.issubset(available) and not excluded.intersection(available):
            print(f"Using sheet: {sheet_name}")
            return df.copy()

    raise ValueError(
        f"Could not find a sheet containing: {required_columns}"
    )


def rename_matching_columns(df, expected_columns):

    normalized_to_original = {
        normalize_column_name(column): column
        for column in df.columns
    }

    rename_map = {}

    for expected in expected_columns:

        key = normalize_column_name(expected)

        if key in normalized_to_original:
            rename_map[normalized_to_original[key]] = expected

    return df.rename(columns=rename_map)


def convert_to_percentage(df, columns):

    for column in columns:

        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        values = df[column].dropna()

        if not values.empty and values.max() <= 1.5:
            df[column] = df[column] * 100

    return df


def save_figure(filename):

    output_path = FIGURE_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.close()


if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"Excel file not found: {EXCEL_FILE}"
    )


all_sheets = pd.read_excel(
    EXCEL_FILE,
    sheet_name=None
)


print("\nAvailable sheets:")

for sheet_name in all_sheets:
    print(f" - {sheet_name}")


# ============================================================
# RANKING DATA
# ============================================================

ranking_required = [
    "Metric",
    "Coverage",
    "Strict Accuracy",
    "Top-set Success"
]


ranking_df = find_sheet_by_columns(
    all_sheets,
    ranking_required
)


ranking_df = rename_matching_columns(
    ranking_df,
    [
        "Metric",
        "Coverage",
        "Strict Accuracy",
        "Top-set Success",
        "Applicable"
    ]
)


ranking_df = ranking_df.dropna(
    subset=["Metric"]
).copy()


ranking_df = convert_to_percentage(
    ranking_df,
    [
        "Coverage",
        "Strict Accuracy",
        "Top-set Success"
    ]
)


if "Applicable" in ranking_df.columns:

    ranking_df["Applicable"] = pd.to_numeric(
        ranking_df["Applicable"],
        errors="coerce"
    )

else:

    ranking_df["Applicable"] = (
        ranking_df["Coverage"]
        * TOTAL_GROUND_TRUTH
        / 100
    ).round()


# ============================================================
# 1. COVERAGE OF EVALUATED METRICS
# ============================================================

metric_names = ranking_df["Metric"].tolist()
coverage_values = ranking_df["Coverage"].tolist()

x = np.arange(len(metric_names))

fig, ax = plt.subplots(figsize=(15, 8))

bars = ax.bar(
    x,
    coverage_values
)

ax.set_xticks(x)

ax.set_xticklabels(
    metric_names,
    rotation=55,
    ha="right",
    fontsize=9
)

ax.set_ylabel("Coverage (%)")
ax.set_xlabel("Metric")
ax.set_title("Coverage of Evaluated Metrics")

ax.set_ylim(0, 108)
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, coverage_values):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
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
# CLUSTERS
# ============================================================

cluster_map = {

    "Text Levenshtein":
        "Textual and Semantic",

    "Text Jaccard":
        "Textual and Semantic",

    "Text Semantic Cosine":
        "Textual and Semantic",


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

    "Content-desc Semantic":
        "DOM Attribute and Element Identity",

    "Content-desc Semantic Cosine":
        "DOM Attribute and Element Identity",


    "Tree Edit Distance":
        "Structural Features",

    "XPath Levenshtein":
        "Structural Features",

    "Neighbor Text Jaccard":
        "Structural Features",


    "Position Euclidean":
        "Spatial and Geometric",

    "Size Euclidean":
        "Spatial and Geometric",

    "Bounding Box IoU":
        "Spatial and Geometric"
}


cluster_order = [
    "Textual and Semantic",
    "DOM Attribute and Element Identity",
    "Structural Features",
    "Spatial and Geometric"
]


cluster_labels = [
    "Textual and\nSemantic",
    "DOM Attribute and\nElement Identity",
    "Structural\nFeatures",
    "Spatial and\nGeometric"
]


ranking_df["Cluster"] = ranking_df[
    "Metric"
].map(cluster_map)


cluster_df = ranking_df.dropna(
    subset=["Cluster"]
).copy()


# ============================================================
# 2. APPLICABLE CASES BY CLUSTER
# ============================================================

cluster_applicable = (
    cluster_df
    .groupby("Cluster")["Applicable"]
    .mean()
    .reindex(cluster_order)
)


fig, ax = plt.subplots(
    figsize=(11, 7)
)


bars = ax.bar(
    cluster_labels,
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

    ax.text(
        bar.get_x() + bar.get_width() / 2,
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
# 3. TOP-SET SUCCESS BY CLUSTER
# ============================================================

cluster_topset = (
    cluster_df
    .groupby("Cluster")["Top-set Success"]
    .mean()
    .reindex(cluster_order)
)


fig, ax = plt.subplots(
    figsize=(11, 7)
)


bars = ax.bar(
    cluster_labels,
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

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.5,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )


save_figure(
    "topset_by_feature_cluster.png"
)


# ============================================================
# CANDIDATE-LEVEL DATA
# ============================================================

classification_columns = [
    "Metric",
    "TP",
    "FP",
    "FN",
    "TN",
    "Precision",
    "Recall",
    "F1-score"
]


classification_df = find_sheet_by_columns(
    all_sheets,
    classification_columns,
    excluded_columns=["App"]
)


classification_df = rename_matching_columns(
    classification_df,
    classification_columns
)


classification_df = classification_df[
    classification_columns
].copy()


classification_df = classification_df.dropna(
    subset=["Metric"]
)


classification_df = convert_to_percentage(
    classification_df,
    [
        "Precision",
        "Recall",
        "F1-score"
    ]
)


classification_df = classification_df.sort_values(
    "F1-score",
    ascending=False,
    na_position="last"
).reset_index(drop=True)


candidate_metrics = classification_df[
    "Metric"
].tolist()


x = np.arange(
    len(candidate_metrics)
)


# ============================================================
# 4. CANDIDATE-LEVEL PRECISION
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 8)
)


bars = ax.bar(
    x,
    classification_df["Precision"]
)


ax.set_xticks(x)

ax.set_xticklabels(
    candidate_metrics,
    rotation=55,
    ha="right",
    fontsize=9
)


ax.set_ylabel(
    "Precision (%)"
)

ax.set_xlabel(
    "Metric"
)

ax.set_title(
    "Candidate-Level Precision"
)


ax.set_ylim(
    0,
    105
)

ax.grid(
    axis="y",
    alpha=0.25
)


save_figure(
    "candidate_level_precision.png"
)


# ============================================================
# 5. CANDIDATE-LEVEL RECALL
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 8)
)


bars = ax.bar(
    x,
    classification_df["Recall"]
)


ax.set_xticks(x)

ax.set_xticklabels(
    candidate_metrics,
    rotation=55,
    ha="right",
    fontsize=9
)


ax.set_ylabel(
    "Recall (%)"
)

ax.set_xlabel(
    "Metric"
)

ax.set_title(
    "Candidate-Level Recall"
)


ax.set_ylim(
    0,
    105
)

ax.grid(
    axis="y",
    alpha=0.25
)


save_figure(
    "candidate_level_recall.png"
)


# ============================================================
# 6. CANDIDATE-LEVEL F1
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 8)
)


bars = ax.bar(
    x,
    classification_df["F1-score"]
)


ax.set_xticks(x)

ax.set_xticklabels(
    candidate_metrics,
    rotation=55,
    ha="right",
    fontsize=9
)


ax.set_ylabel(
    "F1-score (%)"
)

ax.set_xlabel(
    "Metric"
)

ax.set_title(
    "Candidate-Level F1-score"
)


ax.set_ylim(
    0,
    105
)

ax.grid(
    axis="y",
    alpha=0.25
)


save_figure(
    "candidate_level_f1.png"
)


# ============================================================
# 7. COVERAGE VS STRICT ACCURACY
# ============================================================

coverage_plot = pd.to_numeric(
    ranking_df["Coverage"],
    errors="coerce"
).copy()


accuracy_plot = pd.to_numeric(
    ranking_df["Strict Accuracy"],
    errors="coerce"
).copy()


fig, ax = plt.subplots(
    figsize=(12, 8)
)


ax.scatter(
    coverage_plot,
    accuracy_plot,
    s=40
)


label_offsets = {

    "Text Semantic Cosine": (6, 6),
    "Text Levenshtein": (6, 4),
    "Text Jaccard": (6, -8),

    "Content-desc Semantic": (6, 5),
    "Content-desc Semantic Cosine": (6, 5),

    "Neighbor Text Jaccard": (6, 5),

    "Resource ID Exact": (6, 5),
    "Resource ID Exact Match": (6, 5),

    "Bounding Box IoU": (-85, 8),
    "Position Euclidean": (-85, 0),
    "XPath Levenshtein": (-85, -8),
    "Size Euclidean": (-85, -16),

    "Tree Edit Distance": (-95, 8),
    "Class Exact": (-95, 0),
    "Class Exact Match": (-95, 0),
    "DOM Attributes": (-95, -8)
}


ranking_reset = ranking_df.reset_index(
    drop=True
)


for i, row in ranking_reset.iterrows():

    coverage = coverage_plot.iloc[i]
    accuracy = accuracy_plot.iloc[i]
    metric = row["Metric"]

    if pd.notna(coverage) and pd.notna(accuracy):

        dx, dy = label_offsets.get(
            metric,
            (6, 5)
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
# 8. PER-APPLICATION F1 DISTRIBUTION
# ============================================================

per_app_columns = [
    "App",
    "Metric",
    "Precision",
    "Recall",
    "F1-score"
]


per_app_df = find_sheet_by_columns(
    all_sheets,
    per_app_columns
)


per_app_df = rename_matching_columns(
    per_app_df,
    per_app_columns
)


per_app_df = per_app_df[
    per_app_columns
].copy()


per_app_df = convert_to_percentage(
    per_app_df,
    [
        "Precision",
        "Recall",
        "F1-score"
    ]
)


valid_f1 = per_app_df.dropna(
    subset=["F1-score"]
).copy()


metric_order = (
    valid_f1
    .groupby("Metric")["F1-score"]
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
            valid_f1["Metric"] == metric,
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
    len(metric_order) * 0.55
)


fig, ax = plt.subplots(
    figsize=(13, fig_height)
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


print("\n==========================================")
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("==========================================")

print(f"\nFigures saved in:\n{FIGURE_DIR}")
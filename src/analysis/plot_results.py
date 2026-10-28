from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

# Current file:
# Thesis/src/analysis/plot_results.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXCEL_FILE = PROJECT_ROOT / "results" / "candidate_ranking.xlsx"

FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column_name(name):
    """
    Normalize a column name so small differences in spaces
    and capitalization do not affect sheet detection.
    """
    return " ".join(str(name).strip().lower().split())


def find_sheet_by_columns(all_sheets, required_columns):
    """
    Find the first Excel sheet containing all required columns.
    """

    required = {
        normalize_column_name(column)
        for column in required_columns
    }

    for sheet_name, df in all_sheets.items():

        available = {
            normalize_column_name(column)
            for column in df.columns
        }

        if required.issubset(available):

            print(f"Using sheet: {sheet_name}")

            return df.copy()

    raise ValueError(
        "\nCould not find a sheet containing these columns:\n"
        f"{required_columns}\n\n"
        f"Available sheets:\n{list(all_sheets.keys())}"
    )


def rename_matching_columns(df, expected_columns):
    """
    Rename matching columns to standard names.
    """

    normalized_to_original = {
        normalize_column_name(column): column
        for column in df.columns
    }

    rename_map = {}

    for expected in expected_columns:

        normalized_expected = normalize_column_name(expected)

        if normalized_expected in normalized_to_original:

            original = normalized_to_original[
                normalized_expected
            ]

            rename_map[original] = expected

    return df.rename(columns=rename_map)


def convert_to_percentage(df, columns):
    """
    Convert values from 0-1 scale to percentages if necessary.
    Values already expressed as percentages remain unchanged.
    """

    for column in columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid_values = df[column].dropna()

        if (
            not valid_values.empty
            and valid_values.max() <= 1.01
        ):
            df[column] = df[column] * 100

    return df


def save_figure(filename):
    """
    Save the current figure at thesis-quality resolution.
    """

    output_path = FIGURE_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.close()


# ============================================================
# LOAD EXCEL WORKBOOK
# ============================================================

if not EXCEL_FILE.exists():

    raise FileNotFoundError(
        f"\nExcel file not found:\n{EXCEL_FILE}"
    )


print("\n==========================================")
print("READING EXPERIMENTAL RESULTS")
print("==========================================")

print(f"\nExcel file:\n{EXCEL_FILE}\n")


all_sheets = pd.read_excel(
    EXCEL_FILE,
    sheet_name=None
)


print("Available sheets:")

for sheet_name in all_sheets:
    print(f" - {sheet_name}")


# ============================================================
# 1. OVERALL RANKING PERFORMANCE
#    COVERAGE GROUPED BY FEATURE CLUSTER
# ============================================================

print("\n==========================================")
print("1. COVERAGE BY FEATURE CLUSTER")
print("==========================================")


# We still load the other ranking measures because
# they are used later by other figures/analyses.
ranking_columns = [
    "Metric",
    "Coverage",
    "Strict Accuracy",
    "Top-set Success"
]


ranking_df = find_sheet_by_columns(
    all_sheets,
    ranking_columns
)


ranking_df = rename_matching_columns(
    ranking_df,
    ranking_columns
)


ranking_df = ranking_df[
    ranking_columns
].copy()


ranking_df = ranking_df.dropna(
    subset=["Metric"]
)


ranking_df = convert_to_percentage(
    ranking_df,
    [
        "Coverage",
        "Strict Accuracy",
        "Top-set Success"
    ]
)


# ============================================================
# FEATURE CLUSTERS
# ============================================================

cluster_map = {

    # 1. Textual and Semantic
    "Text Levenshtein": "Textual and Semantic",
    "Text Jaccard": "Textual and Semantic",
    "Text Semantic Cosine": "Textual and Semantic",

    # 2. DOM Attribute and Element Identity
    "Resource ID Exact": "DOM Attribute and Element Identity",
    "Resource ID Exact Match": "DOM Attribute and Element Identity",

    "Class Exact": "DOM Attribute and Element Identity",
    "Class Exact Match": "DOM Attribute and Element Identity",

    "DOM Attributes": "DOM Attribute and Element Identity",

    "Content-desc Semantic": "DOM Attribute and Element Identity",
    "Content-desc Semantic Cosine": "DOM Attribute and Element Identity",

    # 3. Structural Features
    "Tree Edit Distance": "Structural Features",
    "XPath Levenshtein": "Structural Features",
    "Neighbor Text Jaccard": "Structural Features",

    # 4. Spatial and Geometric Features
    "Position Euclidean": "Spatial and Geometric Features",
    "Size Euclidean": "Spatial and Geometric Features",
    "Bounding Box IoU": "Spatial and Geometric Features"
}


cluster_order = [
    "Textual and Semantic",
    "DOM Attribute and Element Identity",
    "Structural Features",
    "Spatial and Geometric Features"
]


metric_order = {

    "Text Levenshtein": 1,
    "Text Jaccard": 2,
    "Text Semantic Cosine": 3,

    "Resource ID Exact": 1,
    "Resource ID Exact Match": 1,

    "Class Exact": 2,
    "Class Exact Match": 2,

    "DOM Attributes": 3,

    "Content-desc Semantic": 4,
    "Content-desc Semantic Cosine": 4,

    "Tree Edit Distance": 1,
    "XPath Levenshtein": 2,
    "Neighbor Text Jaccard": 3,

    "Position Euclidean": 1,
    "Size Euclidean": 2,
    "Bounding Box IoU": 3
}


ranking_df["Cluster"] = ranking_df["Metric"].map(
    cluster_map
)


ranking_df["Metric Order"] = ranking_df["Metric"].map(
    metric_order
)


# Report metrics that were not assigned to a cluster
unmapped_metrics = ranking_df.loc[
    ranking_df["Cluster"].isna(),
    "Metric"
].tolist()


if unmapped_metrics:

    print(
        "\nWarning: the following metrics were not assigned "
        "to a feature cluster:"
    )

    for metric in unmapped_metrics:
        print(f" - {metric}")


# Keep only metrics used in the four non-visual clusters
coverage_df = ranking_df.dropna(
    subset=["Cluster"]
).copy()


coverage_df["Cluster"] = pd.Categorical(
    coverage_df["Cluster"],
    categories=cluster_order,
    ordered=True
)


coverage_df = coverage_df.sort_values(
    [
        "Cluster",
        "Metric Order"
    ]
)


# ============================================================
# PREPARE GROUPED POSITIONS
# ============================================================

positions = []
metric_labels = []
coverage_values = []
metric_clusters = []

cluster_positions = {
    cluster: []
    for cluster in cluster_order
}


current_y = 0


for cluster in cluster_order:

    cluster_data = coverage_df[
        coverage_df["Cluster"] == cluster
    ]

    for _, row in cluster_data.iterrows():

        positions.append(
            current_y
        )

        metric_labels.append(
            row["Metric"]
        )

        coverage_values.append(
            row["Coverage"]
        )

        metric_clusters.append(
            cluster
        )

        cluster_positions[
            cluster
        ].append(
            current_y
        )

        current_y += 1

    # Add a gap between feature clusters
    current_y += 0.8


# ============================================================
# 1. COVERAGE BY FEATURE CLUSTER
# ============================================================

print("\n==========================================")
print("1. COVERAGE BY FEATURE CLUSTER")
print("==========================================")


ranking_columns = [
    "Metric",
    "Coverage",
    "Strict Accuracy",
    "Top-set Success"
]


ranking_df = find_sheet_by_columns(
    all_sheets,
    ranking_columns
)


ranking_df = rename_matching_columns(
    ranking_df,
    ranking_columns
)


ranking_df = ranking_df[
    ranking_columns
].copy()


ranking_df = ranking_df.dropna(
    subset=["Metric"]
)


ranking_df = convert_to_percentage(
    ranking_df,
    [
        "Coverage",
        "Strict Accuracy",
        "Top-set Success"
    ]
)


# ============================================================
# ASSIGN EACH METRIC TO ITS FEATURE CLUSTER
# ============================================================

cluster_map = {

    # 1. Textual and Semantic
    "Text Levenshtein": "1. Textual and Semantic",
    "Text Jaccard": "1. Textual and Semantic",
    "Text Semantic Cosine": "1. Textual and Semantic",

    # 2. DOM Attribute and Element Identity
    "Resource ID Exact": "2. DOM Attribute and Element Identity",
    "Resource ID Exact Match": "2. DOM Attribute and Element Identity",

    "Class Exact": "2. DOM Attribute and Element Identity",
    "Class Exact Match": "2. DOM Attribute and Element Identity",

    "DOM Attributes": "2. DOM Attribute and Element Identity",

    "Content-desc Semantic": "2. DOM Attribute and Element Identity",
    "Content-desc Semantic Cosine": "2. DOM Attribute and Element Identity",

    # 3. Structural Features
    "Tree Edit Distance": "3. Structural Features",
    "XPath Levenshtein": "3. Structural Features",
    "Neighbor Text Jaccard": "3. Structural Features",

    # 4. Spatial and Geometric
    "Position Euclidean": "4. Spatial and Geometric",
    "Size Euclidean": "4. Spatial and Geometric",
    "Bounding Box IoU": "4. Spatial and Geometric"
}


cluster_order = [
    "1. Textual and Semantic",
    "2. DOM Attribute and Element Identity",
    "3. Structural Features",
    "4. Spatial and Geometric"
]


ranking_df["Cluster"] = ranking_df["Metric"].map(
    cluster_map
)


cluster_df = ranking_df.dropna(
    subset=["Cluster"]
).copy()


# ============================================================
# 1. COVERAGE BY FEATURE CLUSTER
# ============================================================

print("\n==========================================")
print("1. COVERAGE BY FEATURE CLUSTER")
print("==========================================")


ranking_columns = [
    "Metric",
    "Coverage",
    "Strict Accuracy",
    "Top-set Success"
]


ranking_df = find_sheet_by_columns(
    all_sheets,
    ranking_columns
)


ranking_df = rename_matching_columns(
    ranking_df,
    ranking_columns
)


ranking_df = ranking_df[
    ranking_columns
].copy()


ranking_df = ranking_df.dropna(
    subset=["Metric"]
)


ranking_df = convert_to_percentage(
    ranking_df,
    [
        "Coverage",
        "Strict Accuracy",
        "Top-set Success"
    ]
)


# ============================================================
# ASSIGN EACH METRIC TO ITS FEATURE CLUSTER
# ============================================================

cluster_map = {

    # 1. Textual and Semantic
    "Text Levenshtein": "1. Textual and Semantic",
    "Text Jaccard": "1. Textual and Semantic",
    "Text Semantic Cosine": "1. Textual and Semantic",

    # 2. DOM Attribute and Element Identity
    "Resource ID Exact": "2. DOM Attribute and Element Identity",
    "Resource ID Exact Match": "2. DOM Attribute and Element Identity",

    "Class Exact": "2. DOM Attribute and Element Identity",
    "Class Exact Match": "2. DOM Attribute and Element Identity",

    "DOM Attributes": "2. DOM Attribute and Element Identity",

    "Content-desc Semantic": "2. DOM Attribute and Element Identity",
    "Content-desc Semantic Cosine": "2. DOM Attribute and Element Identity",

    # 3. Structural Features
    "Tree Edit Distance": "3. Structural Features",
    "XPath Levenshtein": "3. Structural Features",
    "Neighbor Text Jaccard": "3. Structural Features",

    # 4. Spatial and Geometric
    "Position Euclidean": "4. Spatial and Geometric",
    "Size Euclidean": "4. Spatial and Geometric",
    "Bounding Box IoU": "4. Spatial and Geometric"
}


cluster_order = [
    "1. Textual and Semantic",
    "2. DOM Attribute and Element Identity",
    "3. Structural Features",
    "4. Spatial and Geometric"
]


ranking_df["Cluster"] = ranking_df["Metric"].map(
    cluster_map
)


cluster_df = ranking_df.dropna(
    subset=["Cluster"]
).copy()


# ============================================================
# CALCULATE MEAN COVERAGE FOR EACH CLUSTER
# ============================================================

cluster_coverage = (
    cluster_df
    .groupby("Cluster")["Coverage"]
    .mean()
    .reindex(cluster_order)
)


print("\nMean Coverage by Feature Cluster:")

for cluster, value in cluster_coverage.items():
    print(f"{cluster}: {value:.2f}%")


# ============================================================
# VERTICAL COVERAGE CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)


short_labels = [
    "1. Textual and\nSemantic",
    "2. DOM Attribute and\nElement Identity",
    "3. Structural\nFeatures",
    "4. Spatial and\nGeometric"
]


bars = ax.bar(
    short_labels,
    cluster_coverage.values
)


ax.set_ylabel(
    "Mean Coverage (%)"
)


ax.set_xlabel(
    "Feature Cluster"
)


ax.set_title(
    "Coverage by Feature Cluster"
)


ax.set_ylim(
    0,
    105
)


ax.grid(
    axis="y",
    alpha=0.25
)


# Add percentage above each bar
for bar, value in zip(
    bars,
    cluster_coverage.values
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )


# Save using the previous figure name
save_figure(
    "overall_ranking_performance.png"
)

# ============================================================
# 2. CANDIDATE-LEVEL PRECISION, RECALL AND F1
# ============================================================

print("\n==========================================")
print("2. CANDIDATE-LEVEL PERFORMANCE")
print("==========================================")


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
    classification_columns
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


# Sort by F1-score:
# strongest metrics at the top
classification_df = classification_df.sort_values(
    "F1-score",
    ascending=False,
    na_position="last"
).reset_index(drop=True)


metrics = classification_df[
    "Metric"
].tolist()


y_positions = list(
    range(len(metrics))
)


bar_height = 0.23


fig, ax = plt.subplots(
    figsize=(12, 8)
)


# Precision
ax.barh(
    [
        y - bar_height
        for y in y_positions
    ],
    classification_df["Precision"],
    height=bar_height,
    label="Precision"
)


# Recall
ax.barh(
    y_positions,
    classification_df["Recall"],
    height=bar_height,
    label="Recall"
)


# F1-score
ax.barh(
    [
        y + bar_height
        for y in y_positions
    ],
    classification_df["F1-score"],
    height=bar_height,
    label="F1-score"
)


ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    metrics,
    fontsize=9
)


# Highest F1 at top
ax.invert_yaxis()


ax.set_xlabel(
    "Score (%)"
)

ax.set_ylabel(
    "Metric"
)

ax.set_title(
    "Candidate-Level Precision, Recall and F1-score"
)


ax.set_xlim(
    0,
    103
)


ax.legend(
    loc="lower right"
)


ax.grid(
    axis="x",
    alpha=0.25
)


save_figure(
    "candidate_level_precision_recall_f1.png"
)


# ============================================================
# 3. COVERAGE VS STRICT ACCURACY
# ============================================================

print("\n==========================================")
print("3. COVERAGE VS STRICT ACCURACY")
print("==========================================")


fig, ax = plt.subplots(
    figsize=(13, 8)
)


ax.scatter(
    ranking_df["Coverage"],
    ranking_df["Strict Accuracy"],
    s=45
)


# ------------------------------------------------------------
# Manual label offsets
#
# Metrics around 100% Coverage are crowded.
# Their labels are moved toward the left so they remain
# readable and do not leave the figure.
# ------------------------------------------------------------

label_offsets = {
    "Text Semantic Cosine": (7, 5),
    "Text Levenshtein": (7, 3),
    "Text Jaccard": (7, -7),

    "Content-desc Semantic": (7, 5),
    "Content-desc Semantic Cosine": (7, 5),

    "Neighbor Text Jaccard": (7, 5),

    "Resource ID Exact": (7, 5),
    "Resource ID Exact Match": (7, 5),

    "Bounding Box IoU": (-92, 8),

    "Position Euclidean": (-92, 1),

    "XPath Levenshtein": (-92, -6),

    "Size Euclidean": (-92, -13),

    "Tree Edit Distance": (-105, 8),

    "Class Exact": (-105, 1),
    "Class Exact Match": (-105, 1),

    "DOM Attributes": (-105, -7)
}


for _, row in ranking_df.iterrows():

    coverage = row["Coverage"]
    accuracy = row["Strict Accuracy"]
    metric = row["Metric"]

    if (
        pd.notna(coverage)
        and
        pd.notna(accuracy)
    ):

        dx, dy = label_offsets.get(
            metric,
            (7, 5)
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


# Extra margin on all sides so labels remain visible
ax.set_xlim(
    -5,
    110
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
# 4. PER-APPLICATION F1 DISTRIBUTION
# ============================================================

print("\n==========================================")
print("4. PER-APPLICATION F1 DISTRIBUTION")
print("==========================================")


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


# IMPORTANT:
#
# Undefined F1 values remain NaN.
# They are NOT converted to zero.
#
# This is consistent with the thesis methodology.

valid_f1 = per_app_df.dropna(
    subset=["F1-score"]
).copy()


# ------------------------------------------------------------
# Order metrics by Median F1
#
# This makes the strongest per-application metrics appear
# at the top, matching the logic of Table 5.3.
# ------------------------------------------------------------

metric_medians = (
    valid_f1
    .groupby("Metric")["F1-score"]
    .median()
    .sort_values(
        ascending=False
    )
)


metric_order = metric_medians.index.tolist()


# Collect F1 values for each metric
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


# IMPORTANT:
#
# We intentionally do not use "labels="
# because some Matplotlib versions reject it.

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


# Highest median F1 at top
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


# Extra space at both 0% and 100%
# prevents boxes/markers being hidden by the border
ax.set_xlim(
    -5,
    105
)


ax.grid(
    axis="x",
    alpha=0.25
)


# Explanation below plot
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
# 5. PRINT PER-APPLICATION SUMMARY
# ============================================================

print("\n==========================================")
print("PER-APPLICATION F1 SUMMARY")
print("==========================================\n")


summary_rows = []


for metric in metric_order:

    metric_rows = per_app_df[
        per_app_df["Metric"] == metric
    ]


    # --------------------------------------------------------
    # An application is treated as evaluated if there is
    # candidate-level evaluation information for the metric.
    # --------------------------------------------------------

    evaluated_mask = (
        metric_rows[
            [
                "Precision",
                "Recall",
                "F1-score"
            ]
        ]
        .notna()
        .any(axis=1)
    )


    evaluated_apps = (
        metric_rows.loc[
            evaluated_mask,
            "App"
        ]
        .nunique()
    )


    # --------------------------------------------------------
    # Applications with a defined F1-score
    # --------------------------------------------------------

    defined_rows = metric_rows[
        metric_rows[
            "F1-score"
        ].notna()
    ]


    apps_with_defined_f1 = (
        defined_rows[
            "App"
        ]
        .nunique()
    )


    defined_f1 = defined_rows[
        "F1-score"
    ]


    if not defined_f1.empty:

        median_f1 = (
            defined_f1.median()
        )

        mean_f1 = (
            defined_f1.mean()
        )

    else:

        median_f1 = float("nan")

        mean_f1 = float("nan")


    summary_rows.append(
        {
            "Metric": metric,
            "Apps evaluated": evaluated_apps,
            "Apps with defined F1": apps_with_defined_f1,
            "Median F1": median_f1,
            "Mean F1": mean_f1
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


print(
    summary_df.to_string(
        index=False,
        formatters={
            "Median F1": lambda x: (
                f"{x:.2f}%"
                if pd.notna(x)
                else "NA"
            ),
            "Mean F1": lambda x: (
                f"{x:.2f}%"
                if pd.notna(x)
                else "NA"
            )
        }
    )
)


# ============================================================
# FINISHED
# ============================================================

print("\n==========================================")
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("==========================================")

print(
    f"\nFigures saved in:\n{FIGURE_DIR}\n"
)

print("Generated files:")

print(
    " - overall_ranking_performance.png"
)

print(
    " - candidate_level_precision_recall_f1.png"
)

print(
    " - coverage_vs_strict_accuracy.png"
)

print(
    " - per_application_f1_distribution.png"
)
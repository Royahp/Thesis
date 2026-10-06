from pathlib import Path

import pandas as pd

from src.metrics.text_metrics import (
    levenshtein_distance,
    jaccard_similarity,
    semantic_cosine_similarity,
)
from src.metrics.dom_metrics import (
    exact_match,
    dom_attributes_similarity,
)
from src.metrics.spatial_metrics import (
    parse_bounds,
    position_distance,
    size_distance,
    iou_similarity,
)
GROUND_TRUTH_FILE = Path("data/ground_truth/ground_truth.xlsx")
DATASET_FOLDER = Path("data/raw/xmls")
RESULTS_FOLDER = Path("results")
OUTPUT_FILE = RESULTS_FOLDER / "candidate_ranking.xlsx"


def clean_value(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def rank_metric(results_df, metric_column, higher_is_better):
    ranking_results = []

    group_columns = [
        "App",
        "Widget description",
        "Old widget id",
        "Ground Truth new id",
    ]

    for group_values, group in results_df.groupby(group_columns):

        app, description, old_id, ground_truth_id = group_values

        valid_candidates = group.dropna(
            subset=[metric_column]
        )

        ground_truth_row = group[
            group["Candidate new id"] == ground_truth_id
        ]

        # Metric cannot be evaluated for this widget
        if (
            valid_candidates.empty
            or ground_truth_row.empty
            or pd.isna(
                ground_truth_row.iloc[0][metric_column]
            )
        ):
            ranking_results.append({
                "App": app,
                "Widget description": description,
                "Old widget id": old_id,
                "Ground Truth new id": ground_truth_id,
                "Predicted new id": None,
                "Best score": None,
                "Ground Truth rank": None,
                "Status": "Not applicable",
            })

            continue

        # Select the best score
        if higher_is_better:
            best_score = valid_candidates[metric_column].max()
        else:
            best_score = valid_candidates[metric_column].min()

        best_candidates = valid_candidates[
            valid_candidates[metric_column] == best_score
        ]

        best_candidate_ids = best_candidates[
            "Candidate new id"
        ].tolist()

        # Rank all valid candidates
        valid_candidates = valid_candidates.copy()

        valid_candidates["Rank"] = valid_candidates[
            metric_column
        ].rank(
            method="min",
            ascending=not higher_is_better
        )

        ground_truth_rank = valid_candidates.loc[
            valid_candidates["Candidate new id"] == ground_truth_id,
            "Rank"
        ].iloc[0]

        # Check whether the best candidate matches Ground Truth
        if ground_truth_id in best_candidate_ids:

            if len(best_candidate_ids) == 1:
                status = "Correct"
            else:
                status = "Tie includes Ground Truth"

        else:
            status = "Wrong"

        ranking_results.append({
            "App": app,
            "Widget description": description,
            "Old widget id": old_id,
            "Ground Truth new id": ground_truth_id,
            "Predicted new id": ", ".join(
                str(x) for x in best_candidate_ids
            ),
            "Best score": best_score,
            "Ground Truth rank": ground_truth_rank,
            "Status": status,
        })

    return pd.DataFrame(ranking_results)


# Load Ground Truth
ground_truth = pd.read_excel(GROUND_TRUTH_FILE)

results = []


# Process every Ground Truth widget
for _, row in ground_truth.iterrows():

    app = row["App"]
    old_node = int(row["Widget id_old"])
    correct_new_node = int(row["Widget id_new"])
    description = row["Widget description"]

    old_csv = DATASET_FOLDER / f"{app}_old.csv"
    new_csv = DATASET_FOLDER / f"{app}_new.csv"

    old_df = pd.read_csv(old_csv)
    new_df = pd.read_csv(new_csv)

    # Select the old Ground Truth widget
    old_widget = old_df.iloc[old_node]

    old_text = clean_value(
        old_widget.get("text")
    )
    old_resource_id = clean_value(
    old_widget.get("resource-id")
    )

    old_class = clean_value(
    old_widget.get("class")
    )
    old_content_desc = clean_value(
    old_widget.get("content-desc")
    )
    old_bounds = parse_bounds(
    old_widget.get("bounds")
)
        # Compare this old widget with ALL widgets in the new version
    for candidate_id, new_widget in new_df.iterrows():

        new_text = clean_value(
            new_widget.get("text")
        )

        new_resource_id = clean_value(
            new_widget.get("resource-id")
        )

        new_class = clean_value(
            new_widget.get("class")
        )
        new_content_desc = clean_value(
            new_widget.get("content-desc")
        )
        new_bounds = parse_bounds(
        new_widget.get("bounds")
)

        levenshtein = None
        jaccard = None
        semantic_cosine = None

        resource_id_match = exact_match(
            old_resource_id,
            new_resource_id
        )

        class_match = exact_match(
            old_class,
            new_class
        )
        dom_attributes_score = dom_attributes_similarity(
            old_widget,
            new_widget
        )

        content_desc_cosine = None

        if (
            old_content_desc is not None
            and new_content_desc is not None
        ):
            content_desc_cosine = semantic_cosine_similarity(
                old_content_desc,
                new_content_desc
            )

        if old_text is not None and new_text is not None:

            levenshtein = levenshtein_distance(
                old_text,
                new_text
            )

            jaccard = jaccard_similarity(
                old_text,
                new_text
            )

            semantic_cosine = semantic_cosine_similarity(
                old_text,
                new_text
            )
        position_score = None
        size_score = None
        iou_score = None

        if old_bounds is not None and new_bounds is not None:

            position_score = position_distance(
                old_bounds,
                new_bounds
            )

            size_score = size_distance(
                old_bounds,
                new_bounds
            )

            iou_score = iou_similarity(
                old_bounds,
                new_bounds
            )
        results.append({
            "App": app,
            "Widget description": description,
            "Old widget id": old_node,
            "Ground Truth new id": correct_new_node,
            "Candidate new id": candidate_id,

            "Old text": old_text,
            "Candidate text": new_text,
            "Levenshtein distance on text": levenshtein,
            "Jaccard similarity on text": jaccard,
            "Semantic cosine similarity on text": semantic_cosine,

            "Old resource-id": old_resource_id,
            "Candidate resource-id": new_resource_id,
            "Exact match on resource-id": resource_id_match,

            "Old class": old_class,
            "Candidate class": new_class,
            "Exact match on class": class_match,

            "Is Ground Truth candidate":
                1 if candidate_id == correct_new_node else 0,

            "DOM attributes similarity": dom_attributes_score,

            "Old content-desc": old_content_desc,
            "Candidate content-desc": new_content_desc,
            "Semantic cosine on content-desc": content_desc_cosine,
                        "Old bounds": old_bounds,
            "Candidate bounds": new_bounds,

            "Euclidean distance on position": position_score,
            "Euclidean distance on size": size_score,
            "IoU on bounding box": iou_score,
        })


# Convert candidate comparisons to DataFrame
results_df = pd.DataFrame(results)


# Rank candidates using each text metric
levenshtein_ranking = rank_metric(
    results_df,
    "Levenshtein distance on text",
    higher_is_better=False
)

jaccard_ranking = rank_metric(
    results_df,
    "Jaccard similarity on text",
    higher_is_better=True
)

semantic_ranking = rank_metric(
    results_df,
    "Semantic cosine similarity on text",
    higher_is_better=True
)

resource_id_ranking = rank_metric(
    results_df,
    "Exact match on resource-id",
    higher_is_better=True
)

class_ranking = rank_metric(
    results_df,
    "Exact match on class",
    higher_is_better=True
)
dom_attributes_ranking = rank_metric(
    results_df,
    "DOM attributes similarity",
    higher_is_better=True
)

content_desc_ranking = rank_metric(
    results_df,
    "Semantic cosine on content-desc",
    higher_is_better=True
)
position_ranking = rank_metric(
    results_df,
    "Euclidean distance on position",
    higher_is_better=False
)

size_ranking = rank_metric(
    results_df,
    "Euclidean distance on size",
    higher_is_better=False
)

iou_ranking = rank_metric(
    results_df,
    "IoU on bounding box",
    higher_is_better=True
)


# Create results folder
RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# Save all results to Excel
with pd.ExcelWriter(OUTPUT_FILE) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Candidate Comparisons",
        index=False
    )

    levenshtein_ranking.to_excel(
        writer,
        sheet_name="Levenshtein Ranking",
        index=False
    )

    jaccard_ranking.to_excel(
        writer,
        sheet_name="Jaccard Ranking",
        index=False
    )

    semantic_ranking.to_excel(
        writer,
        sheet_name="Semantic Ranking",
        index=False
    )

    resource_id_ranking.to_excel(
        writer,
        sheet_name="Resource ID Ranking",
        index=False
    )

    class_ranking.to_excel(
        writer,
        sheet_name="Class Ranking",
        index=False
    )

    dom_attributes_ranking.to_excel(
        writer,
        sheet_name="DOM Attributes Ranking",
        index=False
    )

    content_desc_ranking.to_excel(
        writer,
        sheet_name="Content Desc Ranking",
        index=False
    )

    position_ranking.to_excel(
        writer,
        sheet_name="Position Ranking",
        index=False
    )

    size_ranking.to_excel(
        writer,
        sheet_name="Size Ranking",
        index=False
    )

    iou_ranking.to_excel(
        writer,
        sheet_name="IoU Ranking",
        index=False
    )


print("Ground Truth rows:", len(ground_truth))
print("Candidate comparisons:", len(results_df))
print("Results saved to:")
print(OUTPUT_FILE)
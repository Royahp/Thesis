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
from src.metrics.structural_metrics import (
    tree_edit_distance,
    xpath_levenshtein_distance,
)
from src.evaluation.xml_utils import (
    load_xml,
    find_xml_node,
    xml_node_to_tree_text,
    get_node_xpath,
    get_neighbor_text,
)


GROUND_TRUTH_FILE = Path("data/ground_truth/ground_truth.xlsx")
DATASET_FOLDER = Path("data/raw/xmls")
RESULTS_FOLDER = Path("results")
OUTPUT_FILE = RESULTS_FOLDER / "candidate_ranking.xlsx"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clean_value(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def make_candidate_label(
    candidate_id,
    candidate_description_map,
    candidate_text,
    candidate_resource_id,
):
    """Create a readable label for a new-version candidate in the Excel output."""

    description = candidate_description_map.get(candidate_id)
    if description is not None and not pd.isna(description):
        description = str(description).strip()
        if description:
            return description

    if candidate_text is not None:
        return candidate_text

    if candidate_resource_id is not None:
        # Keep only the final part when resource-id looks like package:id/name.
        if "/" in candidate_resource_id:
            return candidate_resource_id.split("/")[-1]
        return candidate_resource_id

    return f"candidate {candidate_id}"


def rank_metric(results_df, metric_column, higher_is_better):
    """
    Produce one summary row per old Ground-Truth widget for one metric.

    Ground Truth is used only for evaluation after candidate scores are computed.
    """

    ranking_results = []

    group_columns = [
        "App",
        "Widget description",
        "Old widget id",
        "Ground Truth new id",
    ]

    for group_values, group in results_df.groupby(group_columns):
        app, description, old_id, ground_truth_id = group_values

        valid_candidates = group.dropna(subset=[metric_column])

        ground_truth_row = group[
            group["Candidate new id"] == ground_truth_id
        ]

        # Metric cannot be evaluated for this Ground-Truth widget.
        if (
            valid_candidates.empty
            or ground_truth_row.empty
            or pd.isna(ground_truth_row.iloc[0][metric_column])
        ):
            ranking_results.append(
                {
                    "App": app,
                    "Widget description": description,
                    "Old widget id": old_id,
                    "Ground Truth new id": ground_truth_id,
                    "Predicted new id": None,
                    "Best score": None,
                    "Ground Truth rank": None,
                    "Best candidate count": None,
                    "Best set contains Ground Truth": None,
                    "Status": "Not applicable",
                }
            )
            continue

        # Similarities: maximum is best. Distances: minimum is best.
        if higher_is_better:
            best_score = valid_candidates[metric_column].max()
        else:
            best_score = valid_candidates[metric_column].min()

        best_candidates = valid_candidates[
            valid_candidates[metric_column] == best_score
        ]

        best_candidate_ids = best_candidates["Candidate new id"].tolist()
        best_candidate_count = len(best_candidate_ids)
        best_set_contains_gt = ground_truth_id in best_candidate_ids

        # Rank every valid candidate. Ties receive the same minimum rank.
        valid_candidates = valid_candidates.copy()
        valid_candidates["Rank"] = valid_candidates[metric_column].rank(
            method="min",
            ascending=not higher_is_better,
        )

        ground_truth_rank = valid_candidates.loc[
            valid_candidates["Candidate new id"] == ground_truth_id,
            "Rank",
        ].iloc[0]

        # Keep the existing high-level status definition.
        if best_set_contains_gt:
            if best_candidate_count == 1:
                status = "Correct"
            else:
                status = "Tie includes Ground Truth"
        else:
            status = "Wrong"

        ranking_results.append(
            {
                "App": app,
                "Widget description": description,
                "Old widget id": old_id,
                "Ground Truth new id": ground_truth_id,
                "Predicted new id": ", ".join(
                    str(x) for x in best_candidate_ids
                ),
                "Best score": best_score,
                "Ground Truth rank": ground_truth_rank,
                "Best candidate count": best_candidate_count,
                "Best set contains Ground Truth": best_set_contains_gt,
                "Status": status,
            }
        )

    return pd.DataFrame(ranking_results)


def add_pairwise_metric_details(results_df, metric_configs):
    """
    Add rank / is-best / best-count / detailed pair result for every candidate pair.

    This is the part requested by the supervisor: every old-widget vs new-candidate
    comparison remains visible, even when that candidate is not the best one.
    """

    group_columns = [
        "App",
        "Widget description",
        "Old widget id",
        "Ground Truth new id",
    ]

    for metric_column, higher_is_better in metric_configs.items():
        if metric_column not in results_df.columns:
            print(f"Skipping missing metric column: {metric_column}")
            continue

        rank_column = f"{metric_column} - Rank"
        best_column = f"{metric_column} - Is Best"
        best_count_column = f"{metric_column} - Best Count"
        result_column = f"{metric_column} - Pair Result"

        # Rank every valid candidate within the same old-widget comparison group.
        results_df[rank_column] = (
            results_df.groupby(group_columns)[metric_column]
            .rank(
                method="min",
                ascending=not higher_is_better,
            )
        )

        # Rank == 1 means this candidate belongs to the best candidate set.
        results_df[best_column] = (
            results_df[rank_column] == 1
        )

        # Number of candidates sharing the best score for this old widget.
        results_df[best_count_column] = (
            results_df.groupby(group_columns)[best_column]
            .transform("sum")
            .astype(int)
        )

        def pair_result(row):
            score = row[metric_column]

            if pd.isna(score):
                return "Not applicable"

            is_ground_truth = (
                row["Candidate new id"]
                == row["Ground Truth new id"]
            )
            is_best = bool(row[best_column])
            best_count = int(row[best_count_column])

            if is_ground_truth and is_best:
                if best_count == 1:
                    return "Correct pair - Unique best"
                return "Correct pair - Tied best"

            if is_ground_truth and not is_best:
                return "Correct pair - Not best"

            if (not is_ground_truth) and is_best:
                if best_count == 1:
                    return "Incorrect pair - Best candidate"
                return "Incorrect pair - Tied best"

            return "Incorrect pair"

        results_df[result_column] = results_df.apply(
            pair_result,
            axis=1,
        )

    return results_df


def summarize_ranking(metric_name, ranking_df):
    correct = (ranking_df["Status"] == "Correct").sum()

    tie = (
        ranking_df["Status"]
        == "Tie includes Ground Truth"
    ).sum()

    wrong = (ranking_df["Status"] == "Wrong").sum()

    not_applicable = (
        ranking_df["Status"]
        == "Not applicable"
    ).sum()

    total = len(ranking_df)
    applicable = correct + tie + wrong

    strict_accuracy = None
    top_set_success = None
    coverage = None
    tie_rate = None
    overall_correct_rate = None
    overall_top_set_rate = None

    if applicable > 0:
        strict_accuracy = correct / applicable
        top_set_success = (correct + tie) / applicable
        tie_rate = tie / applicable

    if total > 0:
        coverage = applicable / total
        overall_correct_rate = correct / total
        overall_top_set_rate = (correct + tie) / total

    return {
        "Metric": metric_name,
        "Correct": correct,
        "Tie": tie,
        "Wrong": wrong,
        "Not applicable": not_applicable,
        "Applicable": applicable,
        "Total": total,
        "Coverage": coverage,
        "Strict Accuracy": strict_accuracy,
        "Top-set Success": top_set_success,
        "Tie Rate": tie_rate,
        "Overall Correct Rate": overall_correct_rate,
        "Overall Top-set Rate": overall_top_set_rate,
    }


# -----------------------------------------------------------------------------
# Load Ground Truth
# -----------------------------------------------------------------------------

ground_truth = pd.read_excel(GROUND_TRUTH_FILE)

# Normalize GT IDs once so candidate-description lookup is reliable.
ground_truth["Widget id_old"] = ground_truth["Widget id_old"].astype(int)
ground_truth["Widget id_new"] = ground_truth["Widget id_new"].astype(int)

results = []


# -----------------------------------------------------------------------------
# Compare every old Ground-Truth widget with ALL candidates in the new version
# -----------------------------------------------------------------------------

for _, row in ground_truth.iterrows():
    app = row["App"]
    old_node = int(row["Widget id_old"])
    correct_new_node = int(row["Widget id_new"])
    description = row["Widget description"]

    old_csv = DATASET_FOLDER / f"{app}_old.csv"
    new_csv = DATASET_FOLDER / f"{app}_new.csv"

    old_xml = DATASET_FOLDER / f"{app}_old.xml"
    new_xml = DATASET_FOLDER / f"{app}_new.xml"

    old_df = pd.read_csv(old_csv)
    new_df = pd.read_csv(new_csv)

    # Human-readable descriptions for candidates that are also present in GT.
    app_gt = ground_truth[ground_truth["App"] == app]
    candidate_description_map = {
        int(gt_row["Widget id_new"]): gt_row["Widget description"]
        for _, gt_row in app_gt.iterrows()
    }

    # Select the old Ground-Truth widget.
    old_widget = old_df.iloc[old_node]

    old_text = clean_value(old_widget.get("text"))
    old_resource_id = clean_value(old_widget.get("resource-id"))
    old_class = clean_value(old_widget.get("class"))
    old_content_desc = clean_value(old_widget.get("content-desc"))
    old_bounds = parse_bounds(old_widget.get("bounds"))

    # Prepare old XML structural information once per old widget.
    old_root = None
    new_root = None
    old_xml_node = None
    old_tree_text = None
    old_xpath = None
    old_neighbor_text = None

    if old_xml.exists() and new_xml.exists():
        old_root = load_xml(old_xml)
        new_root = load_xml(new_xml)

        old_xml_node = find_xml_node(
            old_root,
            old_widget,
        )

        if old_xml_node is not None:
            old_tree_text = xml_node_to_tree_text(
                old_xml_node
            )

            old_xpath = get_node_xpath(
                old_root,
                old_xml_node,
            )

            old_neighbor_text = get_neighbor_text(
                old_root,
                old_xml_node,
            )

    # Compare this old widget with ALL widgets in the new version.
    for candidate_id, new_widget in new_df.iterrows():
        candidate_id = int(candidate_id)

        new_text = clean_value(new_widget.get("text"))
        new_resource_id = clean_value(new_widget.get("resource-id"))
        new_class = clean_value(new_widget.get("class"))
        new_content_desc = clean_value(new_widget.get("content-desc"))
        new_bounds = parse_bounds(new_widget.get("bounds"))

        candidate_label = make_candidate_label(
            candidate_id,
            candidate_description_map,
            new_text,
            new_resource_id,
        )

        # ----- Text metrics -----
        levenshtein = None
        jaccard = None
        semantic_cosine = None

        if old_text is not None and new_text is not None:
            levenshtein = levenshtein_distance(
                old_text,
                new_text,
            )

            jaccard = jaccard_similarity(
                old_text,
                new_text,
            )

            semantic_cosine = semantic_cosine_similarity(
                old_text,
                new_text,
            )

        # ----- Identity / attribute metrics -----
        resource_id_match = exact_match(
            old_resource_id,
            new_resource_id,
        )

        class_match = exact_match(
            old_class,
            new_class,
        )

        dom_attributes_score = dom_attributes_similarity(
            old_widget,
            new_widget,
        )

        content_desc_cosine = None
        if (
            old_content_desc is not None
            and new_content_desc is not None
        ):
            content_desc_cosine = semantic_cosine_similarity(
                old_content_desc,
                new_content_desc,
            )

        # ----- Spatial metrics -----
        position_score = None
        size_score = None
        iou_score = None

        if old_bounds is not None and new_bounds is not None:
            position_score = position_distance(
                old_bounds,
                new_bounds,
            )

            size_score = size_distance(
                old_bounds,
                new_bounds,
            )

            iou_score = iou_similarity(
                old_bounds,
                new_bounds,
            )

        # ----- Structural / contextual metrics -----
        tree_edit_score = None
        new_xpath = None
        xpath_distance = None
        new_neighbor_text = None
        neighbor_jaccard = None

        if (
            old_root is not None
            and new_root is not None
            and old_xml_node is not None
            and old_tree_text is not None
        ):
            new_xml_node = find_xml_node(
                new_root,
                new_widget,
            )

            if new_xml_node is not None:
                new_tree_text = xml_node_to_tree_text(
                    new_xml_node
                )

                tree_edit_score = tree_edit_distance(
                    old_tree_text,
                    new_tree_text,
                )

                new_xpath = get_node_xpath(
                    new_root,
                    new_xml_node,
                )

                if old_xpath is not None and new_xpath is not None:
                    xpath_distance = xpath_levenshtein_distance(
                        old_xpath,
                        new_xpath,
                    )

                new_neighbor_text = get_neighbor_text(
                    new_root,
                    new_xml_node,
                )

                if old_neighbor_text and new_neighbor_text:
                    neighbor_jaccard = jaccard_similarity(
                        old_neighbor_text,
                        new_neighbor_text,
                    )

        # ----- Pair-level output -----
        is_ground_truth_candidate = (
            candidate_id == correct_new_node
        )

        results.append(
            {
                "App": app,
                "Widget description": description,
                "Old widget id": old_node,
                "Ground Truth new id": correct_new_node,
                "Candidate new id": candidate_id,
                "Candidate description": candidate_label,
                "Comparison": f"{description} -> {candidate_label}",

                "Is Ground Truth candidate": (
                    1 if is_ground_truth_candidate else 0
                ),
                "Pair Status": (
                    "Correct"
                    if is_ground_truth_candidate
                    else "Incorrect"
                ),

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

                "DOM attributes similarity": dom_attributes_score,

                "Old content-desc": old_content_desc,
                "Candidate content-desc": new_content_desc,
                "Semantic cosine on content-desc": content_desc_cosine,

                "Old bounds": old_bounds,
                "Candidate bounds": new_bounds,
                "Euclidean distance on position": position_score,
                "Euclidean distance on size": size_score,
                "IoU on bounding box": iou_score,

                "Tree Edit Distance on DOM": tree_edit_score,
                "Old XPath": old_xpath,
                "Candidate XPath": new_xpath,
                "Levenshtein distance on XPath": xpath_distance,

                "Neighbor text old": old_neighbor_text,
                "Neighbor text candidate": new_neighbor_text,
                "Jaccard similarity on neighbor text": neighbor_jaccard,
            }
        )


# -----------------------------------------------------------------------------
# Candidate-comparison DataFrame
# -----------------------------------------------------------------------------

results_df = pd.DataFrame(results)

# IMPORTANT: the names here must exactly match the score columns above.
metric_configs = {
    "Levenshtein distance on text": False,
    "Jaccard similarity on text": True,
    "Semantic cosine similarity on text": True,
    "Exact match on resource-id": True,
    "Exact match on class": True,
    "DOM attributes similarity": True,
    "Semantic cosine on content-desc": True,
    "Euclidean distance on position": False,
    "Euclidean distance on size": False,
    "IoU on bounding box": True,
    "Tree Edit Distance on DOM": False,
    "Levenshtein distance on XPath": False,
    "Jaccard similarity on neighbor text": True,
}

# Add rank and detailed status for EVERY pair and EVERY metric.
results_df = add_pairwise_metric_details(
    results_df,
    metric_configs,
)


# -----------------------------------------------------------------------------
# One-row-per-old-widget ranking sheets
# -----------------------------------------------------------------------------

levenshtein_ranking = rank_metric(
    results_df,
    "Levenshtein distance on text",
    higher_is_better=False,
)

jaccard_ranking = rank_metric(
    results_df,
    "Jaccard similarity on text",
    higher_is_better=True,
)

semantic_ranking = rank_metric(
    results_df,
    "Semantic cosine similarity on text",
    higher_is_better=True,
)

resource_id_ranking = rank_metric(
    results_df,
    "Exact match on resource-id",
    higher_is_better=True,
)

class_ranking = rank_metric(
    results_df,
    "Exact match on class",
    higher_is_better=True,
)

dom_attributes_ranking = rank_metric(
    results_df,
    "DOM attributes similarity",
    higher_is_better=True,
)

content_desc_ranking = rank_metric(
    results_df,
    "Semantic cosine on content-desc",
    higher_is_better=True,
)

position_ranking = rank_metric(
    results_df,
    "Euclidean distance on position",
    higher_is_better=False,
)

size_ranking = rank_metric(
    results_df,
    "Euclidean distance on size",
    higher_is_better=False,
)

iou_ranking = rank_metric(
    results_df,
    "IoU on bounding box",
    higher_is_better=True,
)

tree_edit_ranking = rank_metric(
    results_df,
    "Tree Edit Distance on DOM",
    higher_is_better=False,
)

xpath_ranking = rank_metric(
    results_df,
    "Levenshtein distance on XPath",
    higher_is_better=False,
)

neighbor_ranking = rank_metric(
    results_df,
    "Jaccard similarity on neighbor text",
    higher_is_better=True,
)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

summary_results = [
    summarize_ranking(
        "Text Levenshtein",
        levenshtein_ranking,
    ),
    summarize_ranking(
        "Text Jaccard",
        jaccard_ranking,
    ),
    summarize_ranking(
        "Text Semantic Cosine",
        semantic_ranking,
    ),
    summarize_ranking(
        "Resource ID Exact Match",
        resource_id_ranking,
    ),
    summarize_ranking(
        "Class Exact Match",
        class_ranking,
    ),
    summarize_ranking(
        "DOM Attributes",
        dom_attributes_ranking,
    ),
    summarize_ranking(
        "Content-desc Semantic Cosine",
        content_desc_ranking,
    ),
    summarize_ranking(
        "Position Euclidean",
        position_ranking,
    ),
    summarize_ranking(
        "Size Euclidean",
        size_ranking,
    ),
    summarize_ranking(
        "Bounding Box IoU",
        iou_ranking,
    ),
    summarize_ranking(
        "Tree Edit Distance",
        tree_edit_ranking,
    ),
    summarize_ranking(
        "XPath Levenshtein",
        xpath_ranking,
    ),
    summarize_ranking(
        "Neighbor Text Jaccard",
        neighbor_ranking,
    ),
]

summary_df = pd.DataFrame(summary_results)


# -----------------------------------------------------------------------------
# Save Excel workbook
# -----------------------------------------------------------------------------

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

with pd.ExcelWriter(OUTPUT_FILE) as writer:
    results_df.to_excel(
        writer,
        sheet_name="Candidate Comparisons",
        index=False,
    )

    levenshtein_ranking.to_excel(
        writer,
        sheet_name="Levenshtein Ranking",
        index=False,
    )

    jaccard_ranking.to_excel(
        writer,
        sheet_name="Jaccard Ranking",
        index=False,
    )

    semantic_ranking.to_excel(
        writer,
        sheet_name="Semantic Ranking",
        index=False,
    )

    resource_id_ranking.to_excel(
        writer,
        sheet_name="Resource ID Ranking",
        index=False,
    )

    class_ranking.to_excel(
        writer,
        sheet_name="Class Ranking",
        index=False,
    )

    dom_attributes_ranking.to_excel(
        writer,
        sheet_name="DOM Attributes Ranking",
        index=False,
    )

    content_desc_ranking.to_excel(
        writer,
        sheet_name="Content Desc Ranking",
        index=False,
    )

    position_ranking.to_excel(
        writer,
        sheet_name="Position Ranking",
        index=False,
    )

    size_ranking.to_excel(
        writer,
        sheet_name="Size Ranking",
        index=False,
    )

    iou_ranking.to_excel(
        writer,
        sheet_name="IoU Ranking",
        index=False,
    )

    tree_edit_ranking.to_excel(
        writer,
        sheet_name="Tree Edit Ranking",
        index=False,
    )

    xpath_ranking.to_excel(
        writer,
        sheet_name="XPath Ranking",
        index=False,
    )

    neighbor_ranking.to_excel(
        writer,
        sheet_name="Neighbor Ranking",
        index=False,
    )

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )


print("Ground Truth rows:", len(ground_truth))
print("Candidate comparisons:", len(results_df))
print("Results saved to:")
print(OUTPUT_FILE)

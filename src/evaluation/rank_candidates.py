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


# ============================================================
# FILE PATHS
# ============================================================

GROUND_TRUTH_FILE = Path(
    "data/ground_truth/ground_truth.xlsx"
)

DATASET_FOLDER = Path(
    "data/raw/xmls"
)

RESULTS_FOLDER = Path(
    "results"
)

OUTPUT_FILE = (
    RESULTS_FOLDER / "candidate_ranking.xlsx"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_value(value):
    """
    Convert missing or empty values to None.
    Original character case is preserved.
    """

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
    """
    Create a readable label for a candidate widget.
    """

    description = candidate_description_map.get(
        candidate_id
    )

    if (
        description is not None
        and not pd.isna(description)
    ):
        description = str(description).strip()

        if description:
            return description

    if candidate_text is not None:
        return candidate_text

    if candidate_resource_id is not None:

        if "/" in candidate_resource_id:
            return candidate_resource_id.split("/")[-1]

        return candidate_resource_id

    return f"candidate {candidate_id}"


# ============================================================
# RANK ONE METRIC
# ============================================================

def rank_metric(
    results_df,
    metric_column,
    higher_is_better,
):
    """
    Produce one ranking row for every old Ground Truth widget.

    Ground Truth is NOT used to select candidates.
    It is used only after ranking for evaluation.
    """

    ranking_results = []

    group_columns = [
        "App",
        "Widget description",
        "Old widget id",
        "Ground Truth new id",
    ]

    for group_values, group in results_df.groupby(
        group_columns
    ):

        (
            app,
            description,
            old_id,
            ground_truth_id,
        ) = group_values

        valid_candidates = group.dropna(
            subset=[metric_column]
        )

        ground_truth_row = group[
            group["Candidate new id"]
            == ground_truth_id
        ]

        # ----------------------------------------------------
        # NOT APPLICABLE
        # ----------------------------------------------------

        if (
            valid_candidates.empty
            or ground_truth_row.empty
            or pd.isna(
                ground_truth_row.iloc[0][metric_column]
            )
        ):

            ranking_results.append(
                {
                    "App": app,
                    "Widget description": description,
                    "Old widget id": old_id,
                    "Ground Truth new id":
                        ground_truth_id,
                    "Predicted new id": None,
                    "Best score": None,
                    "Ground Truth score": None,
                    "Ground Truth rank": None,
                    "Best candidate count": None,
                    "Best set contains Ground Truth":
                        None,
                    "Status": "Not applicable",
                }
            )

            continue

        # ----------------------------------------------------
        # BEST SCORE
        # ----------------------------------------------------

        if higher_is_better:

            best_score = (
                valid_candidates[
                    metric_column
                ].max()
            )

        else:

            best_score = (
                valid_candidates[
                    metric_column
                ].min()
            )

        best_candidates = valid_candidates[
            valid_candidates[metric_column]
            == best_score
        ]

        best_candidate_ids = (
            best_candidates[
                "Candidate new id"
            ].tolist()
        )

        best_candidate_count = len(
            best_candidate_ids
        )

        best_set_contains_gt = (
            ground_truth_id
            in best_candidate_ids
        )

        # ----------------------------------------------------
        # RANK ALL VALID CANDIDATES
        # ----------------------------------------------------

        valid_candidates = valid_candidates.copy()

        valid_candidates["Rank"] = (
            valid_candidates[
                metric_column
            ].rank(
                method="min",
                ascending=not higher_is_better,
            )
        )

        ground_truth_rank = (
            valid_candidates.loc[
                valid_candidates[
                    "Candidate new id"
                ]
                == ground_truth_id,
                "Rank",
            ].iloc[0]
        )

        ground_truth_score = (
            ground_truth_row.iloc[0][
                metric_column
            ]
        )

        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        if best_set_contains_gt:

            if best_candidate_count == 1:
                status = "Correct"

            else:
                status = (
                    "Tie includes Ground Truth"
                )

        else:
            status = "Wrong"

        ranking_results.append(
            {
                "App": app,
                "Widget description": description,
                "Old widget id": old_id,
                "Ground Truth new id":
                    ground_truth_id,

                "Predicted new id":
                    ", ".join(
                        str(x)
                        for x
                        in best_candidate_ids
                    ),

                "Best score":
                    best_score,

                "Ground Truth score":
                    ground_truth_score,

                "Ground Truth rank":
                    ground_truth_rank,

                "Best candidate count":
                    best_candidate_count,

                "Best set contains Ground Truth":
                    best_set_contains_gt,

                "Status":
                    status,
            }
        )

    return pd.DataFrame(
        ranking_results
    )


# ============================================================
# PAIRWISE RANK + TP / FP / FN / TN
# ============================================================

def add_pairwise_metric_details(
    results_df,
    metric_configs,
):
    """
    For every metric and every old-new candidate pair,
    add only:

    - candidate Rank
    - TP / FP / FN / TN Classification

    Extra columns such as Is Best, Best Count and Pair Result
    are intentionally omitted to keep Candidate Comparisons
    easier to read.
    """

    group_columns = [
        "App",
        "Widget description",
        "Old widget id",
        "Ground Truth new id",
    ]

    for (
        metric_column,
        higher_is_better,
    ) in metric_configs.items():

        if metric_column not in results_df.columns:

            print(
                "Skipping missing metric column:",
                metric_column,
            )

            continue

        rank_column = (
            f"{metric_column} - Rank"
        )

        classification_column = (
            f"{metric_column} - Classification"
        )

        # ----------------------------------------------------
        # RANK ALL CANDIDATES FOR EACH OLD WIDGET
        # ----------------------------------------------------

        results_df[rank_column] = (
            results_df
            .groupby(group_columns)[metric_column]
            .rank(
                method="min",
                ascending=not higher_is_better,
            )
        )

        # ----------------------------------------------------
        # DETERMINE WHETHER THE METRIC IS APPLICABLE
        # FOR EACH OLD-WIDGET QUERY
        # ----------------------------------------------------

        applicability_map = {}

        for (
            group_values,
            group,
        ) in results_df.groupby(
            group_columns
        ):

            ground_truth_id = (
                group_values[3]
            )

            valid_candidates = (
                group.dropna(
                    subset=[metric_column]
                )
            )

            ground_truth_row = group[
                group["Candidate new id"]
                == ground_truth_id
            ]

            applicable = True

            if valid_candidates.empty:

                applicable = False

            elif ground_truth_row.empty:

                applicable = False

            elif pd.isna(
                ground_truth_row.iloc[0][
                    metric_column
                ]
            ):

                applicable = False

            applicability_map[
                group_values
            ] = applicable

        # ----------------------------------------------------
        # TP / FP / FN / TN
        # ----------------------------------------------------

        def classify_candidate(row):

            group_key = (
                row["App"],
                row["Widget description"],
                row["Old widget id"],
                row["Ground Truth new id"],
            )

            # Metric cannot evaluate this old widget
            if not applicability_map[group_key]:

                return "Not applicable"

            # This specific candidate does not have
            # the required feature
            if pd.isna(
                row[metric_column]
            ):

                return "Not applicable"

            actual_positive = (
                row["Candidate new id"]
                == row["Ground Truth new id"]
            )

            # Rank 1 = predicted as one of the best candidates
            predicted_positive = (
                row[rank_column] == 1
            )

            if (
                actual_positive
                and predicted_positive
            ):
                return "TP"

            if (
                not actual_positive
                and predicted_positive
            ):
                return "FP"

            if (
                actual_positive
                and not predicted_positive
            ):
                return "FN"

            return "TN"

        results_df[
            classification_column
        ] = results_df.apply(
            classify_candidate,
            axis=1,
        )

    return results_df


# ============================================================
# ORIGINAL RANKING SUMMARY
# ============================================================

def summarize_ranking(
    metric_name,
    ranking_df,
):

    correct = (
        ranking_df["Status"]
        == "Correct"
    ).sum()

    tie = (
        ranking_df["Status"]
        == "Tie includes Ground Truth"
    ).sum()

    wrong = (
        ranking_df["Status"]
        == "Wrong"
    ).sum()

    not_applicable = (
        ranking_df["Status"]
        == "Not applicable"
    ).sum()

    total = len(
        ranking_df
    )

    applicable = (
        correct
        + tie
        + wrong
    )

    strict_accuracy = None
    top_set_success = None
    coverage = None
    tie_rate = None
    overall_correct_rate = None
    overall_top_set_rate = None

    if applicable > 0:

        strict_accuracy = (
            correct / applicable
        )

        top_set_success = (
            (correct + tie)
            / applicable
        )

        tie_rate = (
            tie / applicable
        )

    if total > 0:

        coverage = (
            applicable / total
        )

        overall_correct_rate = (
            correct / total
        )

        overall_top_set_rate = (
            (correct + tie)
            / total
        )

    return {
        "Metric":
            metric_name,

        "Correct":
            correct,

        "Tie":
            tie,

        "Wrong":
            wrong,

        "Not applicable":
            not_applicable,

        "Applicable":
            applicable,

        "Total":
            total,

        "Coverage":
            coverage,

        "Strict Accuracy":
            strict_accuracy,

        "Top-set Success":
            top_set_success,

        "Tie Rate":
            tie_rate,

        "Overall Correct Rate":
            overall_correct_rate,

        "Overall Top-set Rate":
            overall_top_set_rate,
    }


# ============================================================
# TP / FP / FN / TN STATISTICS
# ============================================================

def classification_statistics(
    dataframe,
    classification_column,
):
    """
    Calculate TP, FP, FN, TN,
    Precision, Recall and F1-score.
    """

    valid = dataframe[
        dataframe[
            classification_column
        ].isin(
            [
                "TP",
                "FP",
                "FN",
                "TN",
            ]
        )
    ]

    tp = (
        valid[
            classification_column
        ]
        == "TP"
    ).sum()

    fp = (
        valid[
            classification_column
        ]
        == "FP"
    ).sum()

    fn = (
        valid[
            classification_column
        ]
        == "FN"
    ).sum()

    tn = (
        valid[
            classification_column
        ]
        == "TN"
    ).sum()

    evaluated_pairs = (
        tp
        + fp
        + fn
        + tn
    )

    precision = None
    recall = None
    f1_score = None

    if (
        tp + fp
    ) > 0:

        precision = (
            tp / (tp + fp)
        )

    if (
        tp + fn
    ) > 0:

        recall = (
            tp / (tp + fn)
        )

    if (
        precision is not None
        and recall is not None
        and (
            precision + recall
        ) > 0
    ):

        f1_score = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    return {
        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "TN":
            tn,

        "Evaluated candidate pairs":
            evaluated_pairs,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1-score":
            f1_score,
    }


def make_classification_summary(
    results_df,
    metric_display_names,
):

    rows = []

    for (
        metric_column,
        metric_name,
    ) in metric_display_names.items():

        classification_column = (
            f"{metric_column} - Classification"
        )

        if (
            classification_column
            not in results_df.columns
        ):
            continue

        stats = (
            classification_statistics(
                results_df,
                classification_column,
            )
        )

        rows.append(
            {
                "Metric":
                    metric_name,

                **stats,
            }
        )

    return pd.DataFrame(
        rows
    )


def make_per_app_summary(
    results_df,
    metric_display_names,
):

    rows = []

    for app, app_df in (
        results_df.groupby("App")
    ):

        for (
            metric_column,
            metric_name,
        ) in metric_display_names.items():

            classification_column = (
                f"{metric_column} - Classification"
            )

            if (
                classification_column
                not in app_df.columns
            ):
                continue

            stats = (
                classification_statistics(
                    app_df,
                    classification_column,
                )
            )

            rows.append(
                {
                    "App":
                        app,

                    "Metric":
                        metric_name,

                    **stats,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

ground_truth = pd.read_excel(
    GROUND_TRUTH_FILE
)

ground_truth[
    "Widget id_old"
] = (
    ground_truth[
        "Widget id_old"
    ].astype(int)
)

ground_truth[
    "Widget id_new"
] = (
    ground_truth[
        "Widget id_new"
    ].astype(int)
)

results = []


# ============================================================
# PROCESS EVERY GROUND-TRUTH WIDGET
# ============================================================

for _, row in (
    ground_truth.iterrows()
):

    app = row["App"]

    old_node = int(
        row["Widget id_old"]
    )

    correct_new_node = int(
        row["Widget id_new"]
    )

    description = row[
        "Widget description"
    ]

    old_csv = (
        DATASET_FOLDER
        / f"{app}_old.csv"
    )

    new_csv = (
        DATASET_FOLDER
        / f"{app}_new.csv"
    )

    old_xml = (
        DATASET_FOLDER
        / f"{app}_old.xml"
    )

    new_xml = (
        DATASET_FOLDER
        / f"{app}_new.xml"
    )

    old_df = pd.read_csv(
        old_csv
    )

    new_df = pd.read_csv(
        new_csv
    )

    # --------------------------------------------------------
    # READABLE DESCRIPTIONS FOR CANDIDATES
    # --------------------------------------------------------

    app_ground_truth = (
        ground_truth[
            ground_truth["App"]
            == app
        ]
    )

    candidate_description_map = {
        int(gt_row["Widget id_new"]):
            gt_row[
                "Widget description"
            ]

        for _, gt_row
        in app_ground_truth.iterrows()
    }

    # --------------------------------------------------------
    # OLD GROUND-TRUTH WIDGET
    # --------------------------------------------------------

    old_widget = (
        old_df.iloc[
            old_node
        ]
    )

    old_text = clean_value(
        old_widget.get(
            "text"
        )
    )

    old_resource_id = clean_value(
        old_widget.get(
            "resource-id"
        )
    )

    old_class = clean_value(
        old_widget.get(
            "class"
        )
    )

    old_content_desc = clean_value(
        old_widget.get(
            "content-desc"
        )
    )

    old_bounds = parse_bounds(
        old_widget.get(
            "bounds"
        )
    )

    # --------------------------------------------------------
    # STRUCTURAL DATA
    # --------------------------------------------------------

    old_root = None
    new_root = None

    old_xml_node = None
    old_tree_text = None
    old_xpath = None
    old_neighbor_text = None

    if (
        old_xml.exists()
        and new_xml.exists()
    ):

        old_root = load_xml(
            old_xml
        )

        new_root = load_xml(
            new_xml
        )

        old_xml_node = (
            find_xml_node(
                old_root,
                old_widget,
            )
        )

        if (
            old_xml_node
            is not None
        ):

            old_tree_text = (
                xml_node_to_tree_text(
                    old_xml_node
                )
            )

            old_xpath = (
                get_node_xpath(
                    old_root,
                    old_xml_node,
                )
            )

            old_neighbor_text = (
                get_neighbor_text(
                    old_root,
                    old_xml_node,
                )
            )

    # ========================================================
    # COMPARE WITH ALL WIDGETS IN NEW VERSION
    # ========================================================

    for (
        candidate_id,
        new_widget,
    ) in new_df.iterrows():

        candidate_id = int(
            candidate_id
        )

        new_text = clean_value(
            new_widget.get(
                "text"
            )
        )

        new_resource_id = clean_value(
            new_widget.get(
                "resource-id"
            )
        )

        new_class = clean_value(
            new_widget.get(
                "class"
            )
        )

        new_content_desc = clean_value(
            new_widget.get(
                "content-desc"
            )
        )

        new_bounds = parse_bounds(
            new_widget.get(
                "bounds"
            )
        )

        candidate_label = (
            make_candidate_label(
                candidate_id,
                candidate_description_map,
                new_text,
                new_resource_id,
            )
        )

        # ====================================================
        # TEXT METRICS
        # ====================================================

        levenshtein = None
        jaccard = None
        semantic_cosine = None

        if (
            old_text is not None
            and new_text is not None
        ):

            levenshtein = (
                levenshtein_distance(
                    old_text,
                    new_text,
                )
            )

            jaccard = (
                jaccard_similarity(
                    old_text,
                    new_text,
                )
            )

            semantic_cosine = (
                semantic_cosine_similarity(
                    old_text,
                    new_text,
                )
            )

        # ====================================================
        # IDENTITY / ATTRIBUTE METRICS
        # ====================================================

        resource_id_match = (
            exact_match(
                old_resource_id,
                new_resource_id,
            )
        )

        class_match = (
            exact_match(
                old_class,
                new_class,
            )
        )

        dom_attributes_score = (
            dom_attributes_similarity(
                old_widget,
                new_widget,
            )
        )

        content_desc_cosine = None

        if (
            old_content_desc is not None
            and new_content_desc is not None
        ):

            content_desc_cosine = (
                semantic_cosine_similarity(
                    old_content_desc,
                    new_content_desc,
                )
            )

        # ====================================================
        # SPATIAL METRICS
        # ====================================================

        position_score = None
        size_score = None
        iou_score = None

        if (
            old_bounds is not None
            and new_bounds is not None
        ):

            position_score = (
                position_distance(
                    old_bounds,
                    new_bounds,
                )
            )

            size_score = (
                size_distance(
                    old_bounds,
                    new_bounds,
                )
            )

            iou_score = (
                iou_similarity(
                    old_bounds,
                    new_bounds,
                )
            )

        # ====================================================
        # STRUCTURAL / CONTEXTUAL METRICS
        # ====================================================

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

            new_xml_node = (
                find_xml_node(
                    new_root,
                    new_widget,
                )
            )

            if (
                new_xml_node
                is not None
            ):

                new_tree_text = (
                    xml_node_to_tree_text(
                        new_xml_node
                    )
                )

                tree_edit_score = (
                    tree_edit_distance(
                        old_tree_text,
                        new_tree_text,
                    )
                )

                new_xpath = (
                    get_node_xpath(
                        new_root,
                        new_xml_node,
                    )
                )

                if (
                    old_xpath is not None
                    and new_xpath is not None
                ):

                    xpath_distance = (
                        xpath_levenshtein_distance(
                            old_xpath,
                            new_xpath,
                        )
                    )

                new_neighbor_text = (
                    get_neighbor_text(
                        new_root,
                        new_xml_node,
                    )
                )

                if (
                    old_neighbor_text
                    and new_neighbor_text
                ):

                    neighbor_jaccard = (
                        jaccard_similarity(
                            old_neighbor_text,
                            new_neighbor_text,
                        )
                    )

        # ====================================================
        # SAVE THIS OLD × CANDIDATE PAIR
        # ====================================================

        is_ground_truth_candidate = (
            candidate_id
            == correct_new_node
        )

        results.append(
            {
                "App":
                    app,

                "Widget description":
                    description,

                "Old widget id":
                    old_node,

                "Ground Truth new id":
                    correct_new_node,

                "Candidate new id":
                    candidate_id,

                "Candidate description":
                    candidate_label,

                "Comparison":
                    f"{description} -> "
                    f"{candidate_label}",

                "Actual Pair":
                    (
                        "Ground Truth"
                        if is_ground_truth_candidate
                        else "Non-Ground-Truth"
                    ),

                # --------------------------------------------
                # RAW TEXT
                # --------------------------------------------

                "Old text":
                    old_text,

                "Candidate text":
                    new_text,

                # --------------------------------------------
                # TEXT METRICS
                # --------------------------------------------

                "Levenshtein distance on text":
                    levenshtein,

                "Jaccard similarity on text":
                    jaccard,

                "Semantic cosine similarity on text":
                    semantic_cosine,

                # --------------------------------------------
                # RESOURCE ID
                # --------------------------------------------

                "Old resource-id":
                    old_resource_id,

                "Candidate resource-id":
                    new_resource_id,

                "Exact match on resource-id":
                    resource_id_match,

                # --------------------------------------------
                # CLASS
                # --------------------------------------------

                "Old class":
                    old_class,

                "Candidate class":
                    new_class,

                "Exact match on class":
                    class_match,

                # --------------------------------------------
                # GENERAL ATTRIBUTES
                # --------------------------------------------

                "DOM attributes similarity":
                    dom_attributes_score,

                # --------------------------------------------
                # CONTENT DESCRIPTION
                # --------------------------------------------

                "Old content-desc":
                    old_content_desc,

                "Candidate content-desc":
                    new_content_desc,

                "Semantic cosine on content-desc":
                    content_desc_cosine,

                # --------------------------------------------
                # SPATIAL DATA
                # --------------------------------------------

                "Old bounds":
                    old_bounds,

                "Candidate bounds":
                    new_bounds,

                "Euclidean distance on position":
                    position_score,

                "Euclidean distance on size":
                    size_score,

                "IoU on bounding box":
                    iou_score,

                # --------------------------------------------
                # STRUCTURAL DATA
                # --------------------------------------------

                "Tree Edit Distance on DOM":
                    tree_edit_score,

                "Old XPath":
                    old_xpath,

                "Candidate XPath":
                    new_xpath,

                "Levenshtein distance on XPath":
                    xpath_distance,

                # --------------------------------------------
                # NEIGHBOR CONTEXT
                # --------------------------------------------

                "Neighbor text old":
                    old_neighbor_text,

                "Neighbor text candidate":
                    new_neighbor_text,

                "Jaccard similarity on neighbor text":
                    neighbor_jaccard,
            }
        )


# ============================================================
# CREATE CANDIDATE COMPARISON DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# METRIC CONFIGURATION
#
# True  = larger value is better
# False = smaller value is better
# ============================================================

metric_configs = {

    "Levenshtein distance on text":
        False,

    "Jaccard similarity on text":
        True,

    "Semantic cosine similarity on text":
        True,

    "Exact match on resource-id":
        True,

    "Exact match on class":
        True,

    "DOM attributes similarity":
        True,

    "Semantic cosine on content-desc":
        True,

    "Euclidean distance on position":
        False,

    "Euclidean distance on size":
        False,

    "IoU on bounding box":
        True,

    "Tree Edit Distance on DOM":
        False,

    "Levenshtein distance on XPath":
        False,

    "Jaccard similarity on neighbor text":
        True,
}


metric_display_names = {

    "Levenshtein distance on text":
        "Text Levenshtein",

    "Jaccard similarity on text":
        "Text Jaccard",

    "Semantic cosine similarity on text":
        "Text Semantic Cosine",

    "Exact match on resource-id":
        "Resource ID Exact Match",

    "Exact match on class":
        "Class Exact Match",

    "DOM attributes similarity":
        "DOM Attributes",

    "Semantic cosine on content-desc":
        "Content-desc Semantic Cosine",

    "Euclidean distance on position":
        "Position Euclidean",

    "Euclidean distance on size":
        "Size Euclidean",

    "IoU on bounding box":
        "Bounding Box IoU",

    "Tree Edit Distance on DOM":
        "Tree Edit Distance",

    "Levenshtein distance on XPath":
        "XPath Levenshtein",

    "Jaccard similarity on neighbor text":
        "Neighbor Text Jaccard",
}


# ============================================================
# ADD ONLY RANK + CLASSIFICATION
# ============================================================

results_df = (
    add_pairwise_metric_details(
        results_df,
        metric_configs,
    )
)


# ============================================================
# ORGANIZE CANDIDATE COMPARISON COLUMNS
# ============================================================

general_columns = [
    "App",
    "Widget description",
    "Old widget id",
    "Ground Truth new id",
    "Candidate new id",
    "Candidate description",
    "Comparison",
    "Actual Pair",
]


raw_feature_columns = [
    "Old text",
    "Candidate text",

    "Old resource-id",
    "Candidate resource-id",

    "Old class",
    "Candidate class",

    "Old content-desc",
    "Candidate content-desc",

    "Old bounds",
    "Candidate bounds",

    "Old XPath",
    "Candidate XPath",

    "Neighbor text old",
    "Neighbor text candidate",
]


metric_output_columns = []

for metric_column in metric_configs:

    rank_column = (
        f"{metric_column} - Rank"
    )

    classification_column = (
        f"{metric_column} - Classification"
    )

    metric_output_columns.extend(
        [
            metric_column,
            rank_column,
            classification_column,
        ]
    )


candidate_output_columns = (
    general_columns
    + raw_feature_columns
    + metric_output_columns
)


candidate_output_columns = [
    column
    for column in candidate_output_columns
    if column in results_df.columns
]


results_df = results_df[
    candidate_output_columns
]


# ============================================================
# CREATE RANKING SHEETS
# ============================================================

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


# ============================================================
# ORIGINAL SUMMARY
# ============================================================

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

summary_df = pd.DataFrame(
    summary_results
)


# ============================================================
# CLASSIFICATION SUMMARIES
# ============================================================

classification_summary_df = (
    make_classification_summary(
        results_df,
        metric_display_names,
    )
)


per_app_summary_df = (
    make_per_app_summary(
        results_df,
        metric_display_names,
    )
)


# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE EVERYTHING TO EXCEL
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE
) as writer:

    # --------------------------------------------------------
    # DETAILED OLD × CANDIDATE COMPARISONS
    # --------------------------------------------------------

    results_df.to_excel(
        writer,
        sheet_name="Candidate Comparisons",
        index=False,
    )

    # --------------------------------------------------------
    # METRIC RANKING SHEETS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ORIGINAL OVERALL SUMMARY
    # --------------------------------------------------------

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )

    # --------------------------------------------------------
    # TP / FP / FN / TN OVERALL SUMMARY
    # --------------------------------------------------------

    classification_summary_df.to_excel(
        writer,
        sheet_name="Classification Summary",
        index=False,
    )

    # --------------------------------------------------------
    # TP / FP / FN / TN PER APP
    # --------------------------------------------------------

    per_app_summary_df.to_excel(
        writer,
        sheet_name="Per App Summary",
        index=False,
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print(
    "Ground Truth rows:",
    len(ground_truth),
)

print(
    "Candidate comparisons:",
    len(results_df),
)

print(
    "Applications:",
    ground_truth["App"].nunique(),
)

print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)
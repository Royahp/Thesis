from pathlib import Path

import pandas as pd

from src.metrics.text_metrics import (
    jaccard_similarity,
    levenshtein_distance,
    semantic_cosine_similarity,
)

from src.metrics.structural_metrics import tree_edit_distance

from src.evaluation.xml_utils import (
    load_xml,
    find_xml_node,
    xml_node_to_tree_text,
)
from src.metrics.dom_metrics import exact_match

from src.metrics.spatial_metrics import (
    parse_bounds,
    position_distance,
    size_distance,
    iou_similarity,
)



GROUND_TRUTH_FILE = Path("data/ground_truth/ground_truth.xlsx")
DATASET_FOLDER = Path("data/raw/xmls")
RESULTS_FOLDER = Path("results")
OUTPUT_FILE = RESULTS_FOLDER / "metric_results.xlsx"


def clean_value(value):
    # Convert missing values to None

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value





# Load the ground truth file
ground_truth = pd.read_excel(GROUND_TRUTH_FILE)

print("Ground Truth rows:", len(ground_truth))


# Store all calculated results
results = []


# Process every ground-truth pair
for _, row in ground_truth.iterrows():

    app = row["App"]
    old_node = int(row["Widget id_old"])
    new_node = int(row["Widget id_new"])
    description = row["Widget description"]


    # Build the file names for this application
    old_csv = DATASET_FOLDER / f"{app}_old.csv" # build the path like data/raw/xmls/02_auto_01_carmax_old.csv to read later
    new_csv = DATASET_FOLDER / f"{app}_new.csv"

    old_xml = DATASET_FOLDER / f"{app}_old.xml"
    new_xml = DATASET_FOLDER / f"{app}_new.xml"


    # Check if the CSV files exist,if not exist => make one row null to prevent crash
    if not old_csv.exists() or not new_csv.exists():

        results.append({
            "Widget description": description,
            "Widget id_old": old_node,
            "Widget id_new": new_node,
            "App": app,
            "Text old": None,
            "Text new": None,
            "Jaccard similarity on text": None,
            "Levenshtein distance on text": None,
            "Exact match on resource-id": None,
            "Semantic cosine similarity": None,
            "Tree Edit Distance on DOM": None,
            "Status": "CSV file missing",
            "Exact match on class": None,
            "Semantic cosine on content-desc": None,
            "Euclidean distance on position": None,
            "Euclidean distance on size": None,
            "IoU on bounding box": None,
        })

        continue


    # Load all old and new CSV files,the ones that we built upper with path
    old_df = pd.read_csv(old_csv)
    new_df = pd.read_csv(new_csv)


    # Check if the node numbers are valid
    if old_node >= len(old_df) or new_node >= len(new_df):

        results.append({
            "Widget description": description,
            "Widget id_old": old_node,
            "Widget id_new": new_node,
            "App": app,
            "Text old": None,
            "Text new": None,
            "Jaccard similarity on text": None,
            "Levenshtein distance on text": None,
            "Exact match on resource-id": None,
            "Semantic cosine similarity": None,
            "Tree Edit Distance on DOM": None,
            "Status": "Node index out of range",
            "Exact match on class": None,
            "Semantic cosine on content-desc": None,
            "Euclidean distance on position": None,
            "Euclidean distance on size": None,
            "IoU on bounding box": None,
           
        })

        continue


    # Get specific row of matching old and new widgets
    old_widget = old_df.iloc[old_node]
    new_widget = new_df.iloc[new_node]


    # Read text values
    old_text = clean_value(
        old_widget.get("text")
    )

    new_text = clean_value(
        new_widget.get("text")
    )


    # Read resource IDs
    old_resource_id = clean_value(
        old_widget.get("resource-id")
    )

    new_resource_id = clean_value(
        new_widget.get("resource-id")
    )


    # Calculate text metrics only when both texts exist
    if old_text is not None and new_text is not None:

        jaccard = jaccard_similarity(
            old_text,
            new_text
        )

        levenshtein = levenshtein_distance(
            old_text,
            new_text
        )

        semantic_cosine = semantic_cosine_similarity(
            old_text,
            new_text
        )

    else:

        jaccard = None
        levenshtein = None
        semantic_cosine = None


       # Compare the resource IDs
    resource_id_match = exact_match(
        old_resource_id,
        new_resource_id
    )


    # Read class values
    old_class = clean_value(
        old_widget.get("class")
    )

    new_class = clean_value(
        new_widget.get("class")
    )


    # Compare class values
    class_match = exact_match(
        old_class,
        new_class
    )


    # Read content descriptions
    old_content_desc = clean_value(
        old_widget.get("content-desc")
    )

    new_content_desc = clean_value(
        new_widget.get("content-desc")
    )


    # Compare content descriptions
    if old_content_desc is not None and new_content_desc is not None:

        content_desc_cosine = semantic_cosine_similarity(
            old_content_desc,
            new_content_desc
        )

    else:

        content_desc_cosine = None
    

    # Compare the local DOM structure
    tree_edit = None

    if old_xml.exists() and new_xml.exists():

        old_root = load_xml(old_xml)
        new_root = load_xml(new_xml)


        # Find the same widgets inside the XML files
        old_xml_node = find_xml_node(
            old_root,
            old_widget
        )

        new_xml_node = find_xml_node(
            new_root,
            new_widget
        )


        # Calculate Tree Edit Distance when both nodes are found
        if old_xml_node is not None and new_xml_node is not None:

            old_tree_text = xml_node_to_tree_text(
                old_xml_node
            )

            new_tree_text = xml_node_to_tree_text(
                new_xml_node
            )

            tree_edit = tree_edit_distance(
                old_tree_text,
                new_tree_text
            )
    # Read bounds
    old_bounds = parse_bounds(
        old_widget.get("bounds")
    )

    new_bounds = parse_bounds(
        new_widget.get("bounds")
    )

        # Calculate spatial metrics
    if old_bounds is not None and new_bounds is not None:

        position_dist = position_distance(
            old_bounds,
            new_bounds
        )

        size_dist = size_distance(
            old_bounds,
            new_bounds
        )

        iou = iou_similarity(
            old_bounds,
            new_bounds
        )

    else:

        position_dist = None
        size_dist = None
        iou = None

    # Add one result row for this widget pair
    results.append({
        "Widget description": description,
        "Widget id_old": old_node,
        "Widget id_new": new_node,
        "App": app,
        "Text old": old_text,
        "Text new": new_text,
        "Jaccard similarity on text": jaccard,
        "Levenshtein distance on text": levenshtein,
        "Exact match on resource-id": resource_id_match,
        "Semantic cosine similarity": semantic_cosine,
        "Tree Edit Distance on DOM": tree_edit,
        
        "Exact match on class": class_match,
        "Semantic cosine on content-desc": content_desc_cosine,
                "Euclidean distance on position": position_dist,
        "Euclidean distance on size": size_dist,
        "IoU on bounding box": iou,
        "Status": "OK",
    })


# Convert all results to a table
results_df = pd.DataFrame(results)


# Create the results folder if needed
RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# Save the results to Excel
results_df.to_excel(
    OUTPUT_FILE,
    index=False
)


# Final checks
print()
print("Total result rows:", len(results))
print("Finished.")
print("Results saved to:")
print(OUTPUT_FILE)
print()

print(results_df.head(15))
import math


def parse_bounds(bounds):
    # Convert bounds text to coordinates
    if bounds is None:
        return None

    bounds = str(bounds).strip().replace(";", ",")

    if bounds == "" or bounds.lower() == "nan":
        return None

    try:
        first, second = bounds.split("][")
        first = first.replace("[", "")
        second = second.replace("]", "")

        x1, y1 = map(float, first.split(","))
        x2, y2 = map(float, second.split(","))

        return x1, y1, x2, y2

    except (ValueError, AttributeError):
        return None


def position_distance(bounds1, bounds2):
    # Compare widget positions
    x1_old, y1_old, _, _ = bounds1
    x1_new, y1_new, _, _ = bounds2

    return math.sqrt(
        (x1_old - x1_new) ** 2 +
        (y1_old - y1_new) ** 2
    )


def size_distance(bounds1, bounds2):
    # Compare widget sizes
    x1_old, y1_old, x2_old, y2_old = bounds1
    x1_new, y1_new, x2_new, y2_new = bounds2

    width_old = x2_old - x1_old
    height_old = y2_old - y1_old

    width_new = x2_new - x1_new
    height_new = y2_new - y1_new

    return math.sqrt(
        (width_old - width_new) ** 2 +
        (height_old - height_new) ** 2
    )


def iou_similarity(bounds1, bounds2):
    # Calculate Intersection over Union
    x1_old, y1_old, x2_old, y2_old = bounds1
    x1_new, y1_new, x2_new, y2_new = bounds2

    intersection_x1 = max(x1_old, x1_new)
    intersection_y1 = max(y1_old, y1_new)

    intersection_x2 = min(x2_old, x2_new)
    intersection_y2 = min(y2_old, y2_new)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    old_area = (
        (x2_old - x1_old) *
        (y2_old - y1_old)
    )

    new_area = (
        (x2_new - x1_new) *
        (y2_new - y1_new)
    )

    union_area = (
        old_area +
        new_area -
        intersection_area
    )

    if union_area == 0:
        return None

    return intersection_area / union_area
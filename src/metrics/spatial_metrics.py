import math


def euclidean_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def iou(box1, box2):
    # box format: (x, y, width, height)

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    intersection_width = max(0, right - left)
    intersection_height = max(0, bottom - top)

    intersection_area = intersection_width * intersection_height

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


if __name__ == "__main__":

    # Position test
    old_position = (100, 200)
    new_position = (103, 204)

    position_distance = euclidean_distance(
        old_position,
        new_position
    )

    print("Position Euclidean distance:", position_distance)

    # Size test
    old_size = (120, 40)
    new_size = (124, 43)

    size_distance = euclidean_distance(
        old_size,
        new_size
    )

    print("Size Euclidean distance:", size_distance)

    # Bounding Box IoU test
    old_box = (100, 200, 120, 40)
    new_box = (105, 202, 120, 40)

    overlap = iou(old_box, new_box)

    print("Bounding Box IoU:", overlap)
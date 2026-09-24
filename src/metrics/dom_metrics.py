def exact_match(value1, value2):
    if value1 is None or value2 is None:
        return 0

    return 1 if str(value1) == str(value2) else 0


def class_jaccard_similarity(class1, class2):
    set1 = set(class1.lower().split())
    set2 = set(class2.lower().split())

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 1.0

    return len(intersection) / len(union)


if __name__ == "__main__":

    # Exact Match test
    id1 = "submit-btn"
    id2 = "submit-btn"

    result = exact_match(id1, id2)
    print("Exact match:", result)

    # CSS Class Jaccard test
    class1 = "btn primary large"
    class2 = "large btn primary"

    similarity = class_jaccard_similarity(class1, class2)
    print("CSS class Jaccard similarity:", similarity)
    
from rapidfuzz.distance import Levenshtein
from apted import APTED
from apted.helpers import Tree


def xpath_levenshtein_distance(xpath1, xpath2):
    if xpath1 is None or xpath2 is None:
        return None

    return Levenshtein.distance(str(xpath1), str(xpath2))


def tree_edit_distance(tree1_text, tree2_text):
    tree1 = Tree.from_text(tree1_text)
    tree2 = Tree.from_text(tree2_text)

    return APTED(tree1, tree2).compute_edit_distance()


if __name__ == "__main__":

    xpath1 = "/html/body/div[2]/button"
    xpath2 = "/html/body/div[3]/button"

    distance = xpath_levenshtein_distance(xpath1, xpath2)
    print("XPath Levenshtein distance:", distance)

    tree1 = "{div{button}}"
    tree2 = "{div{span}{button}}"

    ted = tree_edit_distance(tree1, tree2)
    print("Tree Edit Distance:", ted)
    
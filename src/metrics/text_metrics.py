from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


def levenshtein_distance(text1, text2):
    return Levenshtein.distance(text1, text2)


def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 1.0

    return len(intersection) / len(union)


def semantic_cosine_similarity(text1, text2):
    embeddings = model.encode([text1, text2])

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)


if __name__ == "__main__":
    text1 = "Login"
    text2 = "Logn"

    distance = levenshtein_distance(text1, text2)
    print("Levenshtein distance:", distance)

    text3 = "Create New Account"
    text4 = "New Account Create"

    similarity = jaccard_similarity(text3, text4)
    print("Jaccard similarity:", similarity)

    text5 = "Login"
    text6 = "Sign in"

    semantic_similarity = semantic_cosine_similarity(text5, text6)
    print("Semantic cosine similarity:", semantic_similarity)
    
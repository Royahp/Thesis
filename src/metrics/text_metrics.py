from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip().lower()

    # Normalize multiple spaces
    return " ".join(text.split())


def levenshtein_distance(text1, text2):
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    return Levenshtein.distance(text1, text2)


def jaccard_similarity(text1, text2):
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    set1 = set(text1.split())
    set2 = set(text2.split())

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return None

    return len(intersection) / len(union)


def semantic_cosine_similarity(text1, text2):
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return None

    embeddings = model.encode([text1, text2])

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)
import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from skimage.color import rgb2lab, deltaE_ciede2000


# ---------------------------------------------------------
# 1. VISUAL APPEARANCE
# Visual Embeddings + Cosine Similarity
# ---------------------------------------------------------

_weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT

_visual_model = models.shufflenet_v2_x1_0(weights=_weights)

# Remove the final classification layer.
# The model will now return a visual feature vector.
_visual_model.fc = torch.nn.Identity()

_visual_model.eval()

_visual_transform = _weights.transforms()


def extract_visual_embedding(image_path):
    image = Image.open(image_path).convert("RGB")

    image_tensor = _visual_transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = _visual_model(image_tensor)

    return embedding.numpy().flatten()


def visual_cosine_similarity(image1_path, image2_path):
    embedding1 = extract_visual_embedding(image1_path)
    embedding2 = extract_visual_embedding(image2_path)

    similarity = cosine_similarity(
        [embedding1],
        [embedding2]
    )[0][0]

    return float(similarity)


# ---------------------------------------------------------
# 2. COLOR
# CIEDE2000
# ---------------------------------------------------------

def ciede2000_color_difference(rgb1, rgb2):

    color1 = np.array(
        [[[rgb1[0] / 255,
           rgb1[1] / 255,
           rgb1[2] / 255]]]
    )

    color2 = np.array(
        [[[rgb2[0] / 255,
           rgb2[1] / 255,
           rgb2[2] / 255]]]
    )

    lab1 = rgb2lab(color1)
    lab2 = rgb2lab(color2)

    difference = deltaE_ciede2000(lab1, lab2)

    return float(difference[0][0])


# ---------------------------------------------------------
# 3. LOCAL VISUAL FEATURES
# SIFT + Euclidean Distance
# ---------------------------------------------------------

def sift_distance(image1_path, image2_path):

    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        raise ValueError("Could not load one of the images.")

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.match(descriptors1, descriptors2)

    if not matches:
        return None

    distances = [match.distance for match in matches]

    return float(np.mean(distances))


# ---------------------------------------------------------
# 4. LOCAL VISUAL FEATURES
# AKAZE + Hamming Distance
# ---------------------------------------------------------

def akaze_distance(image1_path, image2_path):

    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        raise ValueError("Could not load one of the images.")

    akaze = cv2.AKAZE_create()

    keypoints1, descriptors1 = akaze.detectAndCompute(image1, None)
    keypoints2, descriptors2 = akaze.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.match(descriptors1, descriptors2)

    if not matches:
        return None

    distances = [match.distance for match in matches]

    return float(np.mean(distances))


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    # CIEDE2000 test
    old_color = (25, 118, 210)
    new_color = (30, 120, 215)

    color_difference = ciede2000_color_difference(
        old_color,
        new_color
    )

    print("CIEDE2000 color difference:", color_difference)

    # Image tests will be activated later when
    # real element screenshots are available.

    # Example:
    #
    # image1 = "old_button.png"
    # image2 = "new_button.png"
    #
    # print(
    #     "Visual cosine similarity:",
    #     visual_cosine_similarity(image1, image2)
    # )
    #
    # print(
    #     "SIFT distance:",
    #     sift_distance(image1, image2)
    # )
    #
    # print(
    #     "AKAZE distance:",
    #     akaze_distance(image1, image2)
    # )
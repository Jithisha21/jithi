import cv2
import numpy as np

def load_images(reference_path, test_path):
    """
    Load the reference currency image and the test image.
    """
    reference_image = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    test_image = cv2.imread(test_path, cv2.IMREAD_COLOR)

    if reference_image is None or test_image is None:
        raise FileNotFoundError("Error: Unable to load one or both images.")
    
    return reference_image, test_image

def preprocess_image(image):
    """
    Convert the image to grayscale for feature detection.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def feature_matching(reference_image, test_image):
    """
    Match features between the reference image and the test image.
    """
    orb = cv2.ORB_create()  # ORB detector for feature matching
    keypoints1, descriptors1 = orb.detectAndCompute(reference_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(test_image, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches, keypoints1, keypoints2

def display_matches(reference_image, test_image, keypoints1, keypoints2, matches):
    """
    Display matches between the two images.
    """
    match_image = cv2.drawMatches(
        reference_image, keypoints1, test_image, keypoints2, matches[:20], None, flags=2
    )
    cv2.imshow("Matches", match_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def authenticate_currency(matches, threshold=10):
    """
    Authenticate the currency based on the number of good matches.
    """
    if len(matches) > threshold:
        return True
    return False

if _name_ == "_main_":
    reference_image_path = "reference_note.jpg"  # Path to the reference currency note image
    test_image_path = "test_note.jpg"  # Path to the test currency note image

    try:
        # Load images
        reference_image, test_image = load_images(reference_image_path, test_image_path)

        # Preprocess images
        reference_grayscale = preprocess_image(reference_image)
        test_grayscale = preprocess_image(test_image)

        # Perform feature matching
        matches, keypoints1, keypoints2 = feature_matching(reference_grayscale, test_grayscale)

        # Display matches
        display_matches(reference_image, test_image, keypoints1, keypoints2, matches)

        # Authenticate currency
        if authenticate_currency(matches):
            print("The currency note is authentic!")
        else:
            print("The currency note is not authentic!")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
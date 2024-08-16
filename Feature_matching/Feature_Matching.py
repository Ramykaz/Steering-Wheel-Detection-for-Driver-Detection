import cv2
import numpy as np
import os

# Preprocessing function to preprocess the images
def preprocess(img):
    if img is None:
        print("Error: Image is None.")
        return None
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    alpha = 2.5  # Contrast
    beta = 0.5   # Brightness
    adjusted_image = cv2.convertScaleAbs(gray_img, alpha=alpha, beta=beta)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(adjusted_image)

    return clahe_img

# Function to collect features from all feature images
def collect_features(feature_img_pathfile):
    sift = cv2.SIFT_create()
    all_kp = []
    all_des = []

    for feature_file in os.listdir(feature_img_pathfile):
        feature_img = cv2.imread(os.path.join(feature_img_pathfile, feature_file))
        if feature_img is None:
            print(f"Warning: Unable to load image {feature_file}")
            continue
        
        processed_feature_img = preprocess(feature_img)
        if processed_feature_img is None:
            print(f"Error: Processed feature image is None for {feature_file}")
            continue
        
        kp, des = sift.detectAndCompute(processed_feature_img, None)

        if des is not None:
            all_kp.extend(kp)
            all_des.append(des)
            print(f"Image: {feature_file}")
            print(f"Number of keypoints: {len(kp)}")
            print(f"Descriptors shape: {des.shape}")
        else:
            print(f"No descriptors found for image {feature_file}")
    
    if all_des:
        all_des = np.vstack(all_des)
    else:
        all_des = np.array([])

    return all_kp, all_des

# Function to perform feature matching and detection
def detect_steering_wheel(test_img_pathfile, all_kp, all_des):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Iterate through all test images
    for test_file in os.listdir(test_img_pathfile):
        test_img = cv2.imread(os.path.join(test_img_pathfile, test_file))
        if test_img is None:
            print(f"Warning: Unable to load image {test_file}")
            continue
        
        processed_test_img = preprocess(test_img)
        if processed_test_img is None:
            print(f"Error: Processed test image is None for {test_file}")
            continue

        # Detect keypoints and descriptors in the test image
        kp_test, des_test = sift.detectAndCompute(processed_test_img, None)
        if des_test is None:
            print(f"No descriptors found for test image {test_file}")
            continue

        if len(all_des) == 0:
            print("Error: No descriptors found in the feature set.")
            continue

        # Use BFMatcher to find the best matches between the combined feature set and the test image
        matches = bf.match(all_des, des_test)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 0:
            # Check if any keypoints and descriptors are available
            if len(all_kp) > 0 and len(kp_test) > 0:
                # Draw the matches
                matching_result = cv2.drawMatches(
                    np.zeros_like(processed_test_img), all_kp, processed_test_img, kp_test, matches[:10], None, flags=2)
                
                if matching_result is None or matching_result.size == 0:
                    print("Error: Matching result is None or empty.")
                else:
                    # Show the matching result if not empty
                    cv2.imshow('Feature Matching', matching_result)
                    cv2.waitKey(0)
            else:
                print("No keypoints available for drawing matches.")

            # Find the bounding box around the detected object
            if len(matches) > 5:
                src_pts = np.float32([all_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_test[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Find homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    h, w = processed_test_img.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    # Draw bounding box on the test image
                    test_img_with_box = cv2.polylines(test_img, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                    # Show the detected area with the bounding box
                    cv2.imshow('Detected Steering Wheel', test_img_with_box)
                    cv2.waitKey(0)
        else:
            print("No matches found.")

    cv2.destroyAllWindows()

# Paths to the test and feature image folders
test_img_pathfile = "C:/Users/ramad/Steering_wheel_Detection/Template_matching/cropped_test"
feature_img_pathfile = "C:/Users/ramad/Steering_wheel_Detection/Template_matching/features"

# Collect features from all feature images
all_kp, all_des = collect_features(feature_img_pathfile)

# Debugging: Check collected keypoints and descriptors
print(f"Total keypoints collected: {len(all_kp)}")
print(f"Descriptors shape: {all_des.shape}")

# Run the steering wheel detection
detect_steering_wheel(test_img_pathfile, all_kp, all_des)

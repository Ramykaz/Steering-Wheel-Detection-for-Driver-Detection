import cv2
import numpy as np
import os

#1 function to preproces  the tempalte images using clahe
def preprocess(img):
    if img is None :
        return None
    
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    alpha= 1.5 # contrast
    beta = 0  # brightness
    adjusted_image = cv2.convertScaleAbs(gray_img, alpha=alpha, beta=beta)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(adjusted_image)

    return clahe_img

#2 function to match to match images with templates
def Image_detection(test_folder, template_folder, threshold=0.7):
    template_imgs=[]
    template_names=[]

    for filename in os.listdir(template_folder):
        if filename.lower().endswith(('.png','.jpg', '.jpeg')):
            template_path= os.path.join(template_folder, filename)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is not None:
                preprocessed_template= preprocess(template)
                if preprocessed_template is not None:
                    template_imgs.append(preprocessed_template)
                    template_names.append(filename)
                else:
                    print(f"Error: Preprocessing failed for template image at {template_path}")
            else:
                print(f"Error: Preprocessing failed for template image at {template_path}")

    if not template_imgs:
        print("No templates were loaded.")
        return

    # Loading and Preprocessing Test images
    test_imgs=[]
    for filename in os.listdir(test_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            test_path = os.path.join(test_folder, filename)
            test = cv2.imread(test_path,cv2.IMREAD_COLOR)
            if test is not None:
                preprocessed_test = preprocess(test)
                if preprocessed_test is not None:
                    test_imgs.append((preprocessed_test,filename))
                else:
                    print(f"Error: Preprocessing failed for test image at {test_path}")
            else:
                print(f"Error: Test image not found at {test_path}")

    if not test_imgs:
        print("No test images were loaded.")
        return
    
    # Iterating over each test image to match
     
    for test_img, test_name in test_imgs:  
        test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)   
        for template_img, template_name in zip(template_imgs, template_names):
            
            template_img = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
            w, h = template_img.shape[1], template_img.shape[0]

            # Perform Matching here
            result = cv2.matchTemplate(test_img, template_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(test_img, top_left, bottom_right, (0, 255, 0), 2)
                print(f"Template {template_name} detected in {test_name} with confidence {max_val:.2f}")

        # Display the result
        cv2.imshow(f'Detected Templates in {test_name}', test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()     

#3 function to template match with video samples
def Video_detection(video_file, template_folder,threshold =0.7):

    #loading and processing template files

    template_imgs=[]
    template_names=[]

    for filename in os.listdr(template_folder):
        if filename.lower.endswith('.jpg','.png','.jpeg'):
            template_path = os.path.join(template_folder, filename)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is not None:
                preprocessed_template = preprocess(template)
                if preprocessed_template is not None:
                    template_imgs.append(preprocessed_template)
                    template_names.append(filename)
                else:
                    print(f"Error: Preprocessing failed for template image at {template_path}")
            else:
                print(f"Error: Template image not found at {template_path}")

    if not template_imgs:
        print("No templates were loaded.")
        return
    
    #Opening Video file 

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        return
    while True:
        ret,frame = cap.read()
        if not ret:
            break

        gray_frame=preprocess(frame)        
        for template_img, template_name in zip(template_imgs, template_names):
            w,h= template_img.shape[::-1]

            # Matching the template iwth the video frames
            result = cv2.matchTemplate(gray_frame,template_img,cv2.TM_CCOEFF_NORMED)
            min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold :
                top_left = max_loc
            bottom_right = (top_left[0]+w, top_left[1] +h)
            cv2.rectangle(frame, top_left, bottom_right,(0,255,0),2)
            print(f"Template {template_name} detected with confidence {max_val:.2f}")
        
        # Display the result
        cv2.imshow('Video Template Matching', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
video_file = "C:/Users/ramad/Downloads/opengeo/opengeo/face detection data/1.avi"
test_folder = "Steering_wheel_detection/Template_matching/test"
template_folder ="Steering_wheel_detection/Template_matching/template"                
Image_detection(test_folder, template_folder)
#Video_detection(test_folder,template_folder)
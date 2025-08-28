# Steering-Wheel-Detection-for-Driver-Detection

This repository contains the implementation of a **Steering Wheel Detection** system for driver monitoring using **Computer Vision** techniques. The project focuses on detecting the presence of a steering wheel in images and videos, which can be a critical step in **driver detection systems** for safety and automation applications.

---

## 📌 **Project Overview**
The goal of this project is to:
- Detect the steering wheel in images or video frames.
- Explore multiple methods for detection without using deep learning.
- Provide a modular and lightweight solution for steering wheel detection in real-time or recorded footage.

The detection pipeline uses:
- **Template Matching** with various OpenCV methods.
- **Feature Matching** using traditional feature detectors and descriptors (e.g., SIFT, SURF, ORB).

---

## 🛠 **Techniques Implemented**
1. **Template Matching**
   - Uses OpenCV’s built-in methods:
     - `TM_CCOEFF`
     - `TM_CCOEFF_NORMED` (Best performing in tests)
     - `TM_CCORR`
     - `TM_CCORR_NORMED`
     - `TM_SQDIFF`
     - `TM_SQDIFF_NORMED`
   - Tested for static images and video frames.

2. **Feature Matching**
   - Implemented with:
     - **ORB (Oriented FAST and Rotated BRIEF)**
     - **SIFT (Scale-Invariant Feature Transform)** [if enabled]
     - **SURF (Speeded-Up Robust Features)** [if available]
   - Matching using **BFMatcher** and **FLANN-based matcher**.

---

## 📂 **Repository Structure**
```
Steering-Wheel-Detection-for-Driver-Detection/
│
├── data/                     # Sample images and templates
├── templates/                # Steering wheel template images
├── scripts/
│   ├── template_matching.py  # Template matching implementation
│   ├── feature_matching.py   # Feature-based matching implementation
│   ├── preprocess.py         # Preprocessing pipeline (CLAHE, brightness, blur)
│   └── utils.py              # Helper functions
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── results/                  # Screenshots or video outputs
```

---

## ⚡ **Features**
✔ Preprocessing with **CLAHE**, brightness/contrast adjustment, and Gaussian blur  
✔ Template matching with multiple OpenCV algorithms  
✔ Feature matching using ORB, SIFT, and SURF  
✔ Real-time detection support (video feed or recorded video)  
✔ Accuracy evaluation for different methods  

---

## 🔧 **Installation**
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Steering-Wheel-Detection-for-Driver-Detection.git
cd Steering-Wheel-Detection-for-Driver-Detection
pip install -r requirements.txt
```

---

## ▶ **Usage**

### **1. Template Matching**
Run the script for template matching:
```bash
python scripts/template_matching.py --image path/to/image.jpg --template templates/steering_wheel.jpg
```

For video:
```bash
python scripts/template_matching.py --video path/to/video.mp4 --template templates/steering_wheel.jpg
```

### **2. Feature Matching**
Run feature matching:
```bash
python scripts/feature_matching.py --image path/to/image.jpg --template templates/steering_wheel.jpg
```

---

## ✅ **Preprocessing**
Before detection, you can preprocess the images for better accuracy:
```bash
python scripts/preprocess.py --input data/raw_images/ --output data/preprocessed/
```

---

## 📊 **Performance**
- **Best template matching method:** `TM_CCOEFF_NORMED`
- Feature matching works well under varied lighting but requires good templates.

---

## 🔮 **Future Improvements**
- Integrate **hand detection** for combined steering wheel + hand validation.
- Implement **tracking** for continuous detection in video streams.
- Add **deep learning-based comparison** for benchmarking.


## 📜 **License**
This project is licensed under the MIT License.

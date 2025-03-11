# VGG-16
## **Face Recognition Using VGG-16: Code Explanation, Documentation, and Implementation**  

This Python script implements a simple **face recognition system** using a **pre-trained VGG-16 deep learning model**. It extracts features from face images and compares them to detect matches.  

---

## **📌 Why Use VGG-16?**
### **What is VGG-16?**
VGG-16 is a **deep convolutional neural network (CNN)** that was developed by the **Visual Geometry Group (VGG) at Oxford**. It has **16 layers** and is widely used for image classification and feature extraction.

### **Why is VGG-16 Used for Face Recognition?**
1. **Pre-trained on ImageNet** → The model already understands general features in images.
2. **Good at Feature Extraction** → Deep layers capture complex patterns like facial features.
3. **No Need to Train from Scratch** → Saves time and computational power.
4. **High Accuracy** → Proven to perform well in various image recognition tasks.

---

## **📜 Code Breakdown with Documentation**
### **🔹 Step 1: Import Required Libraries**
```python
import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from numpy.linalg import norm
```
- `os` → For file handling (listing images in directories).  
- `cv2` (OpenCV) → For image processing (loading, resizing, and converting images).  
- `numpy` → For handling numerical operations (arrays and feature extraction).  
- `tensorflow.keras.applications.VGG16` → Loads the **VGG-16** model for feature extraction.  
- `tensorflow.keras.models.Model` → Used to modify the **VGG-16 model**.  
- `numpy.linalg.norm` → Used for calculating the similarity between extracted features.  

---

### **🔹 Step 2: Load the Pre-Trained VGG-16 Model**
```python
# Load pre-trained VGG-16 model without the classification layer
base_model = VGG16(weights="imagenet", include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)
print("VGG-16 Model Loaded Successfully!")
```
- **`VGG16(weights="imagenet", include_top=False)`**  
  - Loads the **VGG-16 model** with pre-trained ImageNet weights.  
  - **`include_top=False`** removes the classification layers, so we only use **feature extraction layers**.  
- **`Model(inputs=base_model.input, outputs=base_model.output)`**  
  - Creates a new model that **outputs feature maps** instead of classification results.  

---

### **🔹 Step 3: Define a Function to Extract Features from an Image**
```python
def extract_features(image_path):
    """Extracts features from an image using VGG-16"""
    img = cv2.imread(image_path)  # Load the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV format) to RGB
    img = cv2.resize(img, (224, 224))  # Resize to 224x224 (VGG-16 input size)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values (0-1)
    
    features = model.predict(img)  # Extract deep features using VGG-16
    return features.flatten()  # Convert to a 1D array
```
- **Reads and processes an image (resize, normalize, convert colors).**  
- **Extracts deep features using VGG-16.**  
- **Returns a flattened feature vector for easy comparison.**  

---

### **🔹 Step 4: Load and Store Known Faces**
```python
# Dictionary to store known faces
known_faces = {}
face_folder = "known_faces/"

# Load and store known faces
for filename in os.listdir(face_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(face_folder, filename)
        features = extract_features(path)  # Extract features for each image
        known_faces[filename] = features  # Store in dictionary

print("Registered Faces:", known_faces.keys())
```
- **Reads all images** in the `known_faces/` folder.  
- **Extracts features** using `extract_features()`.  
- **Stores extracted features** in a dictionary (`known_faces`) with the filename as the key.  

✅ **This creates a "database" of registered faces.**

---

### **🔹 Step 5: Recognize a Face from a Test Image**
```python
def recognize_face(test_image_path):
    """Recognizes a face by comparing features with known faces"""
    test_features = extract_features(test_image_path)  # Extract features for the test image
    best_match = None
    best_score = float("inf")  # Initialize best score to a very high value

    for name, features in known_faces.items():
        # Cosine Similarity Calculation (1 - cosine similarity)
        score = 1 - np.dot(features, test_features) / (norm(features) * norm(test_features))
        
        if score < best_score:  # Lower score means a better match
            best_match = name
            best_score = score

    if best_score < 0.5:  # 0.5 is an arbitrary threshold for a good match
        print(f"Match Found: {best_match} with score {best_score}")
    else:
        print("No Match Found")
```
- Extracts **features from the test image**.  
- Compares them **to all registered faces** using **Cosine Similarity**.  
- **Selects the closest match** (smallest score).  
- If the similarity score is **less than 0.5**, it considers it a match.  

---

### **🔹 Step 6: Test the Face Recognition System**
```python
# Test recognition
recognize_face("test_faces/unknown.jpg")
```
- **Loads an unknown image** from the `test_faces/` folder.  
- **Extracts its features** and compares them with known faces.  
- **Prints the matched face or says "No Match Found".**  

---

## **📌 Key Features of This Face Recognition System**
✅ **Uses Deep Learning (VGG-16)** → No need to manually define face features.  
✅ **Feature Extraction** → Compares faces based on deep features, not just pixel values.  
✅ **Cosine Similarity Matching** → Measures how close the test face is to known faces.  
✅ **Threshold-Based Matching** → Ensures accurate face recognition.  

---

## **📌 Possible Improvements**
### **1️⃣ Add Face Detection**
- Before extracting features, use **OpenCV's Face Detection** to crop faces.
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
```
---

### **2️⃣ Improve Accuracy with a Fine-Tuned Model**
- Train **VGG-16 on a face dataset** (e.g., CelebA, LFW) instead of using general ImageNet weights.

---

### **3️⃣ Use a More Advanced Model**
- Use **FaceNet** or **DeepFace** instead of VGG-16 for state-of-the-art face recognition.

---

## **📌 Conclusion**
✅ This script provides a **basic face recognition system** using **VGG-16 feature extraction** and **Cosine Similarity** for matching.  
✅ It is **lightweight, pre-trained, and efficient** for small-scale face recognition tasks.  
✅ However, for **real-world applications**, we should integrate **face detection, model fine-tuning, and advanced deep learning models like FaceNet**.  

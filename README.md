# Face-detectio🤖

A Python-based **Face Recognition System** that uses machine learning and computer vision techniques to detect and recognize human faces in images or video streams.  

## 🚀 Features
- Detect human faces in real-time using a webcam or from static images.  
- Recognize and differentiate between known and unknown faces.  
- Store and manage face encodings for multiple users.  
- Lightweight, fast, and easy to integrate into other projects.  

## 🛠️ Tech Stack
- **Python 3.11**  
- **OpenCV** – For image and video processing  
- **face_recognition** (dlib based) – For facial feature encoding and recognition  
- **NumPy** – For numerical operations  

## 📂 Project Structure
Face-Recognition/<br>
│-- dataset/ # Folder containing known faces<br>
│-- images/ # Test images for recognition<br>
│-- src/ # Source code<br>
│ │-- train.py # Encode and train known faces<br>
│ │-- recognize.py # Face recognition script<br>
│ │-- utils.py # Helper functions<br>
│-- requirements.txt # Required dependencies<br>
│-- README.md # Project documentation<br>



## ⚙️ Installation
1. Clone the repository:<br>
   ```bash<br>
   git clone https://github.com/subhadipsinha722133/Face-detectio.git
   cd Face-Recognition

python -m venv venv<br>
source venv/bin/activate    # For Linux/Mac<br>
venv\Scripts\activate       # For Windows<br>

pip install -r requirements.txt

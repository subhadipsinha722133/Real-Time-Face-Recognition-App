# 👤 Face Recognition App 🎭

A powerful and user-friendly **Streamlit-based web application** for face recognition! This app allows you to register faces, train recognition models, and perform real-time face detection with just a few clicks! 🚀

## ✨ Features

- 🎯 **Face Registration**: Capture 100 images of a person's face for training
- 🧠 **Model Training**: Train a convolutional neural network (CNN) for face recognition
- 📹 **Real-time Recognition**: Live face detection using your webcam
- 📸 **Image Upload**: Recognize faces from uploaded images
- 📊 **Training Visualization**: View accuracy and loss graphs during training
- 🎨 **User-friendly Interface**: Intuitive Streamlit-based UI

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   https://github.com/subhadipsinha722133/Face-detection.git
   cd face-recognition-app
Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
```
```bash
pip install -r requirements.txt
```
## 🚀 Usage
Run the application:

```bash
streamlit run app.py
```
Open your browser and navigate to the local URL shown in the terminal (typically http://localhost:8501)

Follow these steps:

👤 Register New Face: Capture images of a person

🧠 Train Model: Train the recognition model

📹 Real-time Recognition: Use your webcam for live detection

📸 Recognize from Image: Upload images for recognition

## 📁 Project Structure
text
face-recognition-app/<br>
├── face_recognition_app.py  # Main Streamlit application<br>
├── requirements.txt         # Python dependencies<br>
├── clean_data/             # Processed training data<br>
├── images/                 # Captured face images<br>
├── final_model.h5          # Trained model (generated after training)<br>
└── README.md               # This file<br>

## 🔧 Technical Details
Python Version: 3.9.23

Deep Learning Framework: TensorFlow/Keras

Computer Vision: OpenCV

Web Framework: Streamlit

Face Detection: Haar Cascade Classifier

Model Architecture: Custom CNN (LeNet-inspired)

## 🎯 How It Works
Face Detection: Uses Haar Cascade classifier to detect faces in images/video

Preprocessing: Converts images to grayscale, resizes, and normalizes

Model Training: CNN with convolutional, pooling, and dense layers

Recognition: Real-time prediction using the trained model

## 🤝 Contributing
We welcome contributions! 🎉 Feel free to:

Fork the project

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

# 👨‍💻 Author
Subhadip Sinha 👨‍💻

GitHub: @subhadipsinha722133

Email: sinhasubhadip34@gmail.com

## 🙏 Acknowledgments
OpenCV community for the Haar Cascade classifiers

TensorFlow/Keras teams for the deep learning framework

Streamlit team for the amazing web app framework

# ⚠️ Important Notes
Ensure good lighting conditions for better accuracy

Front-facing faces work best for recognition

The model needs retraining after adding new faces

Webcam access is required for real-time recognition

**⭐ If you find this project useful, please give it a star on GitHub! ⭐**

https://via.placeholder.com/800x400.png?text=Face+Recognition+Demo+Preview

Happy Coding! 😊 🚀 🎉


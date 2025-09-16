import streamlit as st
import cv2
import numpy as np
import os
import pickle
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Try to import ML dependencies with error handling
try:
    from keras.models import Sequential, load_model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from keras.optimizers import Adam
    from keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder
    ML_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    st.error(f"ML dependencies not installed: {e}")
    st.info("Please install required packages: pip install tensorflow scikit-learn")
    ML_IMPORTS_SUCCESSFUL = False

# Set page configuration
st.set_page_config(
    page_title="Face Recognition App",
    page_icon="👤",
    layout="wide"
)

# Define paths and initialize session state
classifier_path = "haarcascade_frontalface_default.xml"
data_dir = os.path.join(os.getcwd(), "clean_data")
img_dir = os.path.join(os.getcwd(), "images")

# Create necessary directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'le' not in st.session_state:
    st.session_state.le = None
if 'classifier' not in st.session_state:
    try:
        # Try to load the classifier from the provided path
        if os.path.exists(classifier_path):
            st.session_state.classifier = cv2.CascadeClassifier(classifier_path)
        else:
            # Try to load from OpenCV's built-in classifiers
            st.session_state.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"Error loading Haar Cascade classifier: {e}")
        st.session_state.classifier = None

# Preprocessing function
def preprocessing(img):
    img = cv2.equalizeHist(img)
    img = img.reshape(100, 100, 1)
    img = img / 255
    return img

# Function to get prediction label
def get_pred_label(pred, le):
    if le and hasattr(le, 'classes_') and len(le.classes_) > 0:
        return le.inverse_transform([pred])[0]
    else:
        # Default labels if no LabelEncoder is available
        labels = ["Person_0", "Person_1", "Person_2", "Person_3", "Person_4", "Person_5", "Person_6"]
        if pred < len(labels):
            return labels[pred]
        else:
            return f"Unknown ({pred})"

# Function to preprocess image for prediction
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255.0
    return img

# Function to register a new face
def register_face():
    st.header("Register New Face")
    
    if st.session_state.classifier is None:
        st.error("Face detector not available. Please check Haar Cascade classifier.")
        return
    
    # Create a temporary directory for images
    with tempfile.TemporaryDirectory() as temp_dir:
        data = []
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Cannot access webcam. Please check your camera settings.")
            return
        
        st.write("Please position your face in the frame. Press 'Stop' to finish capturing.")
        st.write("Captured images: 0/20")
        
        # Create placeholders for the video feed
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        stop_button = st.button("Stop Capturing")
        
        while len(data) < 20 and not stop_button:  # Reduced from 100 to 20 for faster testing
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            
            # Convert frame to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_points = st.session_state.classifier.detectMultiScale(frame, 1.3, 5)
            
            if len(face_points) > 0:
                for x, y, w, h in face_points:
                    face_frame = frame[y:y+h+1, x:x+w+1]
                    if len(data) < 20:  # Reduced from 100 to 20 for faster testing
                        data.append(face_frame)
                        break  # Only take one face per frame
            
                # Draw rectangle around face
                for (x, y, w, h) in face_points:
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the number of captured images
            cv2.putText(frame_rgb, f"{len(data)}/20", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display the frame
            frame_placeholder.image(frame_rgb, channels="RGB")
            status_placeholder.write(f"Captured images: {len(data)}/20")
            
            # Check for stop condition
            if len(data) >= 20 or stop_button:
                break
        
        cap.release()
        
        if len(data) > 0:
            name = st.text_input("Enter Face holder name:")
            if name:
                if st.button("Save Face Data"):
                    for i in range(len(data)):
                        cv2.imwrite(f"{img_dir}/{name}_{i}.jpg", data[i])
                    st.success(f"Face data for {name} saved successfully! ({len(data)} images)")
                    # Retrain the model with new data
                    train_model()
        else:
            st.warning("No face data captured. Please try again.")

# Function to train the model
def train_model():
    st.header("Training Face Recognition Model")
    
    if not ML_IMPORTS_SUCCESSFUL:
        st.error("Machine learning dependencies not available. Please install tensorflow and scikit-learn.")
        return
    
    image_data = []
    labels = []
    
    if not os.path.exists(img_dir):
        st.error("Images directory does not exist.")
        return
    
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        st.warning("No face images found. Please register faces first.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, img_file in enumerate(image_files):
        try:
            image = cv2.imread(os.path.join(img_dir, img_file))
            if image is None:
                continue
            image = cv2.resize(image, (100, 100))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_data.append(image)
            labels.append(str(img_file).split("_")[0])
        except Exception as e:
            st.error(f"Error processing image {img_file}: {e}")
        
        progress_bar.progress((i + 1) / len(image_files))
        status_text.text(f"Processing image {i+1}/{len(image_files)}")
    
    if not image_data:
        st.error("No valid images found.")
        return
    
    image_data = np.array(image_data)
    labels = np.array(labels)
    
    # Save the processed data
    with open(os.path.join(data_dir, "images.p"), "wb") as f:
        pickle.dump(image_data, f)
    with open(os.path.join(data_dir, "labels.p"), "wb") as f:
        pickle.dump(labels, f)
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    st.session_state.le = le
    
    # Preprocess images
    images_processed = np.array(list(map(preprocessing, image_data)))
    
    # Convert labels to categorical
    labels_categorical = to_categorical(labels_encoded)
    
    # Define the model
    def create_model(num_classes):
        model = Sequential()
        model.add(Conv2D(30, (5, 5), input_shape=(100, 100, 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(15, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(Adam(learning_rate=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
        return model
    
    # Create and train the model
    num_classes = len(np.unique(labels_encoded))
    model = create_model(num_classes)
    
    # Train the model with fewer epochs for faster training
    history = model.fit(images_processed, labels_categorical, validation_split=0.1, epochs=5, verbose=0)
    
    # Save the model
    model.save("final_model.h5")
    st.session_state.model = model
    
    # Display training results
    st.success("Model trained successfully!")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    st.pyplot(fig)
    
    # Show class labels
    st.write("Trained classes:", list(le.classes_))

# Function for real-time face recognition
def real_time_recognition():
    st.header("Real-time Face Recognition")
    
    if st.session_state.model is None or st.session_state.le is None:
        st.warning("Please train the model first in the 'Train Model' section.")
        return
    
    if st.session_state.classifier is None:
        st.error("Face detector not available. Please check Haar Cascade classifier.")
        return
    
    # Custom VideoTransformer for face recognition
    class FaceRecognitionTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = st.session_state.model
            self.le = st.session_state.le
            self.classifier = st.session_state.classifier
        
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            faces = self.classifier.detectMultiScale(img, 1.5, 5)
            
            for x, y, w, h in faces:
                face = img[y:y+h, x:x+w]
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Predict label
                try:
                    pred = self.model.predict(preprocess(face), verbose=0)
                    label = get_pred_label(np.argmax(pred), self.le)
                    confidence = np.max(pred)
                    
                    cv2.putText(img, f"{label} ({confidence:.2f})", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                except Exception as e:
                    cv2.putText(img, "Error", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            return img
    
    # Start webcam stream
    try:
        webrtc_ctx = webrtc_streamer(
            key="face-recognition",
            video_transformer_factory=FaceRecognitionTransformer,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
        )
    except Exception as e:
        st.error(f"WebRTC stream error: {e}")
        st.info("Real-time recognition might not work in all environments. Try the image upload option.")

# Function to recognize face from uploaded image
def recognize_from_image():
    st.header("Recognize Face from Uploaded Image")
    
    if st.session_state.model is None or st.session_state.le is None:
        st.warning("Please train the model first in the 'Train Model' section.")
        return
    
    if st.session_state.classifier is None:
        st.error("Face detector not available. Please check Haar Cascade classifier.")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display the uploaded image
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        
        # Detect faces
        faces = st.session_state.classifier.detectMultiScale(img, 1.5, 5)
        
        if len(faces) > 0:
            for x, y, w, h in faces:
                # Extract face
                face = img[y:y+h, x:x+w]
                
                # Predict label
                pred = st.session_state.model.predict(preprocess(face), verbose=0)
                label = get_pred_label(np.argmax(pred), st.session_state.le)
                confidence = np.max(pred)
                
                # Display result
                st.success(f"Recognized: {label} (Confidence: {confidence:.2f})")
                
                # Draw rectangle and label on image
                img_with_rect = img.copy()
                cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img_with_rect, f"{label} ({confidence:.2f})", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                # Display image with recognition
                st.image(cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB), 
                        caption="Recognized Face", use_column_width=True)
        else:
            st.warning("No faces detected in the uploaded image.")

# Function to load a pre-trained model
def load_pretrained_model():
    st.header("Load Pre-trained Model")
    
    if not ML_IMPORTS_SUCCESSFUL:
        st.error("Machine learning dependencies not available. Please install tensorflow and scikit-learn.")
        return
    
    uploaded_model = st.file_uploader("Upload a trained model (h5 file)", type=["h5"])
    
    if uploaded_model is not None:
        try:
            # Save the uploaded model to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                tmp_file.write(uploaded_model.read())
                model_path = tmp_file.name
            
            # Load the model
            model = load_model(model_path)
            st.session_state.model = model
            st.success("Model loaded successfully!")
            
            # Try to load labels if available
            labels_path = os.path.join(data_dir, "labels.p")
            if os.path.exists(labels_path):
                with open(labels_path, "rb") as f:
                    labels = pickle.load(f)
                le = LabelEncoder()
                le.fit(labels)
                st.session_state.le = le
                st.write("Loaded classes:", list(le.classes_))
            
        except Exception as e:
            st.error(f"Error loading model: {e}")

# Main app
def main():
    st.title("Face Recognition App 👤")
    st.markdown("""
    This app allows you to:
    - Register new faces
    - Train a face recognition model
    - Recognize faces in real-time using your webcam
    - Recognize faces from uploaded images
    """)
    
    # Check if OpenCV is working
    if st.session_state.classifier is None:
        st.warning("""
        ⚠️ **Face detector not properly initialized.** 
        
        This might be because:
        1. The Haar Cascade classifier file is missing
        2. OpenCV is not properly installed
        
        Try installing OpenCV with: `pip install opencv-python-headless`
        """)
    
    # Check if ML dependencies are available
    if not ML_IMPORTS_SUCCESSFUL:
        st.warning("""
        ⚠️ **Machine learning dependencies not available.**
        
        Please install the required packages:
        ```
        pip install tensorflow scikit-learn
        ```
        """)
    
    # Sidebar navigation
    app_mode = st.sidebar.selectbox("Choose the app mode", [
        "Home",
        "Register New Face", 
        "Train Model", 
        "Load Model",
        "Real-time Recognition",
        "Recognize from Image"
    ])
    
    if app_mode == "Home":
        st.header("Welcome to the Face Recognition App!")
        st.markdown("""
        ### Instructions:
        1. **Register New Face**: Capture images of a new person's face to add to the database
        2. **Train Model**: Train the face recognition model with all registered faces
        3. **Load Model**: Upload a pre-trained model
        4. **Real-time Recognition**: Use your webcam for live face recognition
        5. **Recognize from Image**: Upload an image to recognize faces in it
        
        ### Requirements:
        - A webcam for face registration and real-time recognition
        - Good lighting conditions for better accuracy
        - Front-facing faces work best
        """)
        
        # Display sample images if available
        if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0:
            st.subheader("Sample Registered Faces")
            sample_images = os.listdir(img_dir)[:5]  # Show first 5 images
            cols = st.columns(len(sample_images))
            
            for col, img_file in zip(cols, sample_images):
                try:
                    img = cv2.imread(os.path.join(img_dir, img_file))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        col.image(img_rgb, caption=img_file.split('_')[0], use_column_width=True)
                except:
                    pass
    
    elif app_mode == "Register New Face":
        register_face()
    
    elif app_mode == "Train Model":
        train_model()
    
    elif app_mode == "Load Model":
        load_pretrained_model()
    
    elif app_mode == "Real-time Recognition":
        real_time_recognition()
    
    elif app_mode == "Recognize from Image":
        recognize_from_image()

if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse

# === CONFIGURATION ===
# Use the current script directory as base directory (cross-platform)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'hsv_baseline_all_images.keras')
image_size = (64, 64)  # Define image_size here


class SugarPredictor:
    def __init__(self, model_path):
        """Initialize the sugar level predictor"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print("Loading model...")
        self.model = load_model(model_path)
        print("✅ Model loaded successfully!")
        
    def preprocess_image(self, img):
        """Preprocess image to match training format"""
        # Resize to training size
        img_resized = cv2.resize(img, image_size)
        
        # Convert to HSV (same as training)
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        
        # Normalize to [0,1] range
        img_normalized = img_hsv.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def predict_sugar_level(self, img):
        """Predict sugar level from image"""
        processed_img = self.preprocess_image(img)
        prediction = self.model.predict(processed_img, verbose=0)
        return prediction[0][0]

def main():
    parser = argparse.ArgumentParser(description='Sugar Level Prediction from Camera')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--model', type=str, default=model_path, help='Path to model file')
    parser.add_argument('--save-image', action='store_true', help='Save captured image')
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = SugarPredictor(args.model)
        
        # Initialize camera
        print(f"Initializing camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {args.camera}")
            return
        
        print("✅ Camera initialized successfully!")
        print("📸 Taking image in 3 seconds...")
        print("3...")
        import time
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("📸 Capturing image...")
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not capture image from camera")
            return
        
        # Save image if requested
        if args.save_image:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_filename = f"captured_sample_{timestamp}.jpg"
            cv2.imwrite(image_filename, frame)
            print(f"💾 Image saved as: {image_filename}")
        
        # Predict sugar level
        print("� Analyzing image...")
        sugar_level = predictor.predict_sugar_level(frame)
        
        print("\n" + "="*40)
        print("🍯 PREDICTION RESULT")
        print("="*40)
        print(f"Sugar Level: {sugar_level:.2f}")
        print("="*40)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Make sure the model file exists and the path is correct.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        print("✅ Done!")

if __name__ == "__main__":
    main()

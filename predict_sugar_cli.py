import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse
import time
import signal
import sys
from PIL import Image, ImageDraw, ImageFont

# Try to import ST7735 display and GPIO (only available on Raspberry Pi)
try:
    import st7735
    DISPLAY_AVAILABLE = True
except ImportError:
    DISPLAY_AVAILABLE = False
    print("⚠️ ST7735 display not available (running on non-Pi system)")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ GPIO not available (running on non-Pi system)")

# === CONFIGURATION ===
# Use the current script directory as base directory (cross-platform)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'hsv_baseline_all_images.keras')
image_size = (64, 64)  # Define image_size here

# GPIO Configuration
BUTTON_PIN = 18  # GPIO pin for push button (change as needed)
button_pressed = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n🛑 Shutting down gracefully...')
    if GPIO_AVAILABLE:
        GPIO.cleanup()
    sys.exit(0)

def button_callback(channel):
    """Callback function for button press"""
    global button_pressed
    button_pressed = True
    print("🔘 Button pressed! Starting capture process...")


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


class DisplayManager:
    def __init__(self):
        """Initialize ST7735 display if available"""
        self.display_available = DISPLAY_AVAILABLE
        self.disp = None
        
        if self.display_available:
            try:
                # Initialize ST7735 display for Raspberry Pi 5 with Enviro Plus
                self.disp = st7735.ST7735(
                    port=0,
                    cs=0,  # BG_SPI_CS_FRONT for Enviro Plus
                    dc=24,                 # PIN21 for Pi 5 with Enviro Plus
                    backlight=None,          # PIN32 for Pi 5 with Enviro Plus
                    rst=25,
                    width=128,
                    height=160,
                    rotation=90,
                    invert=False
                )
                self.disp.begin()
                self.width = self.disp.width
                self.height = self.disp.height
                print("✅ ST7735 display initialized!")
            except Exception as e:
                print(f"⚠️ Failed to initialize display: {e}")
                self.display_available = False
    
    def show_image_and_prediction(self, cv_image, sugar_level):
        """Display captured image and prediction on ST7735 screen"""
        if not self.display_available or self.disp is None:
            print(f"📺 Display not available - Sugar Level: {sugar_level:.2f}")
            return
        
        try:
            # Convert CV2 image (BGR) to PIL image (RGB) - Fix color channel order
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Create display image
            display_img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            draw = ImageDraw.Draw(display_img)
            
            # Calculate image display area (top 2/3 of screen)
            img_display_height = int(self.height * 0.65)
            img_display_width = self.width
            
            # Resize captured image to fit display area
            img_resized = pil_image.resize((img_display_width, img_display_height))
            
            # Fix color mapping for ST7735 - swap R and B channels if needed
            # Convert to numpy array for color channel manipulation
            img_array = np.array(img_resized)
            # Swap red and blue channels to fix color display issue
            img_array[:, :, [0, 2]] = img_array[:, :, [2, 0]]  # Swap R and B
            img_resized_corrected = Image.fromarray(img_array)
            
            # Paste corrected image at top of display
            display_img.paste(img_resized_corrected, (0, 0))
            
            # Load default font
            try:
                # Try to load a larger font if available
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Add text overlay for sugar level (bottom 1/3 of screen)
            text_y = img_display_height + 10
            text_area_height = self.height - img_display_height
            
            # Background for text
            draw.rectangle([0, img_display_height, self.width, self.height], fill=(0, 0, 0))
            
            # Sugar level text - use proper green color (corrected for R/B swap)
            sugar_text = f"Sugar: {sugar_level:.2f}"
            text_bbox = draw.textbbox((0, 0), sugar_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (self.width - text_width) // 2
            
            # Green color corrected for R/B swap: (0, 255, 0) becomes (0, 255, 0) - stays same
            draw.text((text_x, text_y), sugar_text, font=font, fill=(0, 255, 0))
            
            # Status text - white should remain white
            status_text = "Analysis Complete"
            status_bbox = draw.textbbox((0, 0), status_text, font=font)
            status_width = status_bbox[2] - status_bbox[0]
            status_x = (self.width - status_width) // 2
            status_y = text_y + 25
            
            draw.text((status_x, status_y), status_text, font=font, fill=(255, 255, 255))
            
            # Display on screen
            self.disp.display(display_img)
            print(f"📺 Image and prediction displayed on ST7735: {sugar_level:.2f}")
            
        except Exception as e:
            print(f"❌ Error displaying on ST7735: {e}")
            print(f"📺 Fallback - Sugar Level: {sugar_level:.2f}")
    
    def show_startup_message(self):
        """Show startup message on display"""
        if not self.display_available or self.disp is None:
            return
        
        try:
            img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            # Center the text
            text = "Sugar Predictor"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (self.width - text_width) // 2
            text_y = self.height // 2 - 20
            
            # Yellow text - corrected for R/B swap: (255, 255, 0) becomes (0, 255, 255) for proper yellow
            draw.text((text_x, text_y), text, font=font, fill=(0, 255, 255))
            
            status_text = "Starting..."
            status_bbox = draw.textbbox((0, 0), status_text, font=font)
            status_width = status_bbox[2] - status_bbox[0]
            status_x = (self.width - status_width) // 2
            status_y = text_y + 25
            
            # White text remains white
            draw.text((status_x, status_y), status_text, font=font, fill=(255, 255, 255))
            
            self.disp.display(img)
            
        except Exception as e:
            print(f"⚠️ Error showing startup message: {e}")
    
    def show_waiting_message(self):
        """Show waiting for button press message on display"""
        if not self.display_available or self.disp is None:
            return
        
        try:
            img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Center the text
            text1 = "Sugar Predictor"
            text1_bbox = draw.textbbox((0, 0), text1, font=font)
            text1_width = text1_bbox[2] - text1_bbox[0]
            text1_x = (self.width - text1_width) // 2
            text1_y = self.height // 2 - 30
            
            # Yellow text - corrected for R/B swap
            draw.text((text1_x, text1_y), text1, font=font, fill=(0, 255, 255))
            
            text2 = "Press Button"
            text2_bbox = draw.textbbox((0, 0), text2, font=font)
            text2_width = text2_bbox[2] - text2_bbox[0]
            text2_x = (self.width - text2_width) // 2
            text2_y = text1_y + 20
            
            draw.text((text2_x, text2_y), text2, font=font, fill=(255, 255, 255))
            
            text3 = "to Capture"
            text3_bbox = draw.textbbox((0, 0), text3, font=font)
            text3_width = text3_bbox[2] - text3_bbox[0]
            text3_x = (self.width - text3_width) // 2
            text3_y = text2_y + 20
            
            draw.text((text3_x, text3_y), text3, font=font, fill=(255, 255, 255))
            
            self.disp.display(img)
            
        except Exception as e:
            print(f"⚠️ Error showing waiting message: {e}")

def main():
    global button_pressed
    
    parser = argparse.ArgumentParser(description='Sugar Level Prediction from Camera with Button Trigger')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--model', type=str, default=model_path, help='Path to model file')
    parser.add_argument('--save-image', action='store_true', help='Save captured image')
    parser.add_argument('--button-pin', type=int, default=BUTTON_PIN, help='GPIO pin for button (default: 18)')
    parser.add_argument('--no-button', action='store_true', help='Run without button (capture every 5 seconds)')
    args = parser.parse_args()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize GPIO if available
    if GPIO_AVAILABLE and not args.no_button:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(args.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(args.button_pin, GPIO.FALLING, 
                                callback=button_callback, bouncetime=300)
            print(f"✅ GPIO button initialized on pin {args.button_pin}")
            button_enabled = True
        except Exception as e:
            print(f"⚠️ GPIO setup failed: {e}")
            button_enabled = False
    else:
        button_enabled = False
        if args.no_button:
            print("🔄 Running in automatic mode (no button required)")
    
    try:
        # Initialize display manager
        display = DisplayManager()
        display.show_startup_message()
        time.sleep(2)  # Show startup message for 2 seconds
        
        # Initialize predictor
        predictor = SugarPredictor(args.model)
        
        # Initialize camera
        print(f"Initializing camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {args.camera}")
            return
        
        print("✅ Camera initialized successfully!")
        
        if button_enabled:
            print("� System ready! Press the button to capture and analyze...")
            display.show_waiting_message()
        else:
            print("🔄 System ready! Running in automatic mode...")
        
        capture_count = 0
        
        # Main continuous loop
        while True:
            try:
                # Check for button press or automatic trigger
                should_capture = False
                
                if button_enabled and button_pressed:
                    should_capture = True
                    button_pressed = False  # Reset flag
                elif not button_enabled:
                    should_capture = True
                    print(f"⏰ Auto-capture #{capture_count + 1} starting...")
                
                if should_capture:
                    capture_count += 1
                    print(f"\n📸 === CAPTURE #{capture_count} ===")
                    
                    # Countdown
                    print("📸 Taking image in 3 seconds...")
                    for i in [3, 2, 1]:
                        print(f"{i}...")
                        time.sleep(1)
                    
                    print("📸 Capturing image...")
                    
                    # Capture frame
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error: Could not capture image from camera")
                        continue
                    
                    # Save image if requested
                    if args.save_image:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        image_filename = f"captured_sample_{timestamp}.jpg"
                        cv2.imwrite(image_filename, frame)
                        print(f"💾 Image saved as: {image_filename}")
                    
                    # Predict sugar level
                    print("🔍 Analyzing image...")
                    sugar_level = predictor.predict_sugar_level(frame)
                    
                    # Display results on terminal
                    print("\n" + "="*40)
                    print("🍯 PREDICTION RESULT")
                    print("="*40)
                    print(f"Sugar Level: {sugar_level:.2f}")
                    print(f"Capture Count: {capture_count}")
                    print("="*40)
                    
                    # Display image and prediction on ST7735 screen
                    display.show_image_and_prediction(frame, sugar_level)
                    
                    # Keep display on for a few seconds
                    if display.display_available:
                        print("📺 Results displayed on screen for 5 seconds...")
                        time.sleep(5)
                        
                        # Show waiting message again if button is enabled
                        if button_enabled:
                            display.show_waiting_message()
                            print("🔘 Ready for next capture - press button...")
                        else:
                            print("⏰ Waiting 10 seconds before next capture...")
                            time.sleep(10)
                    else:
                        if button_enabled:
                            print("🔘 Ready for next capture - press button...")
                        else:
                            print("⏰ Waiting 15 seconds before next capture...")
                            time.sleep(15)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error during capture cycle: {e}")
                print("⏳ Waiting 5 seconds before retry...")
                time.sleep(5)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Make sure the model file exists and the path is correct.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Clean up
        print("\n🛑 Shutting down...")
        if 'cap' in locals():
            cap.release()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        print("✅ Cleanup completed!")

if __name__ == "__main__":
    main()

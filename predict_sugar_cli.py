import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse
import time
from PIL import Image, ImageDraw, ImageFont

# Try to import GPIO and ST7735 display (only available on Raspberry Pi)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ RPi.GPIO not available (running on non-Pi system)")

try:
    import st7735
    DISPLAY_AVAILABLE = True
except ImportError:
    DISPLAY_AVAILABLE = False
    print("⚠️ ST7735 display not available (running on non-Pi system)")

# === CONFIGURATION ===
# Use the current script directory as base directory (cross-platform)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'hsv_baseline_all_images.keras')
image_size = (64, 64)  # Define image_size here

# GPIO Configuration
BUTTON_PIN = 18  # GPIO pin for the push button (change as needed)
button_pressed = False

def button_callback(channel):
    """Callback function for button press"""
    global button_pressed
    button_pressed = True
    print("🔘 Button pressed! Starting capture...")

def setup_button():
    """Setup GPIO button if available"""
    if not GPIO_AVAILABLE:
        print("⚠️ GPIO not available - button functionality disabled")
        return False
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=300)
        print(f"✅ Button setup on GPIO pin {BUTTON_PIN}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to setup button: {e}")
        return False

def cleanup_gpio():
    """Clean up GPIO resources"""
    if GPIO_AVAILABLE:
        try:
            GPIO.cleanup()
        except:
            pass


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
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Center the title text
            title_text = "Sugar Predictor"
            text_bbox = draw.textbbox((0, 0), title_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (self.width - text_width) // 2
            text_y = self.height // 2 - 30
            
            # Yellow title - corrected for R/B swap
            draw.text((text_x, text_y), title_text, font=font, fill=(0, 255, 255))
            
            # Ready status
            ready_text = "Ready!"
            ready_bbox = draw.textbbox((0, 0), ready_text, font=font)
            ready_width = ready_bbox[2] - ready_bbox[0]
            ready_x = (self.width - ready_width) // 2
            ready_y = text_y + 25
            
            # Green ready text
            draw.text((ready_x, ready_y), ready_text, font=font, fill=(0, 255, 0))
            
            # Instruction text
            instruction_text = "Press button to capture"
            instruction_bbox = draw.textbbox((0, 0), instruction_text, font=small_font)
            instruction_width = instruction_bbox[2] - instruction_bbox[0]
            instruction_x = (self.width - instruction_width) // 2
            instruction_y = ready_y + 30
            
            # White instruction text
            draw.text((instruction_x, instruction_y), instruction_text, font=small_font, fill=(255, 255, 255))
            
            self.disp.display(img)
            
        except Exception as e:
            print(f"⚠️ Error showing waiting message: {e}")

def main():
    parser = argparse.ArgumentParser(description='Sugar Level Prediction from Camera')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--model', type=str, default=model_path, help='Path to model file')
    parser.add_argument('--save-image', action='store_true', help='Save captured image')
    parser.add_argument('--auto-mode', action='store_true', help='Use automatic countdown instead of button')
    args = parser.parse_args()
    
    # Global variable to track button press
    global button_pressed
    
    try:
        # Initialize display manager
        display = DisplayManager()
        display.show_startup_message()
        
        # Setup button if not in auto mode
        button_available = False
        if not args.auto_mode:
            button_available = setup_button()
        
        # Initialize predictor
        predictor = SugarPredictor(args.model)
        
        # Initialize camera
        print(f"Initializing camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {args.camera}")
            return
        
        print("✅ Camera initialized successfully!")
        
        # Wait for trigger (button or auto mode)
        if args.auto_mode:
            print("📸 Auto mode: Taking image in 3 seconds...")
            print("3...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)
            print("📸 Capturing image...")
        else:
            # Show waiting message on display
            display.show_waiting_message()
            
            if button_available:
                print("🔘 Waiting for button press on GPIO pin", BUTTON_PIN)
                print("   (Press the physical button to capture image)")
                
                # Wait for button press
                while not button_pressed:
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                
                print("📸 Button pressed! Capturing image...")
            else:
                # Fallback for systems without GPIO
                print("⚠️ Button not available - falling back to keyboard input")
                print("📸 Press ENTER to capture image...")
                input("   Waiting for ENTER key...")
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
        print("🔍 Analyzing image...")
        sugar_level = predictor.predict_sugar_level(frame)
        
        # Display results on terminal
        print("\n" + "="*40)
        print("🍯 PREDICTION RESULT")
        print("="*40)
        print(f"Sugar Level: {sugar_level:.2f}")
        print("="*40)
        
        # Display image and prediction on ST7735 screen
        display.show_image_and_prediction(frame, sugar_level)
        
        # Keep display on for a few seconds
        if display.display_available:
            print("📺 Results displayed on screen for 10 seconds...")
            time.sleep(10)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Make sure the model file exists and the path is correct.")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        cleanup_gpio()
        print("✅ Done!")

if __name__ == "__main__":
    main()

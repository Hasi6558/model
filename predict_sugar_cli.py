import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse
import time
from PIL import Image, ImageDraw, ImageFont

# Try to import gpiozero and ST7735 display (only available on Raspberry Pi)
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ gpiozero not available (running on non-Pi system)")

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

# Crop Configuration (as percentages of image dimensions for portability)
# Format: (x_start%, y_start%, x_end%, y_end%) where 0.0 = 0% and 1.0 = 100%

# Center crop for 241x269 pixels (assuming 640x480 camera resolution)
# For 640x480 resolution: 241/640 = 0.3766, 269/480 = 0.5604
# Center position: x_center = 0.5, y_center = 0.5
# x_start = 0.5 - (241/640)/2 = 0.5 - 0.1883 = 0.3117
# x_end = 0.5 + (241/640)/2 = 0.5 + 0.1883 = 0.6883
# y_start = 0.5 - (269/480)/2 = 0.5 - 0.2802 = 0.2198
# y_end = 0.5 + (269/480)/2 = 0.5 + 0.2802 = 0.7802
CROP_REGION = (0.3117, 0.2198, 0.6883, 0.7802)  # Center 241x269 pixels for 640x480

# Alternative crop regions for different camera resolutions:
# For 1280x720 (HD): (0.4059, 0.3132, 0.5941, 0.6868)  # Center 241x269
# For 1920x1080 (FHD): (0.4372, 0.3755, 0.5628, 0.6245)  # Center 241x269
# For 800x600: (0.2994, 0.1758, 0.7006, 0.8242)  # Center 241x269

# Examples of other crop regions:
# (0.0, 0.0, 1.0, 1.0) = Full image (no crop)
# (0.25, 0.25, 0.75, 0.75) = Center 50% of image
# (0.3, 0.2, 0.8, 0.7) = Custom region
# (0.4, 0.4, 0.6, 0.6) = Small center region

# GPIO Configuration
BUTTON_PIN = 17  # GPIO pin for the push button (change as needed)
button_pressed = False
button_obj = None  # Global button object

def button_callback():
    """Callback function for button press"""
    global button_pressed
    if not button_pressed:  # Prevent multiple triggers
        button_pressed = True
        print("🔘 Button press detected! Preparing to capture...")
        # Small delay to ensure clean button release
        time.sleep(0.1)

def setup_button():
    """Setup GPIO button if available"""
    global button_obj
    
    if not GPIO_AVAILABLE:
        print("⚠️ gpiozero not available - button functionality disabled")
        return False
    
    try:
        # Create button object with internal pull-up resistor
        button_obj = Button(BUTTON_PIN, pull_up=True, bounce_time=0.3)
        
        # Assign callback function
        button_obj.when_pressed = button_callback
        
        print(f"✅ Button setup on GPIO pin {BUTTON_PIN}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to setup button: {e}")
        return False

def cleanup_gpio():
    """Clean up GPIO resources"""
    global button_obj
    
    if GPIO_AVAILABLE and button_obj is not None:
        try:
            button_obj.close()
            print("✅ GPIO resources cleaned up")
        except:
            pass


def crop_image(image, crop_region=CROP_REGION):
    """
    Crop image based on percentage coordinates
    
    Args:
        image: OpenCV image (numpy array)
        crop_region: tuple (x_start%, y_start%, x_end%, y_end%)
                    where each value is between 0.0 and 1.0
    
    Returns:
        Cropped image and crop coordinates for display
    """
    height, width = image.shape[:2]
    
    # Convert percentage to pixel coordinates
    x_start = int(crop_region[0] * width)
    y_start = int(crop_region[1] * height)
    x_end = int(crop_region[2] * width)
    y_end = int(crop_region[3] * height)
    
    # Ensure coordinates are within image bounds
    x_start = max(0, min(x_start, width-1))
    y_start = max(0, min(y_start, height-1))
    x_end = max(x_start+1, min(x_end, width))
    y_end = max(y_start+1, min(y_end, height))
    
    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]
    
    # Return cropped image and coordinates for overlay
    crop_coords = (x_start, y_start, x_end, y_end)
    
    return cropped_image, crop_coords


def draw_crop_overlay(image, crop_coords):
    """
    Draw crop region overlay on the original image for display
    
    Args:
        image: Original image to draw on
        crop_coords: tuple (x_start, y_start, x_end, y_end) in pixels
    
    Returns:
        Image with crop overlay
    """
    overlay_img = image.copy()
    x_start, y_start, x_end, y_end = crop_coords
    
    # Draw crop rectangle
    cv2.rectangle(overlay_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    # Add corner markers
    corner_size = 10
    # Top-left corner
    cv2.line(overlay_img, (x_start, y_start), (x_start + corner_size, y_start), (0, 255, 0), 3)
    cv2.line(overlay_img, (x_start, y_start), (x_start, y_start + corner_size), (0, 255, 0), 3)
    
    # Top-right corner
    cv2.line(overlay_img, (x_end, y_start), (x_end - corner_size, y_start), (0, 255, 0), 3)
    cv2.line(overlay_img, (x_end, y_start), (x_end, y_start + corner_size), (0, 255, 0), 3)
    
    # Bottom-left corner
    cv2.line(overlay_img, (x_start, y_end), (x_start + corner_size, y_end), (0, 255, 0), 3)
    cv2.line(overlay_img, (x_start, y_end), (x_start, y_end - corner_size), (0, 255, 0), 3)
    
    # Bottom-right corner
    cv2.line(overlay_img, (x_end, y_end), (x_end - corner_size, y_end), (0, 255, 0), 3)
    cv2.line(overlay_img, (x_end, y_end), (x_end, y_end - corner_size), (0, 255, 0), 3)
    
    # Add crop region label
    cv2.putText(overlay_img, "CROP REGION", (x_start, y_start - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return overlay_img


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
    
    def predict_sugar_level(self, img, use_crop=True, crop_region=CROP_REGION):
        """
        Predict sugar level from image
        
        Args:
            img: Input image (BGR format)
            use_crop: Whether to crop the image before prediction
            crop_region: Crop region as (x_start%, y_start%, x_end%, y_end%)
        
        Returns:
            prediction: Sugar level prediction
            processed_img: The actual image used for prediction (for display)
            crop_coords: Crop coordinates if cropping was used
        """
        processed_img = img.copy()
        crop_coords = None
        
        if use_crop:
            # Crop the image
            processed_img, crop_coords = crop_image(img, crop_region)
            print(f"📐 Using cropped region: {crop_region} -> {crop_coords}")
        
        # Preprocess and predict
        img_preprocessed = self.preprocess_image(processed_img)
        prediction = self.model.predict(img_preprocessed, verbose=0)
        
        return prediction[0][0], processed_img, crop_coords


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
    
    def show_image_and_prediction(self, cv_image, sugar_level, processed_img=None, crop_coords=None, show_original=False):
        """
        Display captured image and prediction on ST7735 screen
        
        Args:
            cv_image: Original captured image
            sugar_level: Predicted sugar level
            processed_img: Processed/cropped image used for prediction
            crop_coords: Crop coordinates if cropping was used
            show_original: If True, show original with crop overlay; if False, show processed image
        """
        if not self.display_available or self.disp is None:
            print(f"📺 Display not available - Sugar Level: {sugar_level:.2f}")
            return
        
        try:
            # Choose which image to display
            if show_original and crop_coords is not None:
                # Show original image with crop overlay
                display_source = draw_crop_overlay(cv_image, crop_coords)
                image_info = "Original + Crop Overlay"
            elif processed_img is not None:
                # Show the processed/cropped image
                display_source = processed_img
                image_info = "Cropped Image"
            else:
                # Fallback to original image
                display_source = cv_image
                image_info = "Original Image"
            
            # Convert CV2 image (BGR) to PIL image (RGB) - Fix color channel order
            rgb_image = cv2.cvtColor(display_source, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Create display image
            display_img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
            draw = ImageDraw.Draw(display_img)
            
            # Calculate image display area (top 2/3 of screen)
            img_display_height = int(self.height * 0.6)
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
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Add text overlay for sugar level (bottom 1/3 of screen)
            text_y = img_display_height + 5
            
            # Background for text
            draw.rectangle([0, img_display_height, self.width, self.height], fill=(0, 0, 0))
            
            # Sugar level text - use proper green color (corrected for R/B swap)
            sugar_text = f"Sugar: {sugar_level:.2f}"
            text_bbox = draw.textbbox((0, 0), sugar_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (self.width - text_width) // 2
            
            # Green color corrected for R/B swap
            draw.text((text_x, text_y), sugar_text, font=font, fill=(0, 255, 0))
            
            # Image type indicator
            type_text = image_info
            type_bbox = draw.textbbox((0, 0), type_text, font=small_font)
            type_width = type_bbox[2] - type_bbox[0]
            type_x = (self.width - type_width) // 2
            type_y = text_y + 20
            
            draw.text((type_x, type_y), type_text, font=small_font, fill=(255, 255, 255))
            
            # Status text
            status_text = "Analysis Complete"
            status_bbox = draw.textbbox((0, 0), status_text, font=small_font)
            status_width = status_bbox[2] - status_bbox[0]
            status_x = (self.width - status_width) // 2
            status_y = type_y + 15
            
            draw.text((status_x, status_y), status_text, font=small_font, fill=(255, 255, 255))
            
            # Display on screen
            self.disp.display(display_img)
            print(f"📺 {image_info} and prediction displayed on ST7735: {sugar_level:.2f}")
            
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
    parser.add_argument('--single-shot', action='store_true', help='Run once and exit (default is continuous loop)')
    parser.add_argument('--no-crop', action='store_true', help='Disable image cropping')
    parser.add_argument('--show-original', action='store_true', help='Show original image with crop overlay on display')
    parser.add_argument('--crop-region', type=str, help='Crop region as "x1,y1,x2,y2" (percentages 0.0-1.0)')
    args = parser.parse_args()
    
    # Parse custom crop region if provided
    crop_region = CROP_REGION
    if args.crop_region:
        try:
            coords = [float(x.strip()) for x in args.crop_region.split(',')]
            if len(coords) != 4 or any(c < 0.0 or c > 1.0 for c in coords):
                raise ValueError("Invalid coordinates")
            crop_region = tuple(coords)
            print(f"📐 Using custom crop region: {crop_region}")
        except ValueError:
            print("⚠️ Invalid crop region format. Using default.")
            print("   Format: --crop-region 'x1,y1,x2,y2' (e.g., '0.25,0.25,0.75,0.75')")
    
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
        
        # Display crop settings
        if not args.no_crop:
            print(f"📐 Crop region: {crop_region} (x1%, y1%, x2%, y2%)")
            print(f"📺 Display mode: {'Original + Overlay' if args.show_original else 'Cropped Image'}")
        else:
            print("📐 Cropping disabled - using full image")
        
        # Determine if running in continuous loop or single shot
        continuous_mode = not args.single_shot
        if continuous_mode:
            print("🔄 Running in continuous mode (Press Ctrl+C to exit)")
        else:
            print("📸 Running in single-shot mode")
        
        # Main prediction loop
        prediction_count = 0
        while True:
            try:
                prediction_count += 1
                print(f"\n{'='*50}")
                print(f"🎯 PREDICTION #{prediction_count}")
                print(f"{'='*50}")
                
                # Ensure button state is completely reset for each iteration
                button_pressed = False
                
                # Give a moment for any previous operations to complete
                time.sleep(0.2)
                
                # Wait for trigger (button or auto mode)
                if args.auto_mode:
                    if continuous_mode:
                        print("📸 Auto mode: Taking image in 5 seconds... (Press Ctrl+C to stop)")
                        for i in range(5, 0, -1):
                            print(f"{i}...")
                            time.sleep(1)
                    else:
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
                        print(f"🔘 Waiting for button press on GPIO pin {BUTTON_PIN}")
                        if continuous_mode:
                            print("   (Press button for next prediction, Ctrl+C to exit)")
                        else:
                            print("   (Press the physical button to capture image)")
                        
                        # Wait for button press
                        while not button_pressed:
                            time.sleep(0.1)  # Small delay to prevent busy waiting
                        
                        # Clear any buffered frames to get fresh image
                        print("📸 Button pressed! Getting fresh frame...")
                        
                        # Read and discard a few frames to clear buffer
                        for _ in range(3):
                            cap.read()
                        
                        # Reset button state immediately after detection
                        button_pressed = False
                    else:
                        # Fallback for systems without GPIO
                        print("⚠️ Button not available - falling back to keyboard input")
                        if continuous_mode:
                            print("📸 Press ENTER for next prediction (Ctrl+C to exit)...")
                        else:
                            print("📸 Press ENTER to capture image...")
                        input("   Waiting for ENTER key...")
                        print("📸 Capturing image...")
                
                # Capture frame - ensure we get a fresh frame
                print("📸 Capturing fresh image...")
                
                # Clear camera buffer and get the most recent frame
                for i in range(5):  # Read multiple frames to clear buffer
                    ret, frame = cap.read()
                    if not ret:
                        print(f"❌ Error: Could not capture frame {i+1}")
                        break
                    time.sleep(0.02)  # Small delay between reads
                
                if not ret:
                    print("❌ Error: Could not capture image from camera")
                    if not continuous_mode:
                        break
                    print("⚠️ Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                
                print("✅ Fresh frame captured successfully!")
                
                # Save original image if requested
                if args.save_image:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    original_filename = f"original_{timestamp}.jpg"
                    cv2.imwrite(original_filename, frame)
                    print(f"💾 Original image saved as: {original_filename}")
                
                # Predict sugar level with or without cropping
                print("🔍 Analyzing image...")
                use_crop = not args.no_crop
                sugar_level, processed_img, crop_coords = predictor.predict_sugar_level(
                    frame, use_crop=use_crop, crop_region=crop_region
                )
                
                # Save processed image if requested
                if args.save_image and processed_img is not None and use_crop:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    processed_filename = f"cropped_{timestamp}.jpg"
                    cv2.imwrite(processed_filename, processed_img)
                    print(f"💾 Cropped image saved as: {processed_filename}")
                
                # Display results on terminal
                print("\n" + "="*40)
                print("🍯 PREDICTION RESULT")
                print("="*40)
                print(f"Sugar Level: {sugar_level:.2f}")
                print(f"Prediction #{prediction_count}")
                if use_crop and crop_coords:
                    print(f"Crop region: {crop_coords} (pixels)")
                    crop_size = (crop_coords[2] - crop_coords[0], crop_coords[3] - crop_coords[1])
                    print(f"Crop size: {crop_size[0]}x{crop_size[1]} pixels")
                print("="*40)
                
                # Display image and prediction on ST7735 screen
                display.show_image_and_prediction(
                    frame, sugar_level, processed_img, crop_coords, args.show_original
                )
                
                # Keep display on for a few seconds
                if display.display_available:
                    display_time = 5 if continuous_mode else 10
                    print(f"📺 Results displayed on screen for {display_time} seconds...")
                    time.sleep(display_time)
                
                # Exit if single shot mode
                if not continuous_mode:
                    break
                
                # Brief pause between predictions in continuous mode
                if continuous_mode:
                    print("⏳ Ready for next prediction...")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                if continuous_mode:
                    print(f"\n⚠️ Stopping continuous mode after {prediction_count} predictions")
                    break
                else:
                    raise  # Re-raise for single shot mode
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Make sure the model file exists and the path is correct.")
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user after {prediction_count} predictions")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        cleanup_gpio()
        print(f"✅ Done! Completed {prediction_count} predictions.")

if __name__ == "__main__":
    main()

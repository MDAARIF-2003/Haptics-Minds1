import cv2
import numpy as np
import time
import win32com.client # The stable voice library
# --- AI LIBRARIES ---
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. CONFIGURATION ---

# Path to the pre-trained MobileNet-SSD model files
PROTOTXT = r"D:\Vision Beyond Sight\MobileNetSSD_deploy.prototxt"
MODEL = r"D:\Vision Beyond Sight\MobileNetSSD_deploy.caffemodel"

# List of classes MobileNet-SSD was trained to detect
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# --- EXPANDED HUE RANGES (9 COLORS) ---
COLOR_HUE_RANGES = {
    'RED': ((0, 10), (170, 180)), # Red wraps around
    'ORANGE': ((11, 25),),
    'YELLOW': ((26, 35),),
    'GREEN': ((36, 75),),         
    'TEAL': ((76, 95),),          
    'BLUE': ((96, 128),),         
    'PURPLE': ((129, 145),),      
    'MAGENTA': ((146, 160),),     
    'PINK': ((161, 169),),
}

# Detection Confidence Threshold
CONFIDENCE_THRESHOLD = 0.5
ROI_SIZE = 100 # Use a 100x100 region
IMG_SIZE = 64  # The image size our AI model was trained on

# --- 2. INITIALIZATION ---

# Load the Object Detection model
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# --- NEW: Load the custom-trained Pattern AI ---
print("[INFO] Loading custom pattern AI model (pattern_model.h5)...")
pattern_model = load_model(r"D:\Vision Beyond Sight\pattern_model.h5")

# --- THIS MUST MATCH YOUR TRAINING OUTPUT ---
PATTERN_CLASSES = ['checks', 'dots', 'floral', 'plains', 'stripes'] 
print(f"[INFO] Custom pattern AI loaded. Classes: {PATTERN_CLASSES}")

# Camera setup
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] Failed to open camera. Exiting.")
    exit()

# --- NEW VOICE INITIALIZATION ---
print("[INFO] Initializing built-in Windows SAPI voice...")
speaker = win32com.client.Dispatch("SAPI.SpVoice")

# Voice logic variables
last_object_alert_time = 0
last_color_alert_time = 0
SPEECH_COOLDOWN = 3.0 # Speak a new color every 3 seconds
last_spoken_color = ""
last_spoken_pattern = ""

# --- OUTFIT MATCHING VARIABLES ---
saved_color = None
saved_pattern = None


# --- 3. CORE FUNCTIONS ---

# --- FINAL MAX-ACCURACY COLOR FUNCTION (HYBRID) ---
def detect_colors_final_accurate(frame):
    """
    Finds the dominant base color (Histogram) AND its shade (Averaging).
    This is the most accurate hybrid method.
    """
    final_color_name = "Unknown"
    (h, w) = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    try:
        # Define the larger center region
        center_region = frame[center_y - ROI_SIZE//2 : center_y + ROI_SIZE//2, 
                              center_x - ROI_SIZE//2 : center_x + ROI_SIZE//2]
        
        # Convert the ROI to HSV
        hsv_roi = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        
        # --- 1. (HISTOGRAM) Find the dominant base color ---
        color_counts = {}
        color_masks = {} # Store masks for Phase 2

        # Non-Chromatic Colors
        black_mask = cv2.inRange(hsv_roi, (0, 0, 0), (180, 255, 70))
        white_mask = cv2.inRange(hsv_roi, (0, 0, 180), (180, 30, 255))
        gray_mask = cv2.inRange(hsv_roi, (0, 0, 71), (180, 30, 179))
        
        color_counts['BLACK'] = cv2.countNonZero(black_mask)
        color_counts['WHITE'] = cv2.countNonZero(white_mask)
        color_counts['GRAY'] = cv2.countNonZero(gray_mask)
        
        # Store masks
        color_masks['BLACK'] = black_mask
        color_masks['WHITE'] = white_mask
        color_masks['GRAY'] = gray_mask
        
        # Chromatic Colors
        chromatic_mask = cv2.inRange(hsv_roi, (0, 31, 71), (180, 255, 255)) 
        for color_name, hue_ranges in COLOR_HUE_RANGES.items():
            hue_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            for h_range in hue_ranges:
                hue_mask += cv2.inRange(hsv_roi, 
                                        (h_range[0], 50, 50), 
                                        (h_range[1], 255, 255))
            
            final_mask = cv2.bitwise_and(hue_mask, chromatic_mask)
            color_counts[color_name] = cv2.countNonZero(final_mask)
            color_masks[color_name] = final_mask # Store for shade analysis

        # Find the dominant base color
        dominant_base_color = max(color_counts, key=color_counts.get)
        
        # --- 2. (SHADE ANALYSIS) Find the shade of the dominant color ---
        
        if dominant_base_color in ['BLACK', 'WHITE', 'GRAY']:
            final_color_name = dominant_base_color
        else:
            dominant_mask = color_masks[dominant_base_color]
            avg_v = cv2.mean(hsv_roi, mask=dominant_mask)[2]
            
            if avg_v > 200:
                final_color_name = f"LIGHT {dominant_base_color}"
            elif avg_v < 90:
                final_color_name = f"DARK {dominant_base_color}"
            else:
                final_color_name = dominant_base_color

    except Exception as e:
        # print(f"[WARN] Color Hybrid error: {e}") 
        final_color_name = None

    return final_color_name, center_x, center_y, ROI_SIZE

# --- NEW: "ULTIMATE HYBRID" AI PATTERN DETECTION FUNCTION ---
def detect_pattern_ultimate_hybrid(frame):
    """
    Uses a 3-step hybrid approach for maximum accuracy.
    1. Fast Edge Density to find "PLAIN".
    2. Fast Line-Finder (Hough) to find "CHECKS/GRID".
    3. AI (CNN) to find "STRIPES", "DOTS", "FLORAL".
    """
    (h, w) = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    try:
        # Extract the ROI (must be the same size as before)
        roi = frame[center_y - ROI_SIZE//2 : center_y + ROI_SIZE//2, 
                    center_x - ROI_SIZE//2 : center_x + ROI_SIZE//2]
        
        # --- Step 1: Fast Edge Density Check (for PLAINS) ---
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        edges = cv2.Canny(gray_roi, 30, 100) 
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # If density is very low, it's PLAIN. Don't use the AI.
        if edge_density < 0.05:
            return "PLAINS"
            
        # --- Step 2: Fast Line-Finder Check (for CHECKS) ---
        # This logic is very good at finding grids.
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
                
                # Check for horizontal lines (within 20 degrees)
                if abs(angle_deg) < 20 or abs(angle_deg - 180) < 20:
                    horizontal_lines += 1
                # Check for vertical lines (within 20 degrees of 90)
                elif abs(angle_deg - 90) < 20 or abs(angle_deg + 90) < 20:
                    vertical_lines += 1

            # If we find a strong grid, we are very confident it's CHECKS
            if horizontal_lines > 3 and vertical_lines > 3:
                return "CHECKS/GRID"
            
        # --- Step 3: It's not PLAIN or CHECKS. Use the AI (CNN) ---
        
        # Preprocess the image for the AI
        image = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        image = image.astype("float") / 255.0
        image = np.expand_dims(image, axis=0) 
        
        # Make a Prediction
        predictions = pattern_model.predict(image, verbose=0)
        
        # Get the winning class
        class_id = np.argmax(predictions[0])
        
        # Look up the class name
        pattern_name = PATTERN_CLASSES[class_id]
        
        # --- Final Sanity Check ---
        # If the AI *still* guesses "PLAINS" or "CHECKS", but our rules
        # didn't catch it, we'll call it a generic "PATTERNED".
        if pattern_name == "plains":
            return "PATTERNED" # It has edges, so it's not plain.
        if pattern_name == "checks":
            return "PATTERNED" # Our line-finder didn't find a grid.
            
        return pattern_name.upper() # Return "STRIPES", "DOTS", "FLORAL"

    except Exception as e:
        # print(f"[WARN] CNN Pattern detection error: {e}")
        return "Unknown"

# --- OUTFIT MATCHING LOGIC FUNCTION ---
def get_match_recommendation(color1, pattern1, color2, pattern2):
    """
    Provides a simple fashion recommendation.
    """
    # Define simple rules
    neutrals = ['BLACK', 'WHITE', 'GRAY']
    # We now use our new AI classes
    complex_patterns = ['CHECKS/GRID', 'DOTS', 'FLORAL', 'PATTERNED'] 
    simple_patterns = ['PLAINS', 'STRIPES']
    
    # Rule 1: Neutrals match with anything
    if (color1 in neutrals or color2 in neutrals) and (color1 != color2):
        return "Good Match (Neutral Base)"
        
    # Rule 2: Don't mix two complex patterns
    if pattern1 in complex_patterns and pattern2 in complex_patterns:
        return "Pattern Clash!"
        
    # Rule 3: Match a pattern with a plain item
    if (pattern1 == 'PLAINS' and pattern2 != 'PLAINS') or (pattern1 != 'PLAINS' and pattern2 == 'PLAINS'):
        return "Good Match (Pattern + Plain)"
        
    # Rule 4: Match two *different* simple patterns (e.g., PLAINS + STRIPES)
    if pattern1 in simple_patterns and pattern2 in simple_patterns and pattern1 != pattern2:
         return "Good Match (Simple Patterns)"

    # Rule 5: Analogous colors (colors near each other)
    if color1 == color2 and pattern1 == pattern2:
        return "Items are identical"
    if color1 == color2:
        return "Good Match (Monochromatic)"

    # Default
    return "Okay Match"

# --- 4. MAIN LOOP ---

print("[INFO] Starting video stream...")

# This "if __name__ == '__main__':" is CRITICAL
if __name__ == '__main__':

    # --- ADDED: Camera Warm-up ---
    print("[INFO] Giving camera 2 seconds to warm up...")
    time.sleep(2.0)
    print("[INFO] Camera ready.")

    while True:
        ret, frame = cam.read()
        
        # --- THIS IS THE CRITICAL CHECK ---
        if not ret:
            print("[ERROR] Cannot receive frame (stream end?). Exiting.")
            break
        
        (h, w) = frame.shape[:2]
        current_time = time.time() # Get the time once per frame

        # --- Object Detection ---
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843, 
            size=(300, 300),
            mean=127.5 # This is the more common mean for public MobileNetSSD
        )
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                if label == "person":
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # --- STABLE VOICE ALERT FOR OBJECT ---
                    if current_time - last_object_alert_time > 5.0:
                        speaker.Speak("Warning, person detected directly ahead", 1) 
                        last_object_alert_time = current_time
                    break 

        # --- Color and Pattern Detection ---
        
        # Call the NEW ACCURATE functions
        detected_color, cx, cy, region_s = detect_colors_final_accurate(frame) # <-- Hybrid Color
        detected_pattern = detect_pattern_ultimate_hybrid(frame) # <-- ULTIMATE HYBRID AI
        
        # --- Check for Key Presses ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # SAVE the current item
            if detected_color and detected_pattern:
                saved_color = detected_color
                saved_pattern = detected_pattern
                print(f"[INFO] SAVED ITEM: {saved_color}, {saved_pattern}")
                speaker.Speak(f"Saved {saved_color} {saved_pattern}", 1)
            
        if key == ord('c'):
            # CLEAR the saved item
            saved_color = None
            saved_pattern = None
            print("[INFO] Cleared saved item.")
            speaker.Speak("Matcher cleared", 1)

        if key == ord('q'):
            # QUIT the program
            break
        
        if detected_color and detected_pattern:
            # Draw ROI
            cv2.rectangle(frame, (cx - ROI_SIZE//2, cy - ROI_SIZE//2), (cx + ROI_SIZE//2, cy + ROI_SIZE//2), (0, 255, 255), 2) # Yellow ROI Box
            
            color_text = f"Color: {detected_color}"
            pattern_text = f"Pattern: {detected_pattern}"
            
            # --- STABLE VOICE LOGIC ---
            is_new_detection = (detected_color != last_spoken_color) or (detected_pattern != last_spoken_pattern)
            is_time_to_speak = (current_time - last_color_alert_time) > SPEECH_COOLDOWN

            if is_new_detection and is_time_to_speak and saved_color is None:
                # Only speak the detection if we are NOT in matching mode
                full_speech = f"Color: {detected_color}. Pattern: {detected_pattern}"
                speaker.Speak(full_speech, 1) 
                
                last_spoken_color = detected_color
                last_spoken_pattern = detected_pattern
                last_color_alert_time = current_time
                
            # --- TEXT DISPLAY ---
            
            # Color Text
            cv2.rectangle(frame, (10, 10), (250, 40), (0,0,0), -1) # Black box
            cv2.putText(frame, color_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text
            
            # Pattern Text
            cv2.rectangle(frame, (10, 50), (250, 80), (0,0,0), -1) # Black box
            cv2.putText(frame, pattern_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text
        
        
        # --- OUTFIT MATCHER DISPLAY ---
        if saved_color:
            # Display the saved item
            saved_text = f"Saved: {saved_color}, {saved_pattern}"
            cv2.putText(frame, saved_text, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3) # Black outline
            cv2.putText(frame, saved_text, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text
            
            # Get and display the recommendation
            if detected_color and detected_pattern:
                recommendation = get_match_recommendation(saved_color, saved_pattern, detected_color, detected_pattern)
                cv2.putText(frame, f"Match: {recommendation}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frame, f"Match: {recommendation}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # White text
        else:
            # Show instructions
            cv2.putText(frame, "Point at item and press 's' to save for matching", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, "Point at item and press 's' to save for matching", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                
        # --- Display ---
        cv2.imshow("Vision Beyond Sight - Competition Demo", frame)

    # --- 5. CLEANUP ---
    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Program ended gracefully.")
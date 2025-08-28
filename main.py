import cv2
import numpy as np
import os

def preprocess_image(img):
    """Preprocess the image for better plate detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def detect_plate_regions(frame):
    """Detect potential license plate regions in a frame"""
    # Preprocess the frame
    processed = preprocess_image(frame)
    
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and keep the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contours = []
    
    # Find rectangular contours that could be license plates
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        if len(approx) == 4:  # If the contour has 4 vertices
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Check for typical license plate aspect ratio (adjust as needed)
            if 2.0 < aspect_ratio < 5.0:
                plate_contours.append(approx)
    
    return plate_contours

def extract_characters(plate_img):
    """Extract individual characters from a license plate image"""
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort characters left-to-right
    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > 100:  # Filter small noise
            chars.append((x, y, w, h))
    
    # Sort characters by x-coordinate (left to right)
    chars = sorted(chars, key=lambda c: c[0])
    
    # Extract character images
    character_images = []
    for x, y, w, h in chars:
        char_img = thresh[y:y+h, x:x+w]
        # Resize to consistent size for recognition
        char_img = cv2.resize(char_img, (20, 40))
        character_images.append(char_img)
    
    return character_images

def recognize_characters(char_imgs, templates):
    """Recognize characters using template matching"""
    recognized_text = ""
    
    for char_img in char_imgs:
        best_match = None
        best_score = -1
        
        for char, template in templates.items():
            # Resize template to match character size
            resized_template = cv2.resize(template, (char_img.shape[1], char_img.shape[0]))
            
            # Perform template matching
            result = cv2.matchTemplate(char_img, resized_template, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            
            if score > best_score:
                best_score = score
                best_match = char
        
        # Only accept matches with reasonable confidence
        if best_score > 0.5:
            recognized_text += best_match
        else:
            recognized_text += "?"
    
    return recognized_text

def load_templates(template_dir):
    """Load character templates for recognition"""
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            char = filename.split(".")[0].upper()
            template = cv2.imread(os.path.join(template_dir, filename), cv2.IMREAD_GRAYSCALE)
            _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
            templates[char] = template
    return templates

def process_video(video_path, template_dir="templates", output_path=None):
    """Process a video file for license plate recognition"""
    # Load character templates
    templates = load_templates(template_dir)
    
    cap = cv2.VideoCapture(video_path)
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                             (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect license plate regions
        plate_contours = detect_plate_regions(frame)
        
        for contour in plate_contours:
            # Draw rectangle around plate
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            
            # Extract plate image
            x, y, w, h = cv2.boundingRect(contour)
            plate_img = frame[y:y+h, x:x+w]
            
            # Extract and recognize characters
            char_imgs = extract_characters(plate_img)
            if char_imgs:
                plate_text = recognize_characters(char_imgs, templates)
                
                # Display recognized text
                if plate_text:
                    cv2.putText(frame, plate_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if output_path:
            out.write(frame)
        
        cv2.imshow('License Plate Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_file = "car_video.mp4"  # Replace with your video file path
    output_file = "output_video.avi"  # Optional: to save the processed video
    template_directory = "templates"  # Directory containing character templates
    
    # Create template directory if it doesn't exist
    if not os.path.exists(template_directory):
        os.makedirs(template_directory)
        print(f"Created '{template_directory}' directory. Please add character templates (A.png, B.png, 1.png, etc.)")
    
    process_video(video_file, template_directory, output_file)
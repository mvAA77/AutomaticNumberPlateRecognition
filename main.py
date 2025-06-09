import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
print("Tesseract version:", pytesseract.get_tesseract_version())


#pip install opencv-python easyocr ultralytics matplotlib
#pip install ultralytics
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

import cv2

cap = cv2.VideoCapture('car-13.png')
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Process frame
    results = model(frame)  # Detection happens first
    
    # Visualization
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        if confidence > 0.5:
            plate_crop = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Plate", plate_crop)
    
    # Display frame
    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    import easyocr
    reader = easyocr.Reader(['en'])

    results = reader.readtext(plate_crop)
    for (bbox, text, prob) in results:
        if prob > 0.4:  # Confidence filter
            print("Detected Plate Number:", text)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 0, 0), 2)
    
    cv2.imshow("Annotated", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup (outside loop)
cap.release()
cv2.destroyAllWindows()

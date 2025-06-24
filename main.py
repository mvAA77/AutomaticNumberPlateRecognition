import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Users/tanya/OneDrive/Documents/tesseract-ocr-w64-setup-5.5.0.20241111.exe'
from PIL import Image
print("Tesseract version:", pytesseract.get_tesseract_version())


#pip install opencv-python easyocr ultralytics matplotlib
#pip install ultralytics
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

import cv2

cap = cv2.VideoCapture('mvAA77/AutomaticNumberPlateRecognition/IMG_0511.MOV')
if not cap.isOpened():
    print("Error opening video file")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        if confidence > 0.5:
            plate_crop = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            results_text = reader.readtext(plate_crop)

            for (bbox, text, prob) in results_text:
                if prob > 0.4:
                    print("Detected Plate Number:", text)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 0, 0), 2)

    cv2.imshow("Annotated", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
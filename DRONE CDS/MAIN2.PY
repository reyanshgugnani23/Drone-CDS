import cv2
import requests
import time
from ultralytics import YOLO

TELEGRAM_BOT_TOKEN = "8719196524:AAE9vt3lZBVWR1W-fYGRgR3yU7I6MGwcFOE"
TELEGRAM_CHAT_ID = "6759622745"

# Load pre-trained pothole YOLO model (downloadable from Roboflow/GitHub)
model = YOLO("pothole_v8.pt") 

cap = cv2.VideoCapture("road_video.mp4")
last_alert_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.4)[0]
    pothole_count = len(results.boxes)

    # Render detection boxes
    annotated_frame = results.plot()

    if pothole_count > 0 and (time.time() - last_alert_time > 10):
        cv2.imwrite("pothole_alert.jpg", annotated_frame)
        
        # Telegram API Call
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open("pothole_alert.jpg", "rb") as img:
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"⚠️ Potholes Detected: {pothole_count}"}, files={"photo": img})
        
        last_alert_time = time.time()

    cv2.imshow("YOLO Pothole Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()                 
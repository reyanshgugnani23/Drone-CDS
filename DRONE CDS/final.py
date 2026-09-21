import cv2
import numpy as np
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials from Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("[Error] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment variables!")

def send_telegram_alert(image_path, caption):
    if not os.path.exists(image_path):
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as image_file:
            files = {"photo": ("pothole.jpg", image_file, "image/jpeg")}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            response = requests.post(url, data=data, files=files, timeout=10)
            return response.status_code == 200 and response.json().get("ok")
    except Exception as e:
        print(f"[Error] Telegram failed: {e}")
        return False

def get_road_roi_mask(frame):
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    roi_corners = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.35), int(height * 0.15)),
        (int(width * 0.65), int(height * 0.15)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_corners, 255)
    return mask

def detect_potholes_edge_based(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_mask = get_road_roi_mask(frame)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    masked_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(masked_edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pothole_count = 0
    pothole_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            if 0.3 < aspect_ratio < 3.0:
                pothole_count += 1
                pothole_detected = True

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Pothole", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, closed, pothole_detected, pothole_count

def main():
    video_source = "road_video.mp4" 

    if isinstance(video_source, str) and not os.path.exists(video_source):
        print(f"[Warning] File '{video_source}' not found. Falling back to webcam (0)...")
        video_source = 0

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {video_source}")
        return

    cooldown_time = 8
    last_alert_time = 0

    print("Pipeline started successfully. Press 'q' on the preview window to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[Info] Stream ended or frame read failed.")
            break

        processed_frame, debug_frame, detected, count = detect_potholes_edge_based(frame.copy())

        cv2.putText(processed_frame, f"Potholes Detected: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        current_time = time.time()
        if detected and (current_time - last_alert_time > cooldown_time):
            snapshot_path = "pothole_alert.jpg"
            cv2.imwrite(snapshot_path, processed_frame)
            send_telegram_alert(snapshot_path, f"⚠️ Alert: {count} pothole(s) detected!")
            last_alert_time = current_time

        cv2.imshow("Pothole Visual Feed", processed_frame)
        cv2.imshow("Pothole Edge Mask (Debug)", debug_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
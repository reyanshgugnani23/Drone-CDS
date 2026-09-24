
#ye wala pakka final hai

import cv2
import numpy as np
import requests
import time
import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(image_path, caption):
    """Sends an image to Telegram with robust error reporting."""
    if not os.path.exists(image_path):
        print(f"[Error] Image path does not exist: {image_path}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as image_file:
            files = {"photo": ("pothole.jpg", image_file, "image/jpeg")}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            
            response = requests.post(url, data=data, files=files, timeout=10)
            res_json = response.json()
            
            if response.status_code == 200 and res_json.get("ok"):
                print("[Telegram API] Photo sent successfully!")
                return True
            else:
                print(f"[Telegram Error] API returned code {response.status_code}: {res_json}")
                return False

    except Exception as e:
        print(f"[Network Error] Failed to send photo: {e}")
        return False

def remove_road_markings(hsv_frame):
    """
    Creates a mask targeting white and yellow road paint so they can be subtracted
    from pothole detection.
    """
    # White paint range
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Yellow paint range
    lower_yellow = np.array([15, 80, 140])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Combine paint masks
    paint_mask = cv2.bitwise_or(mask_white, mask_yellow)
    return paint_mask

def detect_potholes(frame):
    """
    Filters out road markings and detects true dark depressions (potholes).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Generate mask for white and yellow road lines
    paint_mask = remove_road_markings(hsv)

    # 2. Blur gray image to smooth road texture
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Adaptive thresholding to find dark regions (shadows/depressions)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 19, 4
    )

    # 4. Subtract paint regions from thresholded map
    thresh_no_paint = cv2.bitwise_and(thresh, cv2.bitwise_not(paint_mask))

    # 5. Clean up noise with morphological closing/opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh_no_paint, cv2.MORPH_CLOSE, kernel)

    # 6. Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pothole_count = 0
    pothole_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Size check for candidate potholes
        if 800 < area < 35000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # Road lines are often very tall/thin or very long/flat
            if 0.4 < aspect_ratio < 2.2:
                # 7. Brightness Verification (BRIGHTNESS CHECK)
                # Extract the region of interest (ROI) from grayscale image
                roi_gray = gray[y:y+h, x:x+w]
                mean_brightness = np.mean(roi_gray)

                # Road markings are bright (>130 intensity).
                # True potholes and shadows are darker (<110 intensity).
                if mean_brightness < 110:
                    # Fill ratio check: excludes perfectly rectangular geometry (lane blocks)
                    extent = float(area) / (w * h)
                    if extent < 0.85:  # Potholes have irregular shapes
                        pothole_count += 1
                        pothole_detected = True

                        # Draw box around true pothole
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"Pothole ({int(mean_brightness)})", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, cleaned, pothole_detected, pothole_count

def main():
    # Set to 0 for camera feed or path to video, e.g., "road.mp4"
    video_source = 2 
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {video_source}")
        return

    cooldown_time = 8  # Delay in seconds between Telegram notifications
    last_alert_time = 0

    print("Starting stream... Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[Info] End of video stream.")
            break

        processed_frame, thresh_frame, detected, count = detect_potholes(frame.copy())

        # Dashboard overlay
        cv2.putText(processed_frame, f"Potholes Detected: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Trigger alert
        current_time = time.time()
        if detected and (current_time - last_alert_time > cooldown_time):
            snapshot_path = "pothole_alert.jpg"
            cv2.imwrite(snapshot_path, processed_frame)
            
            print(f"[Alert] Sending notification for {count} detected pothole(s)...")
            send_telegram_alert(snapshot_path, f"⚠️ Alert: {count} pothole(s) detected on the road!")
            last_alert_time = current_time

        # Preview Windows
        cv2.imshow("Pothole Detection - Main Feed", processed_frame)
        cv2.imshow("Pothole Detection - Filtered Binary Feed", thresh_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

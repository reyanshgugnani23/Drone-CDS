import cv2
import numpy as np
import requests
import time
import os

TELEGRAM_BOT_TOKEN = "8719196524:AAE9vt3lZBVWR1W-fYGRgR3yU7I6MGwcFOE"
TELEGRAM_CHAT_ID = "6759622745"

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
    """Mask out everything outside the main road lanes (e.g. upper corners/sides)."""
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Trapezoid matching the road perspective
    roi_corners = np.array([[
        (int(width * 0.05), height),             # Bottom Left
        (int(width * 0.35), int(height * 0.15)), # Top Left
        (int(width * 0.65), int(height * 0.15)), # Top Right
        (int(width * 0.95), height)              # Bottom Right
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_corners, 255)
    return mask

def detect_potholes_edge_based(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_mask = get_road_roi_mask(frame)

    # 1. Edge detection to catch high-texture broken asphalt/gravel
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # 2. Restrict detection strictly to the road surface
    masked_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    # 3. Morphological dilation to close gaps between gravel pieces inside the pothole
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(masked_edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # 4. Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pothole_count = 0
    pothole_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Dynamic size check based on area (relaxes restrictions for distant/small potholes)
        if area > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # Exclude long straight lines (lane markings/guardrails)
            if 0.3 < aspect_ratio < 3.0:
                pothole_count += 1
                pothole_detected = True

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Pothole", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, closed, pothole_detected, pothole_count

def main():
    video_source = "road_video.mp4" # Or 0 for camera
    cap = cv2.VideoCapture(video_source)

    cooldown_time = 8
    last_alert_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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
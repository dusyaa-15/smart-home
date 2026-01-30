import cv2
import time
import os
from ultralytics import YOLO
from pathlib import Path

# ============================
# ESP32 STREAM CONFIG
# ============================
ESP32_STREAM_URL = "http://192.168.4.1:81/stream"
YOLO_SIZE = 416

# ============================
# DETECTION PARAMETERS
# ============================
PERSIST_TIME = 0.7
PERSON_CONF  = 0.35
FIRE_CONF    = 0.45
# ============================

# Increase FFmpeg tolerance for ESP32 MJPEG
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;60000"

def main():
    root = Path(__file__).resolve().parents[1]
    fire_model_path = root / "models" / "fire_smoke_best.pt"

    if not fire_model_path.exists():
        print("❌ Fire model not found:", fire_model_path)
        return

    print("⏳ Loading models...")
    person_model = YOLO("yolov8n.pt")
    fire_model   = YOLO(str(fire_model_path))
    print("✅ Models loaded")

    # ============================
    # Open ESP32 MJPEG stream
    # ============================
    cap = cv2.VideoCapture(ESP32_STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Cannot open ESP32 stream")
        return

    print("✅ ESP32 stream connected")

    # ============================
    # Detection memory
    # ============================
    last_person_seen = 0.0
    last_fire_seen   = 0.0
    last_frame_time  = time.time()

    while True:
        loop_start = time.time()

        # --------------------------------
        # DROP OLD FRAMES (CRITICAL)
        # --------------------------------
        for _ in range(3):
            cap.grab()

        ret, frame = cap.retrieve()

        if not ret or frame is None:
            # Reconnect if stream stalls
            if time.time() - last_frame_time > 5:
                print("⚠️ Stream stalled, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(ESP32_STREAM_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                last_frame_time = time.time()
            continue

        last_frame_time = time.time()

        h, w = frame.shape[:2]
        output = frame.copy()

        # =========================
        # RESIZE FOR YOLO
        # =========================
        yolo_frame = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))

        # =========================
        # YOLO INFERENCE (EVERY FRAME)
        # =========================
        r_person = person_model.predict(
            yolo_frame,
            conf=PERSON_CONF,
            classes=[0],
            verbose=False
        )[0]

        r_fire = fire_model.predict(
            yolo_frame,
            conf=FIRE_CONF,
            verbose=False
        )[0]

        now = time.time()

        if r_person.boxes is not None and len(r_person.boxes) > 0:
            last_person_seen = now

        if r_fire.boxes is not None and len(r_fire.boxes) > 0:
            last_fire_seen = now

        person_present = (now - last_person_seen) < PERSIST_TIME
        fire_present   = (now - last_fire_seen)   < PERSIST_TIME

        sx = w / YOLO_SIZE
        sy = h / YOLO_SIZE

        # =========================
        # DRAW PERSON (STABLE)
        # =========================
        if person_present and r_person.boxes is not None:
            for box in r_person.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x1, x2 = int(x1 * sx), int(x2 * sx)
                y1, y2 = int(y1 * sy), int(y2 * sy)

                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    output, f"Person {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2
                )

        # =========================
        # DRAW FIRE / SMOKE (STABLE)
        # =========================
        if fire_present and r_fire.boxes is not None:
            for box in r_fire.boxes:
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                label_name = fire_model.names.get(cls, "fire")

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = int(x1 * sx), int(x2 * sx)
                y1, y2 = int(y1 * sy), int(y2 * sy)

                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    output, f"{label_name.capitalize()} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

        # =========================
        # FPS
        # =========================
        fps = 1.0 / (time.time() - loop_start + 1e-6)
        cv2.putText(
            output, f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.imshow("ESP32 – Stream Stable Detection", output)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

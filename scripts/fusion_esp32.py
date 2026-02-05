import cv2
import time
import os
import torch
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
COCO_CONF = 0.35
FIRE_CONF = 0.45
# ============================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;60000"


def main():
    root = Path(__file__).resolve().parents[1]
    fire_model_path = root / "models" / "fire_smoke_best.pt"

    if not fire_model_path.exists():
        print("❌ Fire model not found:", fire_model_path)
        return

    # =====================================
    # CHECK GPU
    # =====================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {DEVICE}")

    print("⏳ Loading models...")

    coco_model = YOLO("yolov8n.pt")
    fire_model = YOLO(str(fire_model_path))

    # ---- MOVE TO GPU ----
    coco_model.to(DEVICE)
    fire_model.to(DEVICE)

    print("✅ Models loaded on", DEVICE)

    # ============================
    # Open ESP32 MJPEG stream
    # ============================
    cap = cv2.VideoCapture(ESP32_STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Cannot open ESP32 stream")
        return

    print("✅ ESP32 stream connected")

    last_coco_seen = 0.0
    last_fire_seen = 0.0
    last_frame_time = time.time()

    while True:
        loop_start = time.time()

        # Drop old frames
        for _ in range(3):
            cap.grab()

        ret, frame = cap.retrieve()

        if not ret or frame is None:
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

        # Resize for YOLO
        yolo_frame = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))

        # =========================
        # GPU INFERENCE
        # =========================
        r_coco = coco_model.predict(
            yolo_frame,
            conf=COCO_CONF,
            verbose=False,
            device=DEVICE
        )[0]

        r_fire = fire_model.predict(
            yolo_frame,
            conf=FIRE_CONF,
            verbose=False,
            device=DEVICE
        )[0]

        now = time.time()

        if r_coco.boxes is not None and len(r_coco.boxes) > 0:
            last_coco_seen = now

        if r_fire.boxes is not None and len(r_fire.boxes) > 0:
            last_fire_seen = now

        coco_present = (now - last_coco_seen) < PERSIST_TIME
        fire_present = (now - last_fire_seen) < PERSIST_TIME

        sx = w / YOLO_SIZE
        sy = h / YOLO_SIZE

        # =========================
        # DRAW COCO OBJECTS - GREEN
        # =========================
        if coco_present and r_coco.boxes is not None:
            for box in r_coco.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = coco_model.names.get(cls, "obj")

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = int(x1 * sx), int(x2 * sx)
                y1, y2 = int(y1 * sy), int(y2 * sy)

                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    output, f"{label} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2
                )

        # =========================
        # DRAW FIRE - RED
        # =========================
        if fire_present and r_fire.boxes is not None:
            for box in r_fire.boxes:
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = int(x1 * sx), int(x2 * sx)
                y1, y2 = int(y1 * sy), int(y2 * sy)

                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    output, f"FIRE {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )

        # =========================
        # FPS
        # =========================
        fps = 1.0 / (time.time() - loop_start + 1e-6)
        cv2.putText(
            output, f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 0), 2
        )

        cv2.imshow("ESP32 – Fire + COCO Detection [GPU]", output)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

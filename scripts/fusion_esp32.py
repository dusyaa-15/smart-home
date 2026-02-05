import cv2
import time
import torch
from ultralytics import YOLO
from threading import Thread, Lock

# =========================
# CONFIG
# =========================
ESP32_STREAM_URL = "http://192.168.4.1:81/stream"
YOLO_SIZE = 320
DETECT_EVERY_N_FRAMES = 3
PERSON_CONF = 0.35

# ============================
# DETECTION PARAMETERS
# ============================
PERSIST_TIME = 0.7
COCO_CONF = 0.35
FIRE_CONF = 0.45
# ============================

# =========================
# CAMERA THREAD (LATEST FRAME ONLY)
# =========================
class Camera:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = Lock()
        self.running = True
        Thread(target=self._reader, daemon=True).start()

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print("⏳ Loading models...")
    coco_model = YOLO("yolov8n.pt")       # All COCO classes
    fire_model = YOLO(str(fire_model_path))
    print("✅ Models loaded")

    model = YOLO("yolov8n.pt").to("cuda")

    cam = Camera(ESP32_STREAM_URL)
    time.sleep(2)

    # GPU warm-up (IMPORTANT)
    dummy = torch.zeros((1, 3, YOLO_SIZE, YOLO_SIZE), device="cuda")
    model.predict(dummy, verbose=False)

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

        h, w = frame.shape[:2]
        frame_count += 1

        # =========================
        # YOLO INFERENCE
        # =========================
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            yolo_frame = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
            results = model.predict(
                yolo_frame,
                imgsz=YOLO_SIZE,
                conf=PERSON_CONF,
                classes=[0],
                device="cuda",
                verbose=False
            )[0]
            last_boxes = results.boxes

        # =========================
        # DRAW (NATURAL LOOK)
        # =========================
        output = frame.copy()

        # Resize for YOLO
        yolo_frame = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))

        # =========================
        # INFERENCE
        # =========================
        r_coco = coco_model.predict(
            yolo_frame,
            conf=COCO_CONF,
            verbose=False
        )[0]

        r_fire = fire_model.predict(
            yolo_frame,
            conf=FIRE_CONF,
            verbose=False
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
            output,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 0), 2
        )

        cv2.imshow("ESP32 – Fire + COCO Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

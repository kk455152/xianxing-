import os
import sys
import time

import cv2


def main():
    if len(sys.argv) < 5:
        raise SystemExit("usage: opencv_camera_stream.py camera_index frame_path stop_path log_path [backend]")

    camera_index = int(sys.argv[1])
    frame_path = sys.argv[2]
    stop_path = sys.argv[3]
    log_path = sys.argv[4]
    backend_name = sys.argv[5].upper() if len(sys.argv) >= 6 else "DSHOW"

    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(message):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")

    backends = {
        "DSHOW": cv2.CAP_DSHOW,
        "MSMF": cv2.CAP_MSMF,
        "ANY": 0,
    }
    backend = backends.get(backend_name, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(camera_index, backend) if backend else cv2.VideoCapture(camera_index)
    if not cap.isOpened() and backend_name != "ANY":
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        log(f"cannot open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    log(f"camera {camera_index} started backend={backend_name}")

    temp_path = frame_path + ".tmp.jpg"
    try:
        while not os.path.exists(stop_path):
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            ok = cv2.imwrite(temp_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                os.replace(temp_path, frame_path)
            time.sleep(0.04)
    finally:
        cap.release()
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        log("camera stopped")


if __name__ == "__main__":
    main()

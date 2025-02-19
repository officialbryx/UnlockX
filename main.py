import threading
import cv2
import os
import time
from deepface import DeepFace

# Initialize webcam with lower resolution
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

face_match = False
lock = threading.Lock()  # Prevents race conditions
running = True  # Flag for stopping thread

# Load reference images as grayscale (lighter)
reference_dir = "reference/TIAMZON"
reference_images = [
    cv2.imread(os.path.join(reference_dir, file), cv2.IMREAD_GRAYSCALE)
    for file in os.listdir(reference_dir) if file.endswith(".jpg")
]

def check_face(img):
    global face_match
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        for ref_img in reference_images:
            result = DeepFace.verify(img_gray, ref_img, detector_backend="opencv")["verified"]
            if result:
                with lock:
                    face_match = True
                return
        with lock:
            face_match = False
    except ValueError:
        with lock:
            face_match = False

# Background thread for face processing
def face_processing():
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        check_face(frame.copy())  # Process frame
        time.sleep(1.5)  # Reduce processing frequency

# Start background thread
threading.Thread(target=face_processing, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display match result
    with lock:
        text = "Match!" if face_match else "No Match!"
        color = (0, 255, 0) if face_match else (0, 0, 255)
        cv2.putText(frame, text, (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("video", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):  # Lower refresh rate
        break

running = False  # Stop background thread
cap.release()
cv2.destroyAllWindows()

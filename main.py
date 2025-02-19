import threading
import cv2
import os
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
lock = threading.Lock()  # Prevents race conditions

# Load all reference images from the directory
reference_dir = "reference/TIAMZON"
reference_images = [cv2.imread(os.path.join(reference_dir, file)) 
                    for file in os.listdir(reference_dir) if file.endswith(".jpg")]

def check_face(img):
    global face_match
    try:
        for ref_img in reference_images:
            result = DeepFace.verify(img, ref_img.copy())['verified']
            if result:  # If any match is found, set face_match = True
                with lock:
                    face_match = True
                return  
        # If no match is found, set face_match = False
        with lock:
            face_match = False
    except ValueError:
        with lock:
            face_match = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if counter % 30 == 0:
        threading.Thread(target=check_face, args=(frame.copy(),), daemon=True).start()

    counter += 1

    # Display match result
    with lock:
        if face_match:
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# UnlockX

A lightweight desktop application for biometric face registration and facial authentication, built with **PyQt5** and **DeepFace**.

---

## Features

* **User Registration Pipeline:** Collects multi-pose reference samples (*Front View*, *Left Side*, *Right Side*, *Upward*, *Downward*) stored per user directory.
* **Background Face Verification:** Multi-threaded verification loop using DeepFace (`VGG-Face` model) for non-blocking UI responsiveness.
* **PyQt5 Interface:** Clean multi-page interface powered by `QStackedWidget` for navigation between landing, registration, and login screens.
* **Real-time Camera Stream:** OpenCV video capture integrated into Qt UI frames via `QTimer`.

---

## Directory Structure

```text
.
├── logo/
│   └── unlockx.png         # Application logo icon
├── reference/              # Registered user face images
│   └── [LAST_NAME]/
│       └── [LAST_NAME]_[POSE]_Face.png
└── main.py                 # Core application script
```

## Getting Started
* Prerequisites
* Ensure Python 3.8+ is installed, then install the required dependencies:
```
pip install pyqt5 opencv-python deepface numpy
```

## Run the app
* Execute the main application script from the root directory:
```
python main.py
```

## Usage Workflow

1. Launch App: Start the application to access the main dashboard.

2. Register User:

      Click Register New User.

      Enter the user's First Name and Last Name, then click Save Name.

      Follow the prompt on screen to capture 5 facial poses (Front View, Left Side, Right Side, Upward, Downward).

      Once captured, the system creates a dedicated folder in reference/ and saves the reference photos.

3. Login with Face ID:

      Click Login with Face ID.

      Position yourself in front of the camera. The app runs a background verification thread matching your live video against images in reference/.

      Upon successful match, your greeting message will appear on screen.

import sys
import cv2
import os
from PyQt5.QtWidgets import (
    QApplication, QStackedWidget, QWidget, QVBoxLayout, 
    QPushButton, QLabel, QLineEdit, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from deepface import DeepFace
import numpy as np
import time
import threading

REFERENCE_DIR = "reference"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.label = QLabel("UnlockX", alignment=Qt.AlignCenter)
        self.register_button = QPushButton("Register")
        self.login_button = QPushButton("Login")

        layout.addWidget(self.label)
        layout.addWidget(self.register_button)
        layout.addWidget(self.login_button)
        layout.setAlignment(Qt.AlignCenter)

        self.setLayout(layout)

class RegisterPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.camera = None  
        self.pose_index = 0
        self.poses = ["Front View", "Left Side", "Right Side", "Upward", "Downward"]
        self.user_last_name = ""

        layout = QVBoxLayout()

        # Name Input Fields
        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText("Enter First Name")
        self.last_name_input = QLineEdit()
        self.last_name_input.setPlaceholderText("Enter Last Name")
        self.save_name_button = QPushButton("Save Name")
        self.save_name_button.clicked.connect(self.save_name)

        # Pose Label and Image Label
        self.pose_label = QLabel("Pose: Not Started", alignment=Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setFixedSize(1280, 720)  # Increased resolution for better clarity
        self.image_label.setAlignment(Qt.AlignCenter)

        # Capture Button (Initially Disabled)
        self.capture_button = QPushButton("Capture")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_image)

        layout.addWidget(QLabel("User Registration", alignment=Qt.AlignCenter))
        layout.addWidget(self.first_name_input)
        layout.addWidget(self.last_name_input)
        layout.addWidget(self.save_name_button)
        layout.addWidget(self.pose_label)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.capture_button, alignment=Qt.AlignCenter)
        layout.setAlignment(Qt.AlignCenter)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def save_name(self):
        """Save user name and create a folder based on last name."""
        first_name = self.first_name_input.text().strip().upper()
        last_name = self.last_name_input.text().strip().upper()

        if not first_name or not last_name:
            QMessageBox.warning(self, "Input Error", "Please enter both First Name and Last Name.")
            return

        self.user_last_name = last_name
        user_dir = os.path.join(REFERENCE_DIR, self.user_last_name)
        os.makedirs(user_dir, exist_ok=True)  # Create folder for user

        # Enable camera and capturing once name is set
        self.pose_index = 0
        self.pose_label.setText(f"Pose: {self.poses[self.pose_index]}")
        self.capture_button.setEnabled(True)
        self.start_camera()

    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)  # Initialize the camera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set max width
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set max height
            self.timer.start(30)  # Refresh every 30ms

    def stop_camera(self):
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()
            self.camera = None  

    def update_frame(self):
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))  # Resize to fit QLabel
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def capture_image(self):
        if self.camera is None:
            print("Camera is not initialized.")
            return

        ret, frame = self.camera.read()
        if ret:
            user_dir = os.path.join(REFERENCE_DIR, self.user_last_name)
            pose_name = self.poses[self.pose_index]  # Get the current pose name
            filename = os.path.join(user_dir, f"{self.user_last_name}_{pose_name}_Face.png")  # Save as .png
            cv2.imwrite(filename, frame)

            self.pose_index += 1

            if self.pose_index < len(self.poses):
                self.pose_label.setText(f"Pose: {self.poses[self.pose_index]}")
            else:
                self.stop_camera()
                self.stacked_widget.setCurrentIndex(0)

    def showEvent(self, event):
        super().showEvent(event)

    def hideEvent(self, event):
        self.stop_camera()
        super().hideEvent(event)

class LoginPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.face_match = False
        self.camera = None

        layout = QVBoxLayout()

        self.label = QLabel("Face Login")
        self.label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setFixedSize(1280, 720)  # Webcam feed size
        self.image_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.label)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        # Initialize timer for updating frames but do NOT start the camera here
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.lock = threading.Lock()
        self.running = True

    def start_login_camera(self):
        """Start the webcam when login is clicked."""
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.timer.start(30)  # Refresh every 30ms

    def stop_camera(self):
        """Stop the camera when leaving the login page."""
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()
            self.camera = None

    def update_frame(self):
        """Capture and update the webcam feed in QLabel."""
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def showEvent(self, event):
        """Start the camera when the page is shown."""
        super().showEvent(event)

    def hideEvent(self, event):
        """Stop the camera when leaving the login page."""
        self.stop_camera()
        super().hideEvent(event)

def main():
    app = QApplication(sys.argv)
    stacked_widget = QStackedWidget()
    stacked_widget.setFixedSize(1366, 768)  # Set the constant window size

    main_window = MainWindow()
    register_page = RegisterPage(stacked_widget)
    login_page = LoginPage(stacked_widget)

    stacked_widget.addWidget(main_window)
    stacked_widget.addWidget(register_page)
    stacked_widget.addWidget(login_page)

    main_window.register_button.clicked.connect(lambda: stacked_widget.setCurrentWidget(register_page))
    main_window.login_button.clicked.connect(lambda: stacked_widget.setCurrentWidget(login_page))
    main_window.login_button.clicked.connect(login_page.start_login_camera)
    main_window.login_button.clicked.connect(register_page.stop_camera)

    stacked_widget.setCurrentWidget(main_window)
    stacked_widget.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

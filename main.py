import sys
import cv2
import os
from PyQt5.QtWidgets import (
    QApplication, QStackedWidget, QWidget, QVBoxLayout, 
    QPushButton, QLabel, QLineEdit, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt, QSize
from deepface import DeepFace
import numpy as np
import time
import threading

REFERENCE_DIR = "reference"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 15px 32px;
                font-size: 16px;
                min-width: 200px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QLabel {
                color: #333333;
                font-size: 18px;
            }
        """)

        layout = QVBoxLayout()
        
        # Logo container
        logo_container = QVBoxLayout()
        self.logo_label = QLabel()
        # Replace 'logo.png' with your actual logo file
        logo_pixmap = QPixmap("logo.png")
        if not logo_pixmap.isNull():
            scaled_logo = logo_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_logo)
        else:
            # Fallback if no logo is found
            self.logo_label.setText("👤")
            self.logo_label.setStyleSheet("QLabel { font-size: 100px; }")
        self.logo_label.setAlignment(Qt.AlignCenter)
        
        # Title
        self.title_label = QLabel("UnlockX")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-size: 36px;
                font-weight: bold;
                color: #1976D2;
                margin: 20px;
            }
        """)
        
        # Subtitle
        self.subtitle_label = QLabel("Secure Face Recognition System")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #666666;
                margin-bottom: 30px;
            }
        """)

        # Buttons Container
        buttons_layout = QVBoxLayout()
        self.register_button = QPushButton("Register New User")
        self.login_button = QPushButton("Login with Face ID")
        
        # Add widgets to layout
        logo_container.addStretch()
        logo_container.addWidget(self.logo_label)
        logo_container.addWidget(self.title_label)
        logo_container.addWidget(self.subtitle_label)
        logo_container.addStretch()

        buttons_layout.addWidget(self.register_button, alignment=Qt.AlignCenter)
        buttons_layout.addWidget(self.login_button, alignment=Qt.AlignCenter)

        layout.addLayout(logo_container)
        layout.addLayout(buttons_layout)
        layout.setContentsMargins(50, 50, 50, 50)

        self.setLayout(layout)

class RegisterPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 15px 32px;
                font-size: 16px;
                min-width: 200px;
                margin: 10px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLineEdit {
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 16px;
                min-width: 300px;
                margin: 5px;
            }
            QLabel {
                color: #333333;
                font-size: 16px;
            }
        """)
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
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333333;
                font-size: 16px;
            }
            #status_label {
                font-size: 24px;
                color: #2196F3;
                font-weight: bold;
            }
            #back_button {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 10px 20px;
                font-size: 14px;
                min-width: 100px;
                margin: 10px;
            }
            #back_button:hover {
                background-color: #d32f2f;
            }
            #continue_button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 10px 20px;
                font-size: 14px;
                min-width: 100px;
                margin: 10px;
            }
            #continue_button:hover {
                background-color: #388E3C;
            }
        """)
        self.stacked_widget = stacked_widget
        self.face_match = False
        self.camera = None
        self.last_detection_time = 0
        self.matched_user = None

        layout = QVBoxLayout()

        # Create button container for the bottom
        button_container = QHBoxLayout()
        button_container.addStretch()  # Push buttons to the right

        self.continue_button = QPushButton("Continue")
        self.continue_button.setObjectName("continue_button")
        self.continue_button.clicked.connect(self.on_continue)

        self.back_button = QPushButton("Back")
        self.back_button.setObjectName("back_button")
        self.back_button.clicked.connect(self.go_back)

        button_container.addWidget(self.continue_button)
        button_container.addWidget(self.back_button)

        self.label = QLabel("Face Login")
        self.label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Looking for face...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setObjectName("status_label")

        self.image_label = QLabel()
        self.image_label.setFixedSize(800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.image_label)
        layout.addLayout(button_container)  # Add button container at the bottom

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.lock = threading.Lock()
        self.running = True
        self.verification_thread = threading.Thread(target=self.verify_face, daemon=True)
        self.verification_thread.start()

    def verify_face(self):
        while self.running:
            if self.camera is None:
                time.sleep(0.1)
                continue

            current_time = time.time()
            if current_time - self.last_detection_time < 1.0:  # Check every second
                time.sleep(0.1)
                continue

            self.last_detection_time = current_time

            ret, frame = self.camera.read()
            if not ret:
                continue

            try:
                # Check if there are any directories in the reference folder
                if not os.path.exists(REFERENCE_DIR):
                    continue

                user_folders = [f for f in os.listdir(REFERENCE_DIR) 
                              if os.path.isdir(os.path.join(REFERENCE_DIR, f))]

                for user_folder in user_folders:
                    user_dir = os.path.join(REFERENCE_DIR, user_folder)
                    reference_images = [f for f in os.listdir(user_dir) 
                                     if f.endswith('_Front View_Face.png')]

                    if not reference_images:
                        continue

                    reference_image_path = os.path.join(user_dir, reference_images[0])
                    
                    result = DeepFace.verify(
                        img1_path=frame,
                        img2_path=reference_image_path,
                        enforce_detection=False,
                        model_name='VGG-Face'
                    )

                    if result['verified']:
                        self.matched_user = user_folder
                        self.status_label.setText(f"Hello, {self.matched_user}")
                        return  # Exit thread after successful match

            except Exception as e:
                print(f"Verification error: {str(e)}")

            time.sleep(0.1)

    def start_login_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            self.timer.start(30)
            self.status_label.setText("Looking for face...")
            self.matched_user = None
            self.running = True
            if not self.verification_thread.is_alive():
                self.verification_thread = threading.Thread(target=self.verify_face, daemon=True)
                self.verification_thread.start()

    def stop_camera(self):
        self.running = False
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()
            self.camera = None
        if self.verification_thread.is_alive():
            self.verification_thread.join(timeout=1.0)

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

    def go_back(self):
        """Handle back button click"""
        self.stop_camera()
        self.stacked_widget.setCurrentIndex(0)  # Go back to main window

    def on_continue(self):
        """Handle continue button click"""
        # Add your continue logic here
        pass

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

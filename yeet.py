import sys
import cv2
import os
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt, QSize
from deepface import DeepFace
import numpy as np
import time
import threading
from datetime import datetime
from retinaface import RetinaFace
from face_recognition_util import FaceRecognizer

FAMILY_DIR = "family_photos"
SAFE_LOG_FILE = "safe_members.txt"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UnlockX")  # Set window title
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
        logo_container.setAlignment(Qt.AlignCenter)  # Center the container
        self.logo_label = QLabel()
        # Replace 'logo.png' with your actual logo file
        logo_pixmap = QPixmap("logo/unlockx.png")
        if not logo_pixmap.isNull():
            scaled_logo = logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_logo)  # Set fixed size for the label
            self.logo_label.setFixedSize(100, 100)  # Set fixed size for the label
        else:
            # Fallback if no logo is found
            self.logo_label.setText("👤")
            self.logo_label.setStyleSheet("QLabel { font-size: 100px; }")
            self.logo_label.setFixedSize(250, 250)  # Set fixed size even for fallback
        self.logo_label.setAlignment(Qt.AlignCenter)
        # Title
        self.title_label = QLabel("UnlockX")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-size: 36px;
                font-weight: bold;
                color: #4CAF50;
                margin: 20px;
            }
        """)
        # Subtitle
        self.subtitle_label = QLabel("Lightweight & Secure Face Recognition System")
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
        self.family_status_button = QPushButton("Family Reunification Status")
        self.family_status_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 15px 32px;
                font-size: 16px;
                min-width: 200px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.family_status_button.clicked.connect(self.show_family_status)
        # Add widgets to layout with center alignment
        logo_container.addStretch()
        logo_container.addWidget(self.logo_label, alignment=Qt.AlignCenter)  # Add explicit alignment
        logo_container.addWidget(self.title_label)
        logo_container.addWidget(self.subtitle_label)
        logo_container.addStretch()
        buttons_layout.addWidget(self.register_button, alignment=Qt.AlignCenter)
        buttons_layout.addWidget(self.login_button, alignment=Qt.AlignCenter)
        buttons_layout.addWidget(self.family_status_button, alignment=Qt.AlignCenter)
        layout.addLayout(logo_container)
        layout.addLayout(buttons_layout)
        layout.setContentsMargins(50, 50, 50, 50)
        self.setLayout(layout)

    def show_family_status(self):
        """Show overall family reunification status"""
        status_dialog = QDialog(self)
        status_dialog.setWindowTitle("Family Reunification Status")
        status_dialog.setFixedSize(600, 400)
        layout = QVBoxLayout()
        status_text = QTextEdit()
        status_text.setReadOnly(True)
        
        # Get all safe members
        safe_members = set()
        if os.path.exists(SAFE_LOG_FILE):
            with open(SAFE_LOG_FILE, 'r') as f:
                safe_members = {line.strip().split(',')[0] for line in f}
        
        # Check each family
        status = []
        if os.path.exists(FAMILY_DIR):
            for family in os.listdir(FAMILY_DIR):
                family_path = os.path.join(FAMILY_DIR, family)
                if not os.path.isdir(family_path):
                    continue
                    
                # Get members from face files
                face_files = [f for f in os.listdir(family_path) if f.endswith('_face.jpg')]
                if not face_files:
                    continue
                    
                members = {os.path.splitext(f)[0][:-5] for f in face_files}  # Remove _face.jpg
                safe = {m for m in members if m in safe_members}
                missing = members - safe
                
                status.append(f"\nFamily: {family}")
                status.append(f"Safe Members: {', '.join(safe) if safe else 'None'}")
                status.append(f"Missing Members: {', '.join(missing) if missing else 'None'}")
        
        status_text.setText('\n'.join(status) if status else "No family records found.")
        layout.addWidget(status_text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(status_dialog.accept)
        layout.addWidget(close_btn)
        
        status_dialog.setLayout(layout)
        status_dialog.exec_()

class RegisterPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.setWindowTitle("Register | UnlockX")
        self.setFixedSize(800, 600)
        self.stacked_widget = stacked_widget
        self.family_name = ""
        self.faces = []
        self.current_face_index = 0
        self.original_image = None
        self.camera = None  # Add camera attribute
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
            }
            QLineEdit {
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 16px;
            }
            QLabel {
                color: #333333;
                font-size: 16px;
            }
        """)
        layout = QVBoxLayout()
        # Family name input
        self.family_name_input = QLineEdit()
        self.family_name_input.setPlaceholderText("Enter Family Name")
        # Photo selection
        photo_layout = QHBoxLayout()
        self.photo_path = QLineEdit()
        self.photo_path.setReadOnly(True)
        browse_button = QPushButton("Browse Family Photo")
        browse_button.clicked.connect(self.browse_photo)
        photo_layout.addWidget(self.photo_path)
        photo_layout.addWidget(browse_button)
        # Face display and controls
        self.photo_display = QLabel()
        self.photo_display.setFixedSize(200, 200)
        self.photo_display.setStyleSheet("border: 2px solid #ccc;")
        self.face_label = QLabel("Upload a family photo to start")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name for highlighted face")
        self.name_input.setEnabled(False)
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous Face")
        self.next_button = QPushButton("Next Face")
        self.finish_button = QPushButton("Finish Registration")
        self.cancel_button = QPushButton("Cancel")
        self.prev_button.clicked.connect(self.previous_face)
        self.next_button.clicked.connect(self.next_face)
        self.finish_button.clicked.connect(self.finish_registration)
        self.cancel_button.clicked.connect(self.cancel_registration)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.finish_button)
        nav_layout.addWidget(self.cancel_button)
        # Add all to layout
        layout.addWidget(QLabel("Family Registration"))
        layout.addWidget(self.family_name_input)
        layout.addLayout(photo_layout)
        layout.addWidget(self.photo_display)
        layout.addWidget(self.face_label)
        layout.addWidget(self.name_input)
        layout.addLayout(nav_layout)
        self.setLayout(layout)

    def browse_photo(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Family Photo", "", "Images (*.png *.xpm *.jpg *.jpeg)", options=options)
        if file_path:
            self.photo_path.setText(file_path)
            self.detect_faces(file_path)

    def detect_faces(self, image_path):
        try:
            # Load image as BGR then convert to RGB for display
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise Exception("Failed to load image")
                
            # Convert to RGB for display purposes
            self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Detect faces using RetinaFace
            faces = RetinaFace.detect_faces(image_path)
            self.faces = []
            
            if isinstance(faces, dict):
                for face_data in faces.values():
                    facial_area = face_data["facial_area"]
                    x1, y1, x2, y2 = map(int, facial_area)
                    
                    # Add padding
                    pad = 20
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(self.original_image.shape[1], x2 + pad)
                    y2 = min(self.original_image.shape[0], y2 + pad)
                    
                    # Extract face region
                    face_img = self.original_image[y1:y2, x1:x2].copy()
                    
                    self.faces.append({
                        'box': (x1, y1, x2, y2),
                        'name': f"Face {len(self.faces) + 1}",
                        'image': face_img
                    })
            
            if self.faces:
                self.current_face_index = 0
                self.name_input.setEnabled(True)
                self.name_input.setText(self.faces[0]['name'])
                self.update_display()
            else:
                QMessageBox.warning(self, "Warning", "No faces detected in the image")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to detect faces: {str(e)}")

    def update_display(self):
        if not self.faces or self.current_face_index >= len(self.faces):
            return
            
        face = self.faces[self.current_face_index]
        face_img = face['image']
        
        if face_img is not None and face_img.size > 0:
            # Convert BGR to RGB for display
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Create QImage from RGB data
            h, w, ch = rgb_face.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_face.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to pixmap and scale
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.photo_display.size(), Qt.KeepAspectRatio)
            self.photo_display.setPixmap(scaled_pixmap)
            
        self.face_label.setText(f"Face {self.current_face_index + 1} of {len(self.faces)}")

    def next_face(self):
        if self.current_face_index < len(self.faces) - 1:
            # Save current name before moving to next face
            current_name = self.name_input.text().strip()
            self.faces[self.current_face_index]['name'] = current_name
            
            self.current_face_index += 1
            self.name_input.setText(self.faces[self.current_face_index]['name'])
            self.update_display()

    def previous_face(self):
        if self.current_face_index > 0:
            # Save current name before moving to previous face
            current_name = self.name_input.text().strip()
            self.faces[self.current_face_index]['name'] = current_name
            
            self.current_face_index -= 1
            self.name_input.setText(self.faces[self.current_face_index]['name'])
            self.update_display()

    def finish_registration(self):
        family_name = self.family_name_input.text().strip()
        if not family_name:
            QMessageBox.warning(self, "Input Error", "Please enter a family name.")
            return
        
        if not self.faces:
            QMessageBox.warning(self, "Input Error", "No faces detected. Please upload a valid family photo.")
            return

        # Save current face name before processing
        current_name = self.name_input.text().strip()
        self.faces[self.current_face_index]['name'] = current_name
            
        # Verify all faces have names
        if any(not face['name'] or face['name'].startswith('Face ') for face in self.faces):
            QMessageBox.warning(self, "Input Error", "Please name all faces before proceeding.")
            return
            
        try:
            family_dir = os.path.join(FAMILY_DIR, family_name)
            os.makedirs(family_dir, exist_ok=True)
            
            # Save original photo
            if self.photo_path.text():
                photo_ext = os.path.splitext(self.photo_path.text())[1]
                shutil.copy(self.photo_path.text(), 
                          os.path.join(family_dir, f"family_original{photo_ext}"))
            
            # Save each face with its name
            for face in self.faces:
                face_name = face['name'].strip()
                # Save in BGR color space for correct colors
                face_path = os.path.join(family_dir, f"{face_name}_face.jpg")
                cv2.imwrite(face_path, face['image'])
            
            QMessageBox.information(self, "Success", 
                f"Family {family_name} registered successfully with {len(self.faces)} members.")
            self.stacked_widget.setCurrentIndex(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save faces: {str(e)}")

    def cancel_registration(self):
        self.stacked_widget.setCurrentIndex(0)

    def stop_camera(self):
        """Stop and release camera if running"""
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
            self.camera = None

class LoginPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.setWindowTitle("Login | UnlockX")  # Set window title
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
        self.camera = None
        self.last_detection_time = 0
        self.matched_user = None
        self.matched_family = None
        self.confidence_score = 0
        self.face_embeddings_cache = {}  # Cache for face embeddings
        self.detector_backend = 'retinaface'  # Specify better detector backend
        self.face_detector = None
        self.current_frame = None
        self.detection_active = False
        self.detection_cooldown = 0.3  # Reduce cooldown for more frequent checks
        self.face_recognizer = FaceRecognizer(FAMILY_DIR)
        
        # Set up UI components
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # Camera container
        camera_container = QWidget()
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setAlignment(Qt.AlignCenter)
        
        self.label = QLabel("Face Login")
        self.label.setAlignment(Qt.AlignCenter)
        
        self.status_label = QLabel("Looking for face...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setObjectName("status_label")
        
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: #666666; font-size: 14px;")
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 10px;
                background-color: #ffffff;
            }
        """)
        
        # Button container
        button_container = QHBoxLayout()
        button_container.setAlignment(Qt.AlignCenter)
        
        self.continue_button = QPushButton("Continue")
        self.continue_button.setObjectName("continue_button")
        self.continue_button.clicked.connect(self.on_continue)
        self.continue_button.setEnabled(False)  # Disable until face is detected
        
        self.back_button = QPushButton("Back")
        self.back_button.setObjectName("back_button")
        self.back_button.clicked.connect(self.go_back)
        
        button_container.addWidget(self.continue_button)
        button_container.addWidget(self.back_button)
        
        # Add widgets to layout
        layout.addWidget(self.label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.confidence_label)
        camera_layout.addWidget(self.image_label)
        layout.addWidget(camera_container)
        layout.addLayout(button_container)
        
        # Family status label
        self.family_status_label = QLabel("")
        self.family_status_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 14px;
                margin-top: 10px;
            }
        """)
        self.family_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.family_status_label)
        
        self.setLayout(layout)
        
        # Set up timer and thread for face detection
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.lock = threading.Lock()
        self.running = False
        self.verification_thread = None

    def verify_face(self):
        """Thread function to verify faces against the database"""
        while self.running:
            if self.camera is None or self.current_frame is None:
                time.sleep(0.1)
                continue
            
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_detection_time < self.detection_cooldown:
                time.sleep(0.05)
                continue
            
            self.last_detection_time = current_time
            self.detection_active = True
            
            frame = self.current_frame.copy()
            
            try:
                # Use FaceRecognizer to identify face
                name, family, confidence = self.face_recognizer.identify_face(frame)
                
                if name:
                    self.matched_user = name
                    self.matched_family = family
                    self.confidence_score = confidence
                    
                    self.status_label.setText(f"Welcome, {self.matched_user}")
                    self.confidence_label.setText(f"Confidence: {self.confidence_score:.2%}")
                    self.continue_button.setEnabled(True)
                    self.check_family_status(self.matched_user, self.matched_family)
                else:
                    self.status_label.setText("Face detected, no match found")
                    self.confidence_label.setText("Please try again or register")
                    self.continue_button.setEnabled(False)
                    
            except Exception as e:
                print(f"Verification error: {str(e)}")
                self.status_label.setText("Looking for face...")
                self.confidence_label.setText("")
                self.continue_button.setEnabled(False)
            
            finally:
                self.detection_active = False
            
            time.sleep(0.1)

    def log_safe_member(self, member_name):
        """Log member as safe with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(SAFE_LOG_FILE, 'a+') as f:
            f.write(f"{member_name},{timestamp}\n")

    def check_family_status(self, user_name, family_name):
        """Check and update family status display"""
        try:
            family_path = os.path.join(FAMILY_DIR, family_name)
            face_files = [f for f in os.listdir(family_path) if f.endswith('_face.jpg')]
            family_members = [os.path.splitext(f)[0][:-5] for f in face_files]
            
            safe_members = set()
            if os.path.exists(SAFE_LOG_FILE):
                with open(SAFE_LOG_FILE, 'r') as f:
                    safe_members = {line.strip().split(',')[0] for line in f}
            
            found_members = [m for m in family_members if m in safe_members]
            missing_members = [m for m in family_members if m not in safe_members]
            
            status_text = f"Family: {family_name}\n"
            status_text += f"Found: {', '.join(found_members)}\n"
            if missing_members:
                status_text += f"Missing: {', '.join(missing_members)}"
            
            self.family_status_label.setText(status_text)
        except Exception as e:
            print(f"Error checking family status: {str(e)}")

    def draw_face_box(self, frame):
        """Draw a box around detected faces"""
        if frame is None:
            return frame
        
        try:
            # Create a copy of the frame to avoid modifying the original
            display_frame = frame.copy()
            
            # Use RetinaFace for face detection
            # Save to temporary file for detection
            temp_path = "temp_detection.jpg"
            cv2.imwrite(temp_path, display_frame)
            
            # Detect faces
            faces = RetinaFace.detect_faces(temp_path)
            
            # Remove temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Draw boxes around detected faces
            if isinstance(faces, dict):
                for face_key in faces:
                    face = faces[face_key]
                    # Get face coordinates
                    x1, y1, x2, y2 = face['facial_area']
                    
                    # Draw rectangle (green for matched, blue for detected)
                    color = (0, 255, 0) if self.matched_user else (255, 0, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add confidence text if matched
                    if self.matched_user:
                        confidence_text = f"{self.confidence_score:.2%}"
                        cv2.putText(display_frame, confidence_text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return display_frame
        except Exception as e:
            print(f"Error drawing face box: {str(e)}")
            return frame

    def start_login_camera(self):
        """Initialize and start the camera for login"""
        if self.camera is None:
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    QMessageBox.critical(self, "Camera Error",
                                     "Could not access the camera. Please ensure camera permissions are granted.")
                    return
                
                # Set camera properties for better quality
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
                
                # Start timer for frame updates
                self.timer.start(30)  # Update every 30ms for smooth video
                
                # Reset UI state
                self.status_label.setText("Looking for face...")
                self.confidence_label.setText("")
                self.matched_user = None
                self.continue_button.setEnabled(False)
                
                # Start verification thread
                self.running = True
                if not self.verification_thread or not self.verification_thread.is_alive():
                    self.verification_thread = threading.Thread(target=self.verify_face, daemon=True)
                    self.verification_thread.start()
                
                # Pre-load face embeddings
                if not self.face_embeddings_cache:
                    threading.Thread(target=self.load_face_embeddings, daemon=True).start()
                
            except Exception as e:
                QMessageBox.critical(self, "Camera Error",
                                 f"Failed to initialize camera: {str(e)}\nPlease check camera permissions.")
                return

    def stop_camera(self):
        """Stop the camera and verification thread"""
        self.running = False
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()
            self.camera = None
        
        if self.verification_thread and self.verification_thread.is_alive():
            self.verification_thread.join(timeout=1.0)
        
        self.matched_user = None
        self.confidence_score = 0

    def update_frame(self):
        """Capture and update the webcam feed in QLabel with face detection"""
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                # Store current frame for verification thread
                self.current_frame = frame.copy()
                
                # Only process frame for display if we're not currently in detection
                if not self.detection_active:
                    # Draw face detection box
                    display_frame = self.draw_face_box(frame)
                    
                    # Convert to RGB for display
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Create QImage and display
                    height, width, channel = display_frame.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def go_back(self):
        """Handle back button click"""
        self.stop_camera()
        self.stacked_widget.setCurrentIndex(0)  # Go back to main window

    def on_continue(self):
        """Handle continue button click and update family status"""
        if self.matched_user:
            # Log the user as safe with timestamp
            self.log_safe_member(self.matched_user)
            
            # Show reunification status
            self.check_family_status(self.matched_user, self.matched_family)
            
            QMessageBox.information(self, "Status Updated", 
                                 f"{self.matched_user} has been marked as safe.")
            
            # Return to main window
            self.stop_camera()
            self.stacked_widget.setCurrentIndex(0)
        else:
            QMessageBox.warning(self, "No Match", 
                             "No face match detected. Please try again or register.")

    def showEvent(self, event):
        """Start the camera when the page is shown"""
        self.start_login_camera()
        super().showEvent(event)

    def hideEvent(self, event):
        """Stop the camera when leaving the login page"""
        self.stop_camera()
        super().hideEvent(event)
import sys
import cv2
import os
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt, QSize
import numpy as np
import warnings
import time
import threading
from datetime import datetime
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Filter numpy warnings about rcond parameter
warnings.filterwarnings('ignore', category=FutureWarning)

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
        self.register_button = QPushButton("Register Family")
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
        status_dialog.setFixedSize(800, 600)
        
        # Add search functionality
        search_container = QWidget()
        search_layout = QHBoxLayout(search_container)
        search_input = QLineEdit()
        search_input.setPlaceholderText("Search by family name...")
        search_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 20px;
                font-size: 14px;
                min-width: 300px;
            }
        """)
        search_layout.addWidget(search_input)
        
        # Create main container for family cards
        status_container = QWidget()
        status_layout = QVBoxLayout(status_container)
        status_layout.setSpacing(15)
        
        # Create a dictionary to store all family cards
        family_cards = {}
        
        def filter_families(text):
            """Filter family cards based on search text"""
            search_text = text.lower()
            for family, card in family_cards.items():
                if search_text in family.lower():
                    card.show()
                else:
                    card.hide()
        
        search_input.textChanged.connect(filter_families)
        
        # Rest of the dialog setup
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Add title and search bar
        title = QLabel("Family Reunification Status")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #2196F3;
                padding-bottom: 10px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(search_container)
        
        safe_members = set()
        if os.path.exists(SAFE_LOG_FILE):
            with open(SAFE_LOG_FILE, 'r') as f:
                safe_members = {line.strip().split(',')[0] for line in f}
        
        if os.path.exists(FAMILY_DIR):
            total_members = 0
            total_safe = 0
            
            for family in os.listdir(FAMILY_DIR):
                family_path = os.path.join(FAMILY_DIR, family)
                if not os.path.isdir(family_path):
                    continue
                
                face_files = [f for f in os.listdir(family_path) if f.endswith('_face.jpg')]
                if not face_files:
                    continue
                
                members = {os.path.splitext(f)[0][:-5] for f in face_files}
                safe = {m for m in members if m in safe_members}
                missing = members - safe
                
                # Create family card
                family_card = QWidget()
                family_card.setStyleSheet("""
                    QWidget {
                        background-color: white;
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        padding: 10px;
                    }
                """)
                card_layout = QVBoxLayout(family_card)
                
                # Family name header
                family_header = QLabel(f"Family: {family}")
                family_header.setStyleSheet("font-size: 18px; color: #1976D2; padding: 5px;")
                card_layout.addWidget(family_header)
                
                # Safe members with green indicators
                if safe:
                    safe_text = QLabel("✓ Safe Members: " + ", ".join(safe))
                    safe_text.setStyleSheet("color: #4CAF50; padding: 5px;")
                    safe_text.setWordWrap(True)
                    card_layout.addWidget(safe_text)
                
                # Missing members with red indicators
                if missing:
                    missing_text = QLabel("⚠ Missing Members: " + ", ".join(missing))
                    missing_text.setStyleSheet("color: #f44336; padding: 5px;")
                    missing_text.setWordWrap(True)
                    card_layout.addWidget(missing_text)
                
                status_layout.addWidget(family_card)
                family_cards[family] = family_card
                
                total_members += len(members)
                total_safe += len(safe)
            
            # Add summary at the top
            if total_members > 0:
                summary = QLabel(f"Total: {total_safe} of {total_members} members found safe ({(total_safe/total_members*100):.1f}%)")
                summary.setStyleSheet("""
                    QLabel {
                        font-size: 16px;
                        color: #333;
                        padding: 10px;
                        background-color: #e3f2fd;
                        border-radius: 5px;
                    }
                """)
                layout.addWidget(summary)
        
        # Add scrollable area for status
        scroll = QScrollArea()
        scroll.setWidget(status_container)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        layout.addWidget(scroll)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(status_dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
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
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 480))

    def reset(self):
        """Reset registration form to initial state"""
        self.family_name_input.clear()
        self.photo_path.clear()
        self.photo_display.clear()
        self.name_input.clear()
        self.name_input.setEnabled(False)
        self.face_label.setText("Upload a family photo to start")
        self.faces = []
        self.current_face_index = 0
        self.original_image = None

    def browse_photo(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Family Photo", "", "Images (*.png *.xpm *.jpg *.jpeg)", options=options)
        if file_path:
            self.photo_path.setText(file_path)
            self.detect_faces(file_path)

    def detect_faces(self, image_path):
        try:
            # Load image as BGR
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise Exception("Failed to load image")
                
            # Convert to RGB for display purposes
            self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Detect faces using InsightFace
            faces = self.face_app.get(self.original_image)
            self.faces = []
            
            if faces:
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Increase padding for larger face crops
                    pad = 40  # Increased from 20 to 40
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
            self.reset()  # Reset the form after successful registration
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

    def showEvent(self, event):
        """Reset form when page is shown"""
        self.reset()
        super().showEvent(event)

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
        self.face_match = False
        self.camera = None
        self.last_detection_time = 0
        self.matched_user = None
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 480))
        self.face_embeddings_cache = {}
        self.face_detector = None
        self.matched_users = set()  # Track multiple matches
        self.matched_families = {}  # Track family for each match
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)  # Center all content vertically
        # Create a container widget for the camera feed
        camera_container = QWidget()
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setAlignment(Qt.AlignCenter)
        self.label = QLabel("Face Login")
        self.label.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("Looking for face...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setObjectName("status_label")
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)  # Standard 4:3 aspect ratio
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 10px;
                background-color: #ffffff;
            }
        """)
        # Create button container for the bottom
        button_container = QHBoxLayout()
        button_container.setAlignment(Qt.AlignCenter)  # Center the buttons
        self.continue_button = QPushButton("Continue")
        self.continue_button.setObjectName("continue_button")
        self.continue_button.clicked.connect(self.on_continue)
        self.back_button = QPushButton("Back")
        self.back_button.setObjectName("back_button")
        self.back_button.clicked.connect(self.go_back)
        button_container.addWidget(self.continue_button)
        button_container.addWidget(self.back_button)
        # Add widgets to the layout
        layout.addWidget(self.label)
        layout.addWidget(self.status_label)
        camera_layout.addWidget(self.image_label)
        layout.addWidget(camera_container)
        layout.addLayout(button_container)
        self.setLayout(layout)
        # Add family status label
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
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.lock = threading.Lock()
        self.running = True
        self.verification_thread = threading.Thread(target=self.verify_face, daemon=True)
        self.verification_thread.start()

    def cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def verify_face(self):
        while self.running:
            if self.camera is None:
                time.sleep(0.1)
                continue

            current_time = time.time()
            if current_time - self.last_detection_time < 0.5:
                time.sleep(0.1)
                continue

            self.last_detection_time = current_time
            ret, frame = self.camera.read()
            if not ret:
                continue

            try:
                # Cache face embeddings if not already cached
                if not self.face_embeddings_cache:
                    for family_folder in os.listdir(FAMILY_DIR):
                        family_path = os.path.join(FAMILY_DIR, family_folder)
                        if not os.path.isdir(family_path):
                            continue

                        face_files = [f for f in os.listdir(family_path) 
                                    if f.endswith('_face.jpg')]
                        
                        for face_file in face_files:
                            face_path = os.path.join(family_path, face_file)
                            try:
                                img = cv2.imread(face_path)
                                faces = self.face_app.get(img)
                                if faces:
                                    embedding = faces[0].embedding
                                    name = os.path.splitext(face_file)[0][:-5]
                                    self.face_embeddings_cache[face_path] = {
                                        'embedding': embedding,
                                        'name': name,
                                        'family': family_folder
                                    }
                            except Exception as e:
                                print(f"Error caching embedding for {face_path}: {str(e)}")

                # Detect and get embeddings for current frame
                faces = self.face_app.get(frame)
                if not faces:
                    continue

                frame_embedding = faces[0].embedding
                best_match = None
                highest_similarity = 0
                similarity_threshold = 0.35  # Reduced from 0.5 to 0.35 for more lenient matching
                matched_family = None

                for face_path, cache_data in self.face_embeddings_cache.items():
                    try:
                        similarity = self.cosine_similarity(
                            frame_embedding, 
                            cache_data['embedding']
                        )

                        if similarity > similarity_threshold and similarity > highest_similarity:
                            highest_similarity = similarity
                            best_match = cache_data['name']
                            matched_family = cache_data['family']

                    except Exception as e:
                        print(f"Error comparing with {face_path}: {str(e)}")

                if best_match and best_match not in self.matched_users:
                    self.matched_users.add(best_match)
                    self.matched_families[best_match] = matched_family
                    # Update status with all matched users
                    status_text = "Welcome!\nFound: " + ", ".join(self.matched_users)
                    self.status_label.setText(status_text)
                    # Show status for most recently matched user
                    self.check_family_status(best_match, matched_family)
                    # Don't return, keep looking for more faces

            except Exception as e:
                print(f"Verification error: {str(e)}")

            time.sleep(0.1)

    def log_safe_member(self, member_name):
        """Log member as safe with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(SAFE_LOG_FILE, 'a+') as f:
            f.write(f"{member_name},{timestamp}\n")

    def check_family_status(self, user_name, family_name=None):
        """Check and update family status display"""
        try:
            if family_name is None:
                # Try to find family by searching through directories
                for folder in os.listdir(FAMILY_DIR):
                    family_path = os.path.join(FAMILY_DIR, folder)
                    if os.path.isdir(family_path):
                        face_files = [f for f in os.listdir(family_path) if f.endswith('_face.jpg')]
                        members = [os.path.splitext(f)[0][:-5] for f in face_files]
                        if user_name in members:
                            family_name = folder
                            break

            if family_name:
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

    def start_login_camera(self):
        if self.camera is None:
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    QMessageBox.critical(self, "Camera Error",
                                         "Could not access the camera. Please ensure camera permissions are granted in System Settings.")
                    return
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.timer.start(30)
                self.status_label.setText("Looking for face...")
                self.matched_user = None
                self.matched_users.clear()  # Clear previous matches
                self.matched_families.clear()
                self.running = True
                if not self.verification_thread.is_alive():
                    self.verification_thread = threading.Thread(target=self.verify_face, daemon=True)
                    self.verification_thread.start()
            except Exception as e:
                QMessageBox.critical(self, "Camera Error",
                                     f"Failed to initialize camera: {str(e)}\nPlease check camera permissions in System Settings.")
                return

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
                frame = cv2.resize(frame, (640, 480))
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
        """Handle continue button click and update family status"""
        if self.matched_users:
            # Log all matched users as safe
            for user in self.matched_users:
                self.log_safe_member(user)
                self.check_family_status(user, self.matched_families[user])
            
            names = ", ".join(self.matched_users)
            QMessageBox.information(self, "Status Updated", 
                f"The following members have been marked as safe:\n{names}")
            
            # Return to main window
            self.stop_camera()
            self.stacked_widget.setCurrentIndex(0)

    def showEvent(self, event):
        """Start the camera when the page is shown."""
        self.start_login_camera()
        super().showEvent(event)

    def hideEvent(self, event):
        """Stop the camera when leaving the login page."""
        self.stop_camera()
        super().hideEvent(event)

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("logo/unlockx.png")) # Set application icon
    stacked_widget = QStackedWidget()
    stacked_widget.setFixedSize(1366, 768)  # Set the constant window size

    main_window = MainWindow()
    register_page = RegisterPage(stacked_widget)
    login_page = LoginPage(stacked_widget)

    stacked_widget.addWidget(main_window)
    stacked_widget.addWidget(register_page)
    stacked_widget.addWidget(login_page)

    def update_title(widget):
        if isinstance(widget, MainWindow):
            stacked_widget.setWindowTitle("UnlockX")
        elif isinstance(widget, RegisterPage):
            stacked_widget.setWindowTitle("Register | UnlockX")
        elif isinstance(widget, LoginPage):
            stacked_widget.setWindowTitle("Login | UnlockX")

    stacked_widget.currentChanged.connect(lambda: update_title(stacked_widget.currentWidget()))

    main_window.register_button.clicked.connect(lambda: stacked_widget.setCurrentWidget(register_page))
    main_window.login_button.clicked.connect(lambda: stacked_widget.setCurrentWidget(login_page))
    main_window.login_button.clicked.connect(login_page.start_login_camera)

    os.makedirs(FAMILY_DIR, exist_ok=True)
    if not os.path.exists(SAFE_LOG_FILE):
        open(SAFE_LOG_FILE, 'a').close()

    stacked_widget.setCurrentWidget(main_window)
    stacked_widget.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


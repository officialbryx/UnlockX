import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
import time
from datetime import datetime
import numpy as np
from insightface.app import FaceAnalysis

class LoginPage(ttk.Frame):
    def __init__(self, parent, family_dir, safe_log_file):
        super().__init__(parent)
        self.parent = parent
        self.family_dir = family_dir
        self.safe_log_file = safe_log_file
        self.camera = None
        self.last_detection_time = 0
        self.matched_users = set()
        self.matched_families = {}
        
        # Initialize face detection
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 480))
        self.face_embeddings_cache = {}
        
        self.running = True
        self.lock = threading.Lock()
        
        self.create_widgets()
        
    def create_widgets(self):
        # Status label
        self.status_label = ttk.Label(self, 
                                    text="Looking for face...",
                                    font=('Helvetica', 14))
        self.status_label.pack(pady=10)
        
        # Camera feed display
        self.image_label = ttk.Label(self)
        self.image_label.pack(expand=True, pady=20)
        
        # Family status
        self.family_status_label = ttk.Label(self,
                                           text="",
                                           font=('Helvetica', 12))
        self.family_status_label.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', pady=20)
        
        self.continue_button = ttk.Button(button_frame,
                                        text="Continue",
                                        command=self.on_continue)
        self.continue_button.pack(side='left', padx=5)
        
        self.back_button = ttk.Button(button_frame,
                                    text="Back to Main",
                                    command=self.go_back)
        self.back_button.pack(side='right', padx=5)
    
    def start_camera(self):
        if self.camera is None:
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    messagebox.showerror("Camera Error",
                        "Could not access the camera. Please check permissions.")
                    return
                
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.running = True
                self.matched_users.clear()
                self.matched_families.clear()
                
                self.update_frame()
                
                self.verification_thread = threading.Thread(
                    target=self.verify_face,
                    daemon=True
                )
                self.verification_thread.start()
                
            except Exception as e:
                messagebox.showerror("Camera Error",
                    f"Failed to initialize camera: {str(e)}")
    
    def stop_camera(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if hasattr(self, 'verification_thread'):
            self.verification_thread.join(timeout=1.0)
    
    def update_frame(self):
        if self.camera is not None and self.running:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                
                self.image_label.config(image=photo)
                self.image_label.image = photo
            
            self.after(30, self.update_frame)
    
    def cosine_similarity(self, emb1, emb2):
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
                if not self.face_embeddings_cache:
                    for family_folder in os.listdir(self.family_dir):
                        family_path = os.path.join(self.family_dir, family_folder)
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
                                print(f"Error caching embedding: {str(e)}")
                
                faces = self.face_app.get(frame)
                if not faces:
                    continue
                
                frame_embedding = faces[0].embedding
                best_match = None
                highest_similarity = 0
                similarity_threshold = 0.35
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
                        print(f"Error comparing faces: {str(e)}")
                
                if best_match and best_match not in self.matched_users:
                    self.matched_users.add(best_match)
                    self.matched_families[best_match] = matched_family
                    
                    self.after(0, self.update_status,
                             f"Welcome!\nFound: {', '.join(self.matched_users)}")
                    
                    self.after(0, self.check_family_status,
                             best_match, matched_family)
                    
            except Exception as e:
                print(f"Verification error: {str(e)}")
            
            time.sleep(0.1)
    
    def update_status(self, text):
        self.status_label.config(text=text)
    
    def check_family_status(self, user_name, family_name=None):
        try:
            if family_name:
                family_path = os.path.join(self.family_dir, family_name)
                face_files = [f for f in os.listdir(family_path)
                            if f.endswith('_face.jpg')]
                family_members = [os.path.splitext(f)[0][:-5]
                                for f in face_files]
                
                safe_members = set()
                if os.path.exists(self.safe_log_file):
                    with open(self.safe_log_file, 'r') as f:
                        safe_members = {line.strip().split(',')[0]
                                      for line in f}
                
                found = [m for m in family_members if m in safe_members]
                missing = [m for m in family_members if m not in safe_members]
                
                status = f"Family: {family_name}\n"
                status += f"Found: {', '.join(found)}\n"
                if missing:
                    status += f"Missing: {', '.join(missing)}"
                
                self.family_status_label.config(text=status)
                
        except Exception as e:
            print(f"Error checking family status: {str(e)}")
    
    def log_safe_member(self, member_name):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.safe_log_file, 'a+') as f:
            f.write(f"{member_name},{timestamp}\n")
    
    def on_continue(self):
        if self.matched_users:
            for user in self.matched_users:
                self.log_safe_member(user)
                self.check_family_status(user, self.matched_families[user])
            
            names = ", ".join(self.matched_users)
            messagebox.showinfo("Status Updated",
                f"The following members have been marked as safe:\n{names}")
            
            self.stop_camera()
            self.parent.show_main_page()
    
    def go_back(self):
        self.stop_camera()
        self.parent.show_main_page()
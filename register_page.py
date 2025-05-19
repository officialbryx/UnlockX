import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import shutil
from insightface.app import FaceAnalysis

class RegisterPage(ttk.Frame):
    def __init__(self, parent, family_dir):
        super().__init__(parent)
        self.parent = parent
        self.family_dir = family_dir
        self.family_name = ""
        self.faces = []
        self.current_face_index = 0
        self.original_image = None
        self.camera = None
        
        # Initialize face detection
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 480))
        
        self.create_widgets()
        
    def create_widgets(self):
        # Family name input
        input_frame = ttk.Frame(self)
        input_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(input_frame, text="Family Name:").pack(side='left')
        self.family_name_input = ttk.Entry(input_frame)
        self.family_name_input.pack(side='left', fill='x', expand=True, padx=5)
        
        # Photo selection
        photo_frame = ttk.Frame(self)
        photo_frame.pack(fill='x', padx=20, pady=10)
        
        self.photo_path = ttk.Entry(photo_frame)
        self.photo_path.pack(side='left', fill='x', expand=True)
        
        browse_btn = ttk.Button(photo_frame, 
                              text="Browse Family Photo",
                              command=self.browse_photo)
        browse_btn.pack(side='right', padx=5)
        
        # Face display
        display_frame = ttk.Frame(self)
        display_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        self.photo_display = ttk.Label(display_frame)
        self.photo_display.pack(expand=True)
        
        # Face labeling
        label_frame = ttk.Frame(self)
        label_frame.pack(fill='x', padx=20, pady=10)
        
        self.face_label = ttk.Label(label_frame, text="Upload a family photo to start")
        self.face_label.pack()
        
        self.name_input = ttk.Entry(label_frame)
        self.name_input.pack(fill='x', pady=5)
        self.name_input.insert(0, "Enter name for highlighted face")
        self.name_input.config(state='disabled')
        
        # Navigation buttons
        nav_frame = ttk.Frame(self)
        nav_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(nav_frame, text="Previous Face",
                  command=self.previous_face).pack(side='left', padx=5)
        ttk.Button(nav_frame, text="Next Face",
                  command=self.next_face).pack(side='left', padx=5)
        ttk.Button(nav_frame, text="Finish Registration",
                  command=self.finish_registration).pack(side='left', padx=5)
        ttk.Button(nav_frame, text="Back to Main",
                  command=self.go_back).pack(side='right', padx=5)
    
    def browse_photo(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if file_path:
            self.photo_path.delete(0, tk.END)
            self.photo_path.insert(0, file_path)
            self.detect_faces(file_path)
    
    def detect_faces(self, image_path):
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise Exception("Failed to load image")
            
            faces = self.face_app.get(self.original_image)
            self.faces = []
            
            if faces:
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    pad = 40
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(self.original_image.shape[1], x2 + pad)
                    y2 = min(self.original_image.shape[0], y2 + pad)
                    
                    face_img = self.original_image[y1:y2, x1:x2].copy()
                    
                    self.faces.append({
                        'box': (x1, y1, x2, y2),
                        'name': f"Face {len(self.faces) + 1}",
                        'image': face_img
                    })
            
            if self.faces:
                self.current_face_index = 0
                self.name_input.config(state='normal')
                self.name_input.delete(0, tk.END)
                self.name_input.insert(0, self.faces[0]['name'])
                self.update_display()
            else:
                messagebox.showwarning("Warning", "No faces detected in the image")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect faces: {str(e)}")
    
    def update_display(self):
        if not self.faces or self.current_face_index >= len(self.faces):
            return
            
        face = self.faces[self.current_face_index]
        face_img = face['image']
        
        if face_img is not None and face_img.size > 0:
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_face)
            pil_image = pil_image.resize((200, 200), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            self.photo_display.config(image=self.photo)
            self.face_label.config(text=f"Face {self.current_face_index + 1} of {len(self.faces)}")
    
    def next_face(self):
        if self.current_face_index < len(self.faces) - 1:
            self.faces[self.current_face_index]['name'] = self.name_input.get().strip()
            self.current_face_index += 1
            self.name_input.delete(0, tk.END)
            self.name_input.insert(0, self.faces[self.current_face_index]['name'])
            self.update_display()
    
    def previous_face(self):
        if self.current_face_index > 0:
            self.faces[self.current_face_index]['name'] = self.name_input.get().strip()
            self.current_face_index -= 1
            self.name_input.delete(0, tk.END)
            self.name_input.insert(0, self.faces[self.current_face_index]['name'])
            self.update_display()
    
    def finish_registration(self):
        family_name = self.family_name_input.get().strip()
        if not family_name:
            messagebox.showwarning("Input Error", "Please enter a family name.")
            return
        
        if not self.faces:
            messagebox.showwarning("Input Error", "No faces detected. Please upload a valid family photo.")
            return
        
        self.faces[self.current_face_index]['name'] = self.name_input.get().strip()
        
        if any(not face['name'] or face['name'].startswith('Face ') for face in self.faces):
            messagebox.showwarning("Input Error", "Please name all faces before proceeding.")
            return
            
        try:
            family_dir = os.path.join(self.family_dir, family_name)
            os.makedirs(family_dir, exist_ok=True)
            
            if self.photo_path.get():
                photo_ext = os.path.splitext(self.photo_path.get())[1]
                shutil.copy(self.photo_path.get(), 
                          os.path.join(family_dir, f"family_original{photo_ext}"))
            
            for face in self.faces:
                face_name = face['name'].strip()
                face_path = os.path.join(family_dir, f"{face_name}_face.jpg")
                cv2.imwrite(face_path, face['image'])
            
            messagebox.showinfo("Success", 
                f"Family {family_name} registered successfully with {len(self.faces)} members.")
            self.reset()
            self.go_back()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save faces: {str(e)}")
    
    def go_back(self):
        self.reset()
        self.parent.show_main_page()
    
    def reset(self):
        self.family_name_input.delete(0, tk.END)
        self.photo_path.delete(0, tk.END)
        self.photo_display.config(image='')
        self.name_input.delete(0, tk.END)
        self.name_input.insert(0, "Enter name for highlighted face")
        self.name_input.config(state='disabled')
        self.face_label.config(text="Upload a family photo to start")
        self.faces = []
        self.current_face_index = 0
        self.original_image = None
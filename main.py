import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import platform
import warnings
from register_page import RegisterPage
from login_page import LoginPage
from status_page import StatusPage

# Add platform specific fixes
if platform.system() == 'Linux':  # For Raspberry Pi
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software rendering

# Filter numpy warnings
warnings.filterwarnings('ignore', category=FutureWarning)

FAMILY_DIR = "family_photos"
SAFE_LOG_FILE = "safe_members.txt"

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UnlockX")
        self.geometry("1366x768")
        
        # Configure style
        style = ttk.Style()
        style.configure('Main.TButton', 
                       padding=10, 
                       font=('Helvetica', 12))
        
        # Main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(expand=True, fill='both')
        
        # Initialize pages
        self.current_page = None
        self.create_main_page()
        self.register_page = RegisterPage(self, FAMILY_DIR)
        self.login_page = LoginPage(self, FAMILY_DIR, SAFE_LOG_FILE)
        self.status_page = StatusPage(self, FAMILY_DIR, SAFE_LOG_FILE)
        
        # Show main page initially
        self.show_main_page()
    
    def create_main_page(self):
        self.main_page = ttk.Frame(self)
        
        # Logo and title
        logo_frame = ttk.Frame(self.main_page)
        logo_frame.pack(expand=True, fill='both', pady=20)
        
        try:
            logo_img = Image.open("logo/unlockx.png")
            logo_img = logo_img.resize((100, 100), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(logo_frame, image=self.logo_photo)
        except:
            logo_label = ttk.Label(logo_frame, text="👤", font=('Helvetica', 48))
        logo_label.pack()
        
        title_label = ttk.Label(logo_frame, 
                               text="UnlockX",
                               font=('Helvetica', 24, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(logo_frame,
                                 text="Lightweight & Secure Face Recognition System",
                                 font=('Helvetica', 12))
        subtitle_label.pack()
        
        # Buttons
        button_frame = ttk.Frame(self.main_page)
        button_frame.pack(expand=True, pady=20)
        
        ttk.Button(button_frame,
                  text="Register Family",
                  style='Main.TButton',
                  command=self.show_register_page).pack(pady=10)
        
        ttk.Button(button_frame,
                  text="Login with Face ID",
                  style='Main.TButton',
                  command=self.show_login_page).pack(pady=10)
        
        ttk.Button(button_frame,
                  text="Family Reunification Status",
                  style='Main.TButton',
                  command=self.show_status_page).pack(pady=10)
    
    def show_main_page(self):
        if self.current_page:
            self.current_page.pack_forget()
        self.main_page.pack(expand=True, fill='both', padx=50, pady=50)
        self.current_page = self.main_page
        self.title("UnlockX")
    
    def show_register_page(self):
        if self.current_page:
            self.current_page.pack_forget()
        self.register_page.pack(expand=True, fill='both', padx=20, pady=20)
        self.current_page = self.register_page
        self.title("Register Family | UnlockX")
    
    def show_login_page(self):
        if self.current_page:
            self.current_page.pack_forget()
        self.login_page.pack(expand=True, fill='both', padx=20, pady=20)
        self.current_page = self.login_page
        self.title("Login | UnlockX")
        self.login_page.start_camera()
    
    def show_status_page(self):
        if self.current_page:
            self.current_page.pack_forget()
        self.status_page.pack(expand=True, fill='both', padx=20, pady=20)
        self.current_page = self.status_page
        self.title("Family Status | UnlockX")
        self.status_page.refresh_status()

if __name__ == "__main__":
    # Create required directories
    os.makedirs(FAMILY_DIR, exist_ok=True)
    if not os.path.exists(SAFE_LOG_FILE):
        open(SAFE_LOG_FILE, 'a').close()
    
    # Start application
    app = MainWindow()
    app.mainloop()


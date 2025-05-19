import tkinter as tk
from tkinter import ttk
import os

class StatusPage(ttk.Frame):
    def __init__(self, parent, family_dir, safe_log_file):
        super().__init__(parent)
        self.parent = parent
        self.family_dir = family_dir
        self.safe_log_file = safe_log_file
        
        self.create_widgets()
        
    def create_widgets(self):
        # Add search
        search_frame = ttk.Frame(self)
        search_frame.pack(fill='x', padx=10, pady=5)
        
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, 
                               textvariable=search_var,
                               font=('Helvetica', 10))
        search_entry.pack(fill='x', expand=True)
        search_entry.insert(0, "Search by family name...")
        
        # Status container with scrollbar
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient='vertical', 
                                command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.load_status()
        
        self.canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', pady=10)
        
        refresh_btn = ttk.Button(button_frame, 
                               text="Refresh",
                               command=self.refresh_status)
        refresh_btn.pack(side='left', padx=5)
        
        back_btn = ttk.Button(button_frame, 
                            text="Back to Main",
                            command=self.go_back)
        back_btn.pack(side='right', padx=5)

    def load_status(self):
        # Clear previous status
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        safe_members = set()
        if os.path.exists(self.safe_log_file):
            with open(self.safe_log_file, 'r') as f:
                safe_members = {line.strip().split(',')[0] for line in f}
        
        total_members = 0
        total_safe = 0
        
        if os.path.exists(self.family_dir):
            for family in os.listdir(self.family_dir):
                family_path = os.path.join(self.family_dir, family)
                if not os.path.isdir(family_path):
                    continue
                    
                face_files = [f for f in os.listdir(family_path) 
                            if f.endswith('_face.jpg')]
                if not face_files:
                    continue
                    
                members = {os.path.splitext(f)[0][:-5] for f in face_files}
                safe = {m for m in members if m in safe_members}
                missing = members - safe
                
                # Create family card
                card = ttk.Frame(self.scrollable_frame, relief='solid', borderwidth=1)
                card.pack(fill='x', padx=5, pady=5)
                
                ttk.Label(card, 
                         text=f"Family: {family}",
                         font=('Helvetica', 12, 'bold')).pack(anchor='w')
                
                if safe:
                    ttk.Label(card,
                             text="✓ Safe Members: " + ", ".join(safe),
                             foreground='green').pack(anchor='w')
                
                if missing:
                    ttk.Label(card,
                             text="⚠ Missing Members: " + ", ".join(missing),
                             foreground='red').pack(anchor='w')
                
                total_members += len(members)
                total_safe += len(safe)
        
        if total_members > 0:
            percentage = (total_safe/total_members*100)
            summary = f"Total: {total_safe} of {total_members} members found safe ({percentage:.1f}%)"
            ttk.Label(self.scrollable_frame, 
                     text=summary,
                     font=('Helvetica', 10, 'bold')).pack()
    
    def refresh_status(self):
        self.load_status()
    
    def go_back(self):
        self.parent.show_main_page()
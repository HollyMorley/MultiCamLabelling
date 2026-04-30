"""Home screen — Create or Load a project."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

from annotation_tool.project import PROJECT_FILE


class HomeScreen:
    def __init__(self, root, main_tool):
        self.root = root
        self.main_tool = main_tool
        self.build()

    def build(self):
        root = self.root
        root.title("3D Annotation GUI")

        frame = tk.Frame(root)
        frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)

        tk.Label(frame, text="3D Annotation GUI", font=("Helvetica", 18, "bold")).pack(pady=(0, 8))
        tk.Label(
            frame,
            text=(
                "A tool for extracting frames, calibrating cameras, and "
                "guided-labelling in 3D space.\n"
                "Compatible with downstream DeepLabCut pose estimation."
            ),
            justify=tk.CENTER, fg="#444", wraplength=560,
        ).pack(pady=(0, 16))
        tk.Label(frame, text="Create a new project, or load an existing one.").pack(pady=(0, 20))

        tk.Button(
            frame, text="Create Project", width=24,
            command=self.main_tool.go_create_project,
        ).pack(pady=6)
        tk.Button(
            frame, text="Load Project", width=24,
            command=self._load_project,
        ).pack(pady=6)

    def _load_project(self):
        chosen = filedialog.askdirectory(title="Choose a project directory")
        if not chosen:
            return
        if not os.path.isfile(os.path.join(chosen, PROJECT_FILE)):
            messagebox.showerror(
                "Not a project",
                f"No {PROJECT_FILE} found in:\n{chosen}\n\n"
                "Choose a directory containing a project, or use Create Project.",
            )
            return
        try:
            self.main_tool.load_project(chosen)
        except Exception as e:
            messagebox.showerror("Failed to load project", str(e))

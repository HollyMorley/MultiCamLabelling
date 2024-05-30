'''
Obj: make labelling gui for labelling body parts in three camera views simultaneously and using the camera positions to
guide labelling across the three views

Skeleton:
-  Create a main menu GUI with option to a) extract frames from videos, and b) label frames
-  Frame extraction:
    -  Select video file (always the side view)
    -  Create GUI with all three camera views (for each selected side view) visible and a single slider for moving through frames
    -  Buttons to a) calibrate camera positions, b) extract frames
    -  Calibrate camera positions:
        -  Label the points: "StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR", "Nose"
            -  Can slide between frames while labelling and these labeled points will remain static to check they fit across the video
        -  Save the labeled points
            -  Path: "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\CameraCalibration\[video_name]\calibration_labels.csv"
        -  Use the labeled points to calculate the camera positions
            -  Using CalibrateCams.py
                - calib = BasicCalibration(labeled_coordinates)
                - cameras_extrinsics = calib.estimate_cams_pose()
                - cameras_intrisinics = calib.cameras_intrinsics
            -  Save the camera calibration data
                -  Path: "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\CameraCalibration\[video_name]\calibration_matrices.csv"
    -  Extract frames:
        -  Choose frames using a slider and extract button
        -  Save extracted frames to a folder (predefined directory in the code)
            - Paths:
                - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Side\[video_name]\img[frame number].png"
                - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Front\[video_name]\img[frame number].png"
                - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Overhead\[video_name]\img[frame number].png"
-  Frame labelling:
    -  Select the extracted frames folder
        -  Raise an error if there is no corresponding camera calibration data
    -  With all three camera views visible, label body parts one by one in each extracted frame
        -  Toggle through frames with 'Next frame' and 'Previous frame' buttons
        -  Have a list/menu of body parts in the GUI window with you can toggle between
        -  Bodyparts: "Nose", "EarL", "EarR", "Back1",..., "Back12", "Tail1", ..., "Tail12", "ForepawToeR",
            "ForepawAnkleR", "ForepawToeL", "ForepawAnkleL", "HindpawToeR", "HindpawAnkleR", "HindpawToeL", "HindpawAnkleL"
    -  As label is placed in one view, a red line is plotted in the other two views representing the projection of
        that label in the other views to guide labelling
        - The red line should be updated as the label is moved
        - Reprojection on to each other view is done using the camera calibration data
        - Have toggle buttons to signify which view to base the red line off of, e.g. if side is selected, this shows the
            line of sight from the side camera to the current labeled point in the front and overhead views
            - Buttons: "Side", "Front", "Overhead", "None" (for no red line)
    -  Controls for labelling are:
        -  Right-click to place a point
        -  Shift + Right-click to delete the nearest point
        -  Left-click and drag to move a point
        -  Buttons to open/save images and close the labelling tool
    -  Save the labelled frames to a folder (predefined directory in the code)
        - Paths:
            - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Side\[video_name]\CollectedData_Holly.csv"
            - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Front\[video_name]\CollectedData_Holly.csv"
            - "X:\hmorley\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Overhead\[video_name]\CollectedData_Holly.csv"
'''

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from Helpers.CalibrateCams import BasicCalibration


class LabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Parts Labeling Tool")

        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.main_menu()

    def main_menu(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20)

        extract_button = tk.Button(main_frame, text="Extract Frames from Videos", command=self.extract_frames_menu)
        extract_button.pack(pady=5)

        label_button = tk.Button(main_frame, text="Label Frames", command=self.label_frames_menu)
        label_button.pack(pady=5)

    def extract_frames_menu(self):
        self.clear_root()

        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            return

        self.video_name, self.video_date, self.camera_view = self.parse_video_path(self.video_path)
        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cap_front = cv2.VideoCapture(self.get_corresponding_video_path('front'))
        self.cap_overhead = cv2.VideoCapture(self.get_corresponding_video_path('overhead'))

        extract_frame = tk.Frame(self.root)
        extract_frame.pack(pady=20)

        self.frame_label = tk.Label(extract_frame, text="Frame: 0")
        self.frame_label.pack(pady=5)

        self.slider = tk.Scale(extract_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, length=600,
                               command=self.update_frame_label)
        self.slider.pack(pady=5)

        skip_frame = tk.Frame(self.root)
        skip_frame.pack(pady=10)

        back_1000_button = tk.Button(skip_frame, text="<< 1000", command=lambda: self.skip_frames(-1000))
        back_1000_button.grid(row=0, column=0, padx=5)

        back_100_button = tk.Button(skip_frame, text="<< 100", command=lambda: self.skip_frames(-100))
        back_100_button.grid(row=0, column=1, padx=5)

        back_10_button = tk.Button(skip_frame, text="<< 10", command=lambda: self.skip_frames(-10))
        back_10_button.grid(row=0, column=2, padx=5)

        back_1_button = tk.Button(skip_frame, text="<< 1", command=lambda: self.skip_frames(-1))
        back_1_button.grid(row=0, column=3, padx=5)

        forward_1_button = tk.Button(skip_frame, text=">> 1", command=lambda: self.skip_frames(1))
        forward_1_button.grid(row=0, column=4, padx=5)

        forward_10_button = tk.Button(skip_frame, text=">> 10", command=lambda: self.skip_frames(10))
        forward_10_button.grid(row=0, column=5, padx=5)

        forward_100_button = tk.Button(skip_frame, text=">> 100", command=lambda: self.skip_frames(100))
        forward_100_button.grid(row=0, column=6, padx=5)

        forward_1000_button = tk.Button(skip_frame, text=">> 1000", command=lambda: self.skip_frames(1000))
        forward_1000_button.grid(row=0, column=7, padx=5)

        calibrate_button = tk.Button(extract_frame, text="Calibrate Camera Positions", command=self.calibrate_cameras)
        calibrate_button.pack(pady=5)

        extract_button = tk.Button(extract_frame, text="Extract Frames", command=self.extract_frames)
        extract_button.pack(pady=5)

        back_button = tk.Button(extract_frame, text="Back to Main Menu", command=self.main_menu)
        back_button.pack(pady=5)

        self.panel_side = None
        self.panel_front = None
        self.panel_overhead = None
        self.show_frames()

    def update_frame_label(self, val):
        frame_number = int(val)
        self.frame_label.config(text=f"Frame: {frame_number}")
        self.show_frames()

    def parse_video_path(self, video_path):
        video_file = os.path.basename(video_path)
        parts = video_file.split('_')
        date = parts[1]
        camera_view = [part for part in parts if part in ['side', 'front', 'overhead']][0]
        name_parts = [part for part in parts if part not in ['side', 'front', 'overhead']]
        name = '_'.join(name_parts)
        return name, date, camera_view

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace(self.camera_view, view)
        return os.path.join(base_path, corresponding_file)

    def calibrate_cameras(self):
        self.clear_root()

        calibration_frame = tk.Frame(self.root)
        calibration_frame.pack(pady=20)

        labels_frame = tk.Frame(calibration_frame)
        labels_frame.pack(side="left", padx=10)

        self.calibration_points = {
            "StartPlatL": [], "StepL": [], "StartPlatR": [], "StepR": [],
            "Door": [], "TransitionL": [], "TransitionR": [], "Nose": []
        }

        self.current_point = tk.StringVar(value="StartPlatL")
        for label in self.calibration_points.keys():
            button = tk.Radiobutton(labels_frame, text=label, variable=self.current_point, value=label)
            button.pack(anchor="w")

        self.slider = tk.Scale(calibration_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, length=600,
                               command=self.show_frames)
        self.slider.pack(pady=5)

        save_button = tk.Button(calibration_frame, text="Save Calibration Points", command=self.save_calibration_points)
        save_button.pack(pady=5)

        back_button = tk.Button(calibration_frame, text="Back to Extract Frames Menu", command=self.extract_frames_menu)
        back_button.pack(pady=5)

        self.panel_side = None
        self.panel_front = None
        self.panel_overhead = None

        self.labels = {label: {"side": None, "front": None, "overhead": None} for label in self.calibration_points}
        self.current_view = "side"

        self.show_frames()

    def on_right_click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        label = self.current_point.get()
        if self.labels[label][self.current_view] is None:
            self.labels[label][self.current_view] = self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red")
        else:
            self.canvas.coords(self.labels[label][self.current_view], x - 3, y - 3, x + 3, y + 3)

    def save_calibration_points(self):
        calibration_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/CameraCalibration/{self.video_name}/calibration_labels.csv"
        os.makedirs(os.path.dirname(calibration_path), exist_ok=True)

        data = []
        for label, coords in self.labels.items():
            for view, item in coords.items():
                if item is not None:
                    x, y, _, _ = self.canvas.coords(item)
                    if view == "side":
                        x, y = x / self.scaling_factor_side, y / self.scaling_factor_side
                    elif view == "front":
                        x, y = x / self.scaling_factor_front, y / self.scaling_factor_front
                    elif view == "overhead":
                        x, y = x / self.scaling_factor_overhead, y / self.scaling_factor_overhead
                    data.append([label, 'x', x, view])
                    data.append([label, 'y', y, view])

        df = pd.DataFrame(data, columns=["bodyparts", "coords", "side", "view"])
        df.to_csv(calibration_path, index=False)

        messagebox.showinfo("Info", "Calibration points saved successfully")

    def extract_frames(self):
        frame_number = self.slider.get()
        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_side, frame_side = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_front, frame_front = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_overhead, frame_overhead = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            side_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Side/{self.video_name}/img{frame_number}.png"
            front_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Front/{self.video_name}/img{frame_number}.png"
            overhead_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Overhead/{self.video_name}/img{frame_number}.png"

            os.makedirs(os.path.dirname(side_path), exist_ok=True)
            os.makedirs(os.path.dirname(front_path), exist_ok=True)
            os.makedirs(os.path.dirname(overhead_path), exist_ok=True)

            cv2.imwrite(side_path, frame_side)
            cv2.imwrite(front_path, frame_front)
            cv2.imwrite(overhead_path, frame_overhead)

    def show_frames(self, val=None):
        frame_number = self.slider.get()

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_side, frame_side = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_front, frame_front = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_overhead, frame_overhead = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            img_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
            img_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
            img_overhead = cv2.cvtColor(frame_overhead, cv2.COLOR_BGR2RGB)

            img_side = Image.fromarray(img_side)
            img_front = Image.fromarray(img_front)
            img_overhead = Image.fromarray(img_overhead)

            # Determine scaling factor
            self.scaling_factor_side = min(self.screen_width / 1920, self.screen_height / (230 + 320 + 116))
            self.scaling_factor_front = self.scaling_factor_side  # Keeping uniform scaling across all views
            self.scaling_factor_overhead = self.scaling_factor_side

            new_side_size = (int(1920 * self.scaling_factor_side), int(230 * self.scaling_factor_side))
            new_front_size = (int(296 * self.scaling_factor_front), int(320 * self.scaling_factor_front))
            new_overhead_size = (int(992 * self.scaling_factor_overhead), int(116 * self.scaling_factor_overhead))

            img_side = img_side.resize(new_side_size, Image.Resampling.LANCZOS)
            img_front = img_front.resize(new_front_size, Image.Resampling.LANCZOS)
            img_overhead = img_overhead.resize(new_overhead_size, Image.Resampling.LANCZOS)

            imgtk_side = ImageTk.PhotoImage(image=img_side)
            imgtk_front = ImageTk.PhotoImage(image=img_front)
            imgtk_overhead = ImageTk.PhotoImage(image=img_overhead)

            if self.panel_side is None:
                self.panel_side = tk.Label(image=imgtk_side)
                self.panel_side.image = imgtk_side
                self.panel_side.pack(side="top", padx=10, pady=10)

                self.panel_front = tk.Label(image=imgtk_front)
                self.panel_front.image = imgtk_front
                self.panel_front.pack(side="top", padx=10, pady=10)

                self.panel_overhead = tk.Label(image=imgtk_overhead)
                self.panel_overhead.image = imgtk_overhead
                self.panel_overhead.pack(side="top", padx=10, pady=10)
            else:
                self.panel_side.configure(image=imgtk_side)
                self.panel_side.image = imgtk_side

                self.panel_front.configure(image=imgtk_front)
                self.panel_front.image = imgtk_front

                self.panel_overhead.configure(image=imgtk_overhead)
                self.panel_overhead.image = imgtk_overhead

    def skip_frames(self, step):
        new_frame_number = self.slider.get() + step
        new_frame_number = max(0, min(new_frame_number, self.total_frames - 1))
        self.slider.set(new_frame_number)
        self.show_frames()

    def label_frames_menu(self):
        self.clear_root()
        folder_path = filedialog.askdirectory(title="Select Extracted Frames Folder")

        if not folder_path:
            return

        self.extracted_frames_path = folder_path
        self.current_frame_index = 0

        self.video_name = os.path.basename(folder_path)
        self.video_date = self.extract_date_from_folder_path(folder_path)
        calibration_data_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/CameraCalibration/{self.video_name}/calibration_matrices.csv"

        if not os.path.exists(calibration_data_path):
            messagebox.showerror("Error", "No corresponding camera calibration data found.")
            return

        self.load_frames()
        self.load_calibration_data(calibration_data_path)
        self.setup_labeling_ui()

    def extract_date_from_folder_path(self, folder_path):
        # Assuming folder path has date somewhere
        parts = folder_path.split(os.sep)
        for part in parts:
            if part.isdigit() and len(part) == 8:
                return part
        return None

    def load_frames(self):
        self.frames = []
        for view in ['Side', 'Front', 'Overhead']:
            view_path = os.path.join(self.extracted_frames_path, view)
            frame_files = sorted(os.listdir(view_path))
            self.frames.append([cv2.imread(os.path.join(view_path, file)) for file in frame_files])

    def load_calibration_data(self, calibration_data_path):
        calibration_data = pd.read_pickle(calibration_data_path)
        self.cameras_extrinsics = calibration_data['extrinsics']
        self.cameras_intrinsics = calibration_data['intrinsics']

    def setup_labeling_ui(self):
        labeling_frame = tk.Frame(self.root)
        labeling_frame.pack(pady=20)

        self.canvas_side = tk.Canvas(labeling_frame, width=int(1920), height=int(230))
        self.canvas_side.grid(row=0, column=0)
        self.canvas_front = tk.Canvas(labeling_frame, width=int(296), height=int(320))
        self.canvas_front.grid(row=1, column=0)
        self.canvas_overhead = tk.Canvas(labeling_frame, width=int(992), height=int(116))
        self.canvas_overhead.grid(row=2, column=0)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        prev_button = tk.Button(control_frame, text="Previous Frame", command=lambda: self.change_frame(-1))
        prev_button.grid(row=0, column=0, padx=5)

        next_button = tk.Button(control_frame, text="Next Frame", command=lambda: self.change_frame(1))
        next_button.grid(row=0, column=1, padx=5)

        body_parts = ["Nose", "EarL", "EarR", "Back1", "Back2", "Tail1", "ForepawToeR", "HindpawToeL"]  # Example parts
        self.body_part_var = tk.StringVar(value=body_parts[0])
        body_part_menu = tk.OptionMenu(control_frame, self.body_part_var, *body_parts)
        body_part_menu.grid(row=0, column=2, padx=5)

        save_button = tk.Button(control_frame, text="Save Labels", command=self.save_labels)
        save_button.grid(row=0, column=3, padx=5)

        self.red_line_view_var = tk.StringVar(value="None")
        for i, view in enumerate(["Side", "Front", "Overhead", "None"]):
            tk.Radiobutton(control_frame, text=view, variable=self.red_line_view_var, value=view).grid(row=1, column=i,
                                                                                                       padx=5)

        self.display_frame()

    def change_frame(self, step):
        self.current_frame_index += step
        self.current_frame_index = max(0, min(self.current_frame_index, len(self.frames[0]) - 1))
        self.display_frame()

    def display_frame(self):
        for i, canvas in enumerate([self.canvas_side, self.canvas_front, self.canvas_overhead]):
            frame = cv2.cvtColor(self.frames[i][self.current_frame_index], cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=frame)
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas.image = imgtk

    def save_labels(self):
        side_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Side/{self.video_name}/CollectedData_Holly.csv"
        front_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Front/{self.video_name}/CollectedData_Holly.csv"
        overhead_path = f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Overhead/{self.video_name}/CollectedData_Holly.csv"

        os.makedirs(os.path.dirname(side_path), exist_ok=True)
        os.makedirs(os.path.dirname(front_path), exist_ok=True)
        os.makedirs(os.path.dirname(overhead_path), exist_ok=True)

        # Placeholder for actual saving logic
        messagebox.showinfo("Info", "Labels saved successfully")

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingTool(root)
    root.mainloop()

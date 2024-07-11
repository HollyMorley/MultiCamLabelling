import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import bisect
import pandas as pd
import numpy as np
import mpld3
from mpld3 import plugins
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from PIL import Image, ImageTk, ImageEnhance
import h5py
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import time
from functools import lru_cache

import Helpers.MultiCamLabelling_config as config
from Helpers.CalibrateCams import BasicCalibration


def get_video_name_with_view(video_name, view):
    split_video_name = video_name.split('_')
    split_video_name.insert(-1, view)
    return '_'.join(split_video_name)

class MainTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Parts Labeling Tool")

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.main_menu()

    def main_menu(self):
        self.clear_root()
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20)

        extract_button = tk.Button(main_frame, text="Extract Frames from Videos", command=self.extract_frames_menu)
        extract_button.pack(pady=5)

        calibrate_button = tk.Button(main_frame, text="Calibrate Camera Positions", command=self.calibrate_cameras_menu)
        calibrate_button.pack(pady=5)

        label_button = tk.Button(main_frame, text="Label Frames", command=self.label_frames_menu)
        label_button.pack(pady=5)

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def extract_frames_menu(self):
        self.clear_root()
        ExtractFramesTool(self.root, self)

    def calibrate_cameras_menu(self):
        self.clear_root()
        CalibrateCamerasTool(self.root, self)

    def label_frames_menu(self):
        self.clear_root()
        LabelFramesTool(self.root, self)


class ExtractFramesTool:
    def __init__(self, root, main_tool):
        self.root = root
        self.main_tool = main_tool
        self.video_path = ""
        self.video_name = ""
        self.video_date = ""
        self.camera_view = ""
        self.cap_side = None
        self.cap_front = None
        self.cap_overhead = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.contrast_var = tk.DoubleVar(value=config.DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=config.DEFAULT_BRIGHTNESS)

        self.extract_frames()

    def extract_frames(self):
        self.main_tool.clear_root()

        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            self.main_tool.main_menu()
            return

        self.video_name, self.video_date, self.camera_view = self.parse_video_path(self.video_path)
        self.video_name_stripped = '_'.join(self.video_name.split('_')[:-1])  # Remove the camera view part
        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        self.cap_front = cv2.VideoCapture(self.get_corresponding_video_path('front'))
        self.cap_overhead = cv2.VideoCapture(self.get_corresponding_video_path('overhead'))

        # Print the total number of frames for each video
        total_frames_side = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_front = int(self.cap_front.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_overhead = int(self.cap_overhead.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Total frames - Side: {total_frames_side}, Front: {total_frames_front}, Overhead: {total_frames_overhead}")

        # Load timestamps
        timestamps_side = self.zero_timestamps(self.load_timestamps('side'))
        timestamps_front = self.zero_timestamps(self.load_timestamps('front'))
        timestamps_overhead = self.zero_timestamps(self.load_timestamps('overhead'))

        # Adjust timestamps to offset the drift in front and overhead cameras (where frame rates are different)
        timestamps_front_adj = self.adjust_timestamps(timestamps_side, timestamps_front)
        timestamps_overhead_adj = self.adjust_timestamps(timestamps_side, timestamps_overhead)
        timestamps_side_adj = timestamps_side['Timestamp'].astype(float) # adjust so compatible with scaled front and overhead

        # Extract matching frames
        self.match_frames(timestamps_side_adj, timestamps_front_adj, timestamps_overhead_adj)

        self.show_frames_extraction()

    def load_timestamps(self, view):
        video_name = '_'.join(self.video_name.split('_')[:-1])  # Remove the camera view part
        video_number = self.video_name.split('_')[-1]
        timestamp_file = video_name + f"_{view}_{video_number}_Timestamps.csv"
        timestamp_path = os.path.join(os.path.dirname(self.video_path), timestamp_file)
        timestamps = pd.read_csv(timestamp_path)
        return timestamps

    def zero_timestamps(self, timestamps):
        timestamps['Timestamp'] = timestamps['Timestamp'] - timestamps['Timestamp'][0]
        return timestamps

    def adjust_timestamps(self, side_timestamps, other_timestamps):
        mask = other_timestamps['Timestamp'].diff() < 4.045e+6
        other_timestamps_single_frame = other_timestamps[mask]
        side_timestamps_single_frame = side_timestamps[mask]
        diff = other_timestamps_single_frame['Timestamp'] - side_timestamps_single_frame['Timestamp']

        # find the best fit line for the lower half of the data by straightning the line
        model = LinearRegression().fit(side_timestamps_single_frame['Timestamp'].values.reshape(-1, 1), diff.values)
        slope = model.coef_[0]
        intercept = model.intercept_
        straightened_diff = diff - (slope * side_timestamps_single_frame['Timestamp'] + intercept)
        correct_diff_idx = np.where(straightened_diff < straightened_diff.mean())

        model_true = LinearRegression().fit(side_timestamps_single_frame['Timestamp'].values[correct_diff_idx].reshape(-1, 1), diff.values[correct_diff_idx])
        slope_true = model_true.coef_[0]
        intercept_true = model_true.intercept_
        adjusted_timestamps = other_timestamps['Timestamp'] - (slope_true * other_timestamps['Timestamp'] + intercept_true)
        return adjusted_timestamps

    def match_frames(self, timestamps_side, timestamps_front, timestamps_overhead):
        buffer_ns = int(4.04e+6)  # Frame duration in nanoseconds

        # Ensure the timestamps are sorted
        timestamps_side = timestamps_side.sort_values().reset_index(drop=True)
        timestamps_front = timestamps_front.sort_values().reset_index(drop=True)
        timestamps_overhead = timestamps_overhead.sort_values().reset_index(drop=True)

        # Convert timestamps to DataFrame for merging
        side_df = pd.DataFrame({'Timestamp': timestamps_side, 'Frame_number_side': range(len(timestamps_side))})
        front_df = pd.DataFrame({'Timestamp': timestamps_front, 'Frame_number_front': range(len(timestamps_front))})
        overhead_df = pd.DataFrame(
            {'Timestamp': timestamps_overhead, 'Frame_number_overhead': range(len(timestamps_overhead))})

        # Perform asof merge to find the closest matching frames within the buffer
        matched_front = pd.merge_asof(side_df, front_df, on='Timestamp', direction='nearest', tolerance=buffer_ns,
                                      suffixes=('_side', '_front'))
        matched_all = pd.merge_asof(matched_front, overhead_df, on='Timestamp', direction='nearest',
                                    tolerance=buffer_ns, suffixes=('_side', '_overhead'))

        # Check column names
        print(matched_all.columns)

        # Handle NaNs explicitly by setting unmatched frames to -1
        matched_frames = matched_all[['Frame_number_side', 'Frame_number_front', 'Frame_number_overhead']].applymap(
            lambda x: int(x) if pd.notnull(x) else -1).values.tolist()

        self.matched_frames = matched_frames

    def show_frames_extraction(self):
        self.main_tool.clear_root()

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, pady=10)

        self.slider = tk.Scale(control_frame, from_=0, to=len(self.matched_frames) - 1, orient=tk.HORIZONTAL,
                               length=600,
                               command=self.update_frame_label)
        self.slider.pack(side=tk.LEFT, padx=5)

        self.frame_label = tk.Label(control_frame, text=f"Frame: {self.matched_frames[0][0]}")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        skip_frame = tk.Frame(self.root)
        skip_frame.pack(side=tk.TOP, pady=10)
        self.add_skip_buttons(skip_frame)

        control_frame_right = tk.Frame(self.root)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

        extract_button = tk.Button(control_frame_right, text="Extract Frames", command=self.save_extracted_frames)
        extract_button.pack(pady=5)

        back_button = tk.Button(control_frame_right, text="Back to Main Menu", command=self.main_tool.main_menu)
        back_button.pack(pady=5)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.display_frame(0)

    def add_skip_buttons(self, parent):
        buttons = [
            ("<< 1000", -1000), ("<< 100", -100), ("<< 10", -10), ("<< 1", -1),
            (">> 1", 1), (">> 10", 10), (">> 100", 100), (">> 1000", 1000)
        ]
        for i, (text, step) in enumerate(buttons):
            button = tk.Button(parent, text=text, command=lambda s=step: self.skip_frames(s))
            button.grid(row=0, column=i, padx=5)

    def skip_frames(self, step):
        new_frame_number = self.current_frame_index + step
        new_frame_number = max(0, min(new_frame_number, len(self.matched_frames) - 1))
        self.slider.set(new_frame_number)
        self.display_frame(new_frame_number)

    def display_frame(self, index):
        self.current_frame_index = index
        frame_side, frame_front, frame_overhead = self.matched_frames[index]

        # Read frames from respective positions
        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_side)
        ret_side, frame_side_img = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_front)
        ret_front, frame_front_img = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_overhead)
        ret_overhead, frame_overhead_img = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            frame_side_img = self.apply_contrast_brightness(frame_side_img)
            frame_front_img = self.apply_contrast_brightness(frame_front_img)
            frame_overhead_img = self.apply_contrast_brightness(frame_overhead_img)

            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

            self.axs[0].imshow(cv2.cvtColor(frame_side_img, cv2.COLOR_BGR2RGB))
            self.axs[1].imshow(cv2.cvtColor(frame_front_img, cv2.COLOR_BGR2RGB))
            self.axs[2].imshow(cv2.cvtColor(frame_overhead_img, cv2.COLOR_BGR2RGB))

            self.axs[0].set_title('Side View')
            self.axs[1].set_title('Front View')
            self.axs[2].set_title('Overhead View')

            self.canvas.draw()

    def update_frame_label(self, val):
        index = int(val)
        self.frame_label.config(text=f"Frame: {self.matched_frames[index][0]}")
        self.display_frame(index)

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        img_contrast = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_brightness = enhancer.enhance(brightness)
        return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)

    def parse_video_path(self, video_path):
        video_file = os.path.basename(video_path)
        parts = video_file.split('_')
        date = parts[1]
        camera_view = [part for part in parts if part in ['side', 'front', 'overhead']][0]
        name_parts = [part for part in parts if part not in ['side', 'front', 'overhead']]
        name = '_'.join(name_parts).replace('.avi', '')  # Remove .avi extension if present
        return name, date, camera_view

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace(self.camera_view, view).replace('.avi', '')
        return os.path.join(base_path, f"{corresponding_file}.avi")

    def save_extracted_frames(self):
        frame_side, frame_front, frame_overhead = self.matched_frames[self.current_frame_index]

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_side)
        ret_side, frame_side_img = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_front)
        ret_front, frame_front_img = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_overhead)
        ret_overhead, frame_overhead_img = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            video_names = {view: get_video_name_with_view(self.video_name, view) for view in
                           ['side', 'front', 'overhead']}

            side_path = os.path.join(config.FRAME_SAVE_PATH_TEMPLATE["side"].format(video_name=video_names['side']),
                                     f"img{frame_side}.png")
            front_path = os.path.join(config.FRAME_SAVE_PATH_TEMPLATE["front"].format(video_name=video_names['front']),
                                      f"img{frame_front}.png")
            overhead_path = os.path.join(
                config.FRAME_SAVE_PATH_TEMPLATE["overhead"].format(video_name=video_names['overhead']),
                f"img{frame_overhead}.png")

            os.makedirs(os.path.dirname(side_path), exist_ok=True)
            os.makedirs(os.path.dirname(front_path), exist_ok=True)
            os.makedirs(os.path.dirname(overhead_path), exist_ok=True)

            cv2.imwrite(side_path, frame_side_img)
            cv2.imwrite(front_path, frame_front_img)
            cv2.imwrite(overhead_path, frame_overhead_img)
            messagebox.showinfo("Info", "Frames saved successfully")


class CalibrateCamerasTool:
    def __init__(self, root, main_tool):
        self.root = root
        self.main_tool = main_tool
        self.video_path = ""
        self.video_name = ""
        self.video_date = ""
        self.camera_view = ""
        self.cap_side = None
        self.cap_front = None
        self.cap_overhead = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.contrast_var = tk.DoubleVar(value=config.DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=config.DEFAULT_BRIGHTNESS)
        self.marker_size_var = tk.DoubleVar(value=config.DEFAULT_MARKER_SIZE)
        self.mode = 'calibration'
        self.matched_frames = []

        self.calibration_points_static = {}
        self.dragging_point = None
        self.crosshair_lines = []
        self.panning = False
        self.pan_start = None

        self.labels = config.CALIBRATION_LABELS
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])
        self.current_view = tk.StringVar(value="side")

        self.calibrate_cameras_menu()

    def calibrate_cameras_menu(self):
        self.main_tool.clear_root()

        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            self.main_tool.main_menu()
            return

        self.video_name, self.video_date, self.camera_view = self.parse_video_path(self.video_path)
        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        self.cap_front = cv2.VideoCapture(self.get_corresponding_video_path('front'))
        self.cap_overhead = cv2.VideoCapture(self.get_corresponding_video_path('overhead'))

        # Load timestamps
        timestamps_side = self.zero_timestamps(self.load_timestamps('side'))
        timestamps_front = self.zero_timestamps(self.load_timestamps('front'))
        timestamps_overhead = self.zero_timestamps(self.load_timestamps('overhead'))

        # Adjust timestamps to offset the drift in front and overhead cameras (where frame rates are different)
        timestamps_front_adj = self.adjust_timestamps(timestamps_side, timestamps_front)
        timestamps_overhead_adj = self.adjust_timestamps(timestamps_side, timestamps_overhead)
        timestamps_side_adj = timestamps_side['Timestamp'].astype(
            float)  # adjust so compatible with scaled front and overhead

        # Extract matching frames
        self.match_frames(timestamps_side_adj, timestamps_front_adj, timestamps_overhead_adj)

        self.calibration_file_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name)
        default_calibration_file = config.DEFAULT_CALIBRATION_FILE_PATH

        self.mode = 'calibration'

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        settings_frame = tk.Frame(control_frame)
        settings_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(settings_frame, text="Marker Size").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_MARKER_SIZE, to=config.MAX_MARKER_SIZE, orient=tk.HORIZONTAL,
                 resolution=config.MARKER_SIZE_STEP, variable=self.marker_size_var,
                 command=self.update_marker_size).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Contrast").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_CONTRAST, to=config.MAX_CONTRAST, orient=tk.HORIZONTAL,
                 resolution=config.CONTRAST_STEP, variable=self.contrast_var,
                 command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Brightness").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_BRIGHTNESS, to=config.MAX_BRIGHTNESS, orient=tk.HORIZONTAL,
                 resolution=config.BRIGHTNESS_STEP, variable=self.brightness_var,
                 command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(frame_control, text="Frame: 0")
        self.frame_label.pack()

        self.slider = tk.Scale(frame_control, from_=0, to=len(self.matched_frames) - 1, orient=tk.HORIZONTAL,
                               length=400,
                               command=self.update_frame_label)
        self.slider.pack()

        skip_frame = tk.Frame(frame_control)
        skip_frame.pack()
        self.add_skip_buttons(skip_frame)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        home_button = tk.Button(button_frame, text="Home", command=self.reset_view)
        home_button.pack(pady=5)

        save_button = tk.Button(button_frame, text="Save Calibration Points", command=self.save_calibration_points)
        save_button.pack(pady=5)

        back_button = tk.Button(button_frame, text="Back to Main Menu", command=self.main_tool.main_menu)
        back_button.pack(pady=5)

        control_frame_right = tk.Frame(main_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        for label in self.labels:
            color = self.label_colors[label]
            label_frame = tk.Frame(control_frame_right)
            label_frame.pack(fill=tk.X, pady=2)
            color_box = tk.Label(label_frame, bg=color, width=2)
            color_box.pack(side=tk.LEFT, padx=5)
            label_button = tk.Radiobutton(label_frame, text=label, variable=self.current_label, value=label,
                                          indicatoron=0, width=20)
            label_button.pack(side=tk.LEFT)

        for view in ["side", "front", "overhead"]:
            tk.Radiobutton(control_frame_right, text=view.capitalize(), variable=self.current_view, value=view).pack(
                pady=2)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("motion_notify_event", self.update_crosshair)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.calibration_points_static = {label: {"side": None, "front": None, "overhead": None} for label in
                                          self.labels}

        if os.path.exists(self.calibration_file_path):
            response = messagebox.askyesnocancel("Calibration Found",
                                                 "Calibration labels found. Do you want to load them? (Yes to load current, No to load default, Cancel to skip)")
            if response is None:
                pass
            elif response:
                self.load_calibration_points(self.calibration_file_path)
            else:
                if os.path.exists(default_calibration_file):
                    self.load_calibration_points(default_calibration_file)
                else:
                    messagebox.showinfo("Default Calibration Not Found", "Default calibration file not found.")
        else:
            if os.path.exists(default_calibration_file):
                if messagebox.askyesno("Default Calibration",
                                       "No specific calibration file found. Do you want to load the default calibration labels?"):
                    self.load_calibration_points(default_calibration_file)
            else:
                messagebox.showinfo("Default Calibration Not Found", "Default calibration file not found.")

        self.show_frames()

    def match_frames(self, timestamps_side, timestamps_front, timestamps_overhead):
        buffer_ns = int(4.04e+6)  # Frame duration in nanoseconds

        # Ensure the timestamps are sorted
        timestamps_side = timestamps_side.sort_values().reset_index(drop=True)
        timestamps_front = timestamps_front.sort_values().reset_index(drop=True)
        timestamps_overhead = timestamps_overhead.sort_values().reset_index(drop=True)

        # Convert timestamps to DataFrame for merging
        side_df = pd.DataFrame({'Timestamp': timestamps_side, 'Frame_number_side': range(len(timestamps_side))})
        front_df = pd.DataFrame({'Timestamp': timestamps_front, 'Frame_number_front': range(len(timestamps_front))})
        overhead_df = pd.DataFrame(
            {'Timestamp': timestamps_overhead, 'Frame_number_overhead': range(len(timestamps_overhead))})

        # Perform asof merge to find the closest matching frames within the buffer
        matched_front = pd.merge_asof(side_df, front_df, on='Timestamp', direction='nearest', tolerance=buffer_ns,
                                      suffixes=('_side', '_front'))
        matched_all = pd.merge_asof(matched_front, overhead_df, on='Timestamp', direction='nearest',
                                    tolerance=buffer_ns, suffixes=('_side', '_overhead'))

        # Check column names
        print(matched_all.columns)

        # Handle NaNs explicitly by setting unmatched frames to -1
        matched_frames = matched_all[['Frame_number_side', 'Frame_number_front', 'Frame_number_overhead']].applymap(
            lambda x: int(x) if pd.notnull(x) else -1).values.tolist()

        self.matched_frames = matched_frames
        print(f"Matched frames: {len(matched_frames)}")

    def load_timestamps(self, view):
        video_name = '_'.join(self.video_name.split('_')[:-1])  # Remove the camera view part
        video_number = self.video_name.split('_')[-1]
        timestamp_file = video_name + f"_{view}_{video_number}_Timestamps.csv"
        timestamp_path = os.path.join(os.path.dirname(self.video_path), timestamp_file)
        timestamps = pd.read_csv(timestamp_path)
        return timestamps

    def zero_timestamps(self, timestamps):
        timestamps['Timestamp'] = timestamps['Timestamp'] - timestamps['Timestamp'][0]
        return timestamps

    def adjust_timestamps(self, side_timestamps, other_timestamps):
        mask = other_timestamps['Timestamp'].diff() < 4.045e+6
        other_timestamps_single_frame = other_timestamps[mask]
        side_timestamps_single_frame = side_timestamps[mask]
        diff = other_timestamps_single_frame['Timestamp'] - side_timestamps_single_frame['Timestamp']

        # find the best fit line for the lower half of the data by straightening the line
        model = LinearRegression().fit(side_timestamps_single_frame['Timestamp'].values.reshape(-1, 1), diff.values)
        slope = model.coef_[0]
        intercept = model.intercept_
        straightened_diff = diff - (slope * side_timestamps_single_frame['Timestamp'] + intercept)
        correct_diff_idx = np.where(straightened_diff < straightened_diff.mean())

        model_true = LinearRegression().fit(
            side_timestamps_single_frame['Timestamp'].values[correct_diff_idx].reshape(-1, 1),
            diff.values[correct_diff_idx])
        slope_true = model_true.coef_[0]
        intercept_true = model_true.intercept_
        adjusted_timestamps = other_timestamps['Timestamp'] - (
                    slope_true * other_timestamps['Timestamp'] + intercept_true)
        return adjusted_timestamps

    def update_marker_size(self, val):
        current_xlim = [ax.get_xlim() for ax in self.axs]
        current_ylim = [ax.get_ylim() for ax in self.axs]
        self.marker_size = self.marker_size_var.get()
        self.show_frames()
        for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def load_calibration_points(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.set_index(["bodyparts", "coords"], inplace=True)
            for label in df.index.levels[0]:
                for view in ["side", "front", "overhead"]:
                    if not pd.isna(df.loc[(label, 'x'), view]):
                        x, y = df.loc[(label, 'x'), view], df.loc[(label, 'y'), view]
                        self.calibration_points_static[label][view] = self.axs[
                            ["side", "front", "overhead"].index(view)].scatter(
                            x, y, c=self.label_colors[label], s=self.marker_size_var.get() * 10, label=label
                        )
            self.canvas.draw()
            messagebox.showinfo("Info", "Calibration points loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration points: {e}")

    def update_frame_label(self, val):
        self.current_frame_index = int(val)
        self.show_frames()

    def add_skip_buttons(self, parent):
        buttons = [
            ("<< 1000", -1000), ("<< 100", -100), ("<< 10", -10), ("<< 1", -1),
            (">> 1", 1), (">> 10", 10), (">> 100", 100), (">> 1000", 1000)
        ]
        for i, (text, step) in enumerate(buttons):
            button = tk.Button(parent, text=text, command=lambda s=step: self.skip_frames(s))
            button.grid(row=0, column=i, padx=5)

    def skip_frames(self, step):
        new_frame_number = self.current_frame_index + step
        new_frame_number = max(0, min(new_frame_number, self.total_frames - 1))
        self.current_frame_index = new_frame_number
        self.slider.set(new_frame_number)
        self.frame_label.config(text=f"Frame: {new_frame_number}/{self.total_frames - 1}")
        self.show_frames()

    def parse_video_path(self, video_path):
        video_file = os.path.basename(video_path)
        parts = video_file.split('_')
        date = parts[1]
        camera_view = [part for part in parts if part in ['side', 'front', 'overhead']][0]
        name_parts = [part for part in parts if part not in ['side', 'front', 'overhead']]
        name = '_'.join(name_parts).replace('.avi', '')  # Remove .avi extension if present
        return name, date, camera_view

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace(self.camera_view, view).replace('.avi', '')
        return os.path.join(base_path, f"{corresponding_file}.avi")

    def save_calibration_points(self):
        calibration_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name)
        os.makedirs(os.path.dirname(calibration_path), exist_ok=True)

        data = {"bodyparts": [], "coords": [], "side": [], "front": [], "overhead": []}
        for label, coords in self.calibration_points_static.items():
            for coord in ['x', 'y']:
                data["bodyparts"].append(label)
                data["coords"].append(coord)
                for view in ["side", "front", "overhead"]:
                    if coords[view] is not None:
                        x, y = coords[view].get_offsets()[0]
                        if coord == 'x':
                            data[view].append(x)
                        else:
                            data[view].append(y)
                    else:
                        data[view].append(None)

        df = pd.DataFrame(data)
        df.to_csv(calibration_path, index=False)

        messagebox.showinfo("Info", "Calibration points saved successfully")

    def generate_label_colors(self, labels):
        colormap = plt.get_cmap('hsv')
        colors = [colormap(i / len(labels)) for i in range(len(labels))]
        return {label: self.rgb_to_hex(color) for label, color in zip(labels, colors)}

    def rgb_to_hex(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    def update_contrast_brightness(self, val):
        self.show_frames()

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        img_contrast = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_brightness = enhancer.enhance(brightness)
        return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)

    def on_scroll(self, event):
        if event.inaxes:
            ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            xdata = event.xdata
            ydata = event.ydata

            if xdata is not None and ydata is not None:
                zoom_factor = 0.9 if event.button == 'up' else 1.1

                new_xlim = [xdata + (x - xdata) * zoom_factor for x in xlim]
                new_ylim = [ydata + (y - ydata) * zoom_factor for y in ylim]

                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)

                self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.button == 2:
            self.panning = True
            self.pan_start = (event.x, event.y)

    def on_mouse_release(self, event):
        if event.button == 2:
            self.panning = False
            self.pan_start = None

    def on_mouse_move(self, event):
        if self.panning and self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_start = (event.x, event.y)

            ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            scale_x = (xlim[1] - xlim[0]) / self.canvas.get_width_height()[0]
            scale_y = (ylim[1] - ylim[0]) / self.canvas.get_width_height()[1]

            ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
            ax.set_ylim(ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)
            self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[["side", "front", "overhead"].index(view)]
        color = self.label_colors[self.current_label.get()]
        marker_size = self.marker_size_var.get()

        if event.button == MouseButton.RIGHT:
            if event.key == 'shift':
                self.delete_closest_point(ax, event)
            else:
                label = self.current_label.get()
                if self.calibration_points_static[label][view] is not None:
                    self.calibration_points_static[label][view].remove()
                self.calibration_points_static[label][view] = ax.scatter(event.xdata, event.ydata, c=color, s=marker_size * 10, label=label)
                self.canvas.draw()
        elif event.button == MouseButton.LEFT:
            self.dragging_point = self.find_closest_point(ax, event)

    def find_closest_point(self, ax, event):
        min_dist = float('inf')
        closest_point = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = points[self.current_view.get()]
        return closest_point if min_dist < 10 else None

    def delete_closest_point(self, ax, event):
        min_dist = float('inf')
        closest_point_label = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point_label = label

        if closest_point_label:
            self.calibration_points_static[closest_point_label][self.current_view.get()].remove()
            self.calibration_points_static[closest_point_label][self.current_view.get()] = None
            self.canvas.draw()

    def on_drag(self, event):
        if self.dragging_point is None or event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[["side", "front", "overhead"].index(view)]

        if event.button == MouseButton.LEFT:
            self.dragging_point.set_offsets((event.xdata, event.ydata))
            self.canvas.draw()

    def update_crosshair(self, event):
        for line in self.crosshair_lines:
            line.remove()
        self.crosshair_lines = []

        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.crosshair_lines.append(event.inaxes.axhline(y, color='cyan', linestyle='--', linewidth=0.5))
            self.crosshair_lines.append(event.inaxes.axvline(x, color='cyan', linestyle='--', linewidth=0.5))
            self.canvas.draw_idle()

    def reset_view(self):
        for ax in self.axs:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        self.contrast_var.set(config.DEFAULT_CONTRAST)
        self.brightness_var.set(config.DEFAULT_BRIGHTNESS)
        self.show_frames()

    def show_frames(self, val=None):
        frame_number = self.current_frame_index
        self.frame_label.config(text=f"Frame: {frame_number}/{len(self.matched_frames) - 1}")

        frame_side, frame_front, frame_overhead = self.matched_frames[frame_number]

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_side)
        ret_side, frame_side_img = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_front)
        ret_front, frame_front_img = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_overhead)
        ret_overhead, frame_overhead_img = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            frame_side_img = self.apply_contrast_brightness(frame_side_img)
            frame_front_img = self.apply_contrast_brightness(frame_front_img)
            frame_overhead_img = self.apply_contrast_brightness(frame_overhead_img)

            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

            self.axs[0].imshow(cv2.cvtColor(frame_side_img, cv2.COLOR_BGR2RGB))
            self.axs[1].imshow(cv2.cvtColor(frame_front_img, cv2.COLOR_BGR2RGB))
            self.axs[2].imshow(cv2.cvtColor(frame_overhead_img, cv2.COLOR_BGR2RGB))

            self.axs[0].set_title('Side View')
            self.axs[1].set_title('Front View')
            self.axs[2].set_title('Overhead View')

            self.show_static_points()
            self.canvas.draw_idle()

    def show_static_points(self):
        for label, points in self.calibration_points_static.items():
            for view, point in points.items():
                if point is not None:
                    ax = self.axs[["side", "front", "overhead"].index(view)]
                    ax.add_collection(point)
                    point.set_sizes([self.marker_size_var.get() * 10])
        self.canvas.draw()


class LabelFramesTool:
    def __init__(self, root, main_tool):
        self.root = root
        self.main_tool = main_tool
        self.video_name = ""
        self.video_date = ""
        self.extracted_frames_path = {}
        self.current_frame_index = 0
        self.contrast_var = tk.DoubleVar(value=config.DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=config.DEFAULT_BRIGHTNESS)
        self.marker_size_var = tk.DoubleVar(value=config.DEFAULT_MARKER_SIZE)
        self.mode = 'labeling'
        self.calibration_data = None
        self.fig = None
        self.axs = None
        self.canvas = None
        self.labels = config.BODY_PART_LABELS
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value='Nose')
        self.current_view = tk.StringVar(value="side")
        self.projection_view = tk.StringVar(value="side")  # view to project points from
        self.body_part_points = {}
        self.calibration_points_static = {label: {"side": None, "front": None, "overhead": None} for label in config.CALIBRATION_LABELS}
        self.cam_reprojected_points = {'near': {}, 'far': {}}
        self.frames = {'side': [], 'front': [], 'overhead': []}
        self.projection_lines = {'side': None, 'front': None, 'overhead': None}
        self.P = None
        self.tooltip = None
        self.label_buttons = []
        self.tooltip_window = None
        self.matched_frames = []  # Add this to ensure matched_frames is initialized
        self.frame_names = {'side': [], 'front': [], 'overhead': []}
        self.frame_numbers = {'side': [], 'front': [], 'overhead': []}

        self.crosshair_lines = []
        self.dragging_point = None
        self.panning = False
        self.pan_start = None

        self.spacer_lines_active = False
        self.spacer_lines_points = []
        self.spacer_lines = []

        self.last_update_time = 0

        self.label_frames_menu()

    def label_frames_menu(self):
        self.main_tool.clear_root()

        calibration_folder_path = filedialog.askdirectory(title="Select Calibration Folder")

        if not calibration_folder_path:
            self.main_tool.main_menu()
            return

        self.video_name = os.path.basename(calibration_folder_path)
        self.video_date = self.extract_date_from_folder_path(calibration_folder_path)
        self.calibration_file_path = os.path.join(calibration_folder_path, "calibration_labels.csv")

        if not os.path.exists(self.calibration_file_path):
            messagebox.showerror("Error", "No corresponding camera calibration data found.")
            return

        base_path = os.path.dirname(os.path.dirname(calibration_folder_path))

        video_names = {view: get_video_name_with_view(self.video_name, view) for view in ['side', 'front', 'overhead']}

        self.extracted_frames_path = {
            'side': os.path.normpath(os.path.join(base_path, "Side", video_names['side'])),
            'front': os.path.normpath(os.path.join(base_path, "Front", video_names['front'])),
            'overhead': os.path.normpath(os.path.join(base_path, "Overhead", video_names['overhead']))
        }

        if not all(os.path.exists(path) for path in self.extracted_frames_path.values()):
            missing_paths = [path for path in self.extracted_frames_path.values() if not os.path.exists(path)]
            messagebox.showerror("Error", "One or more corresponding extracted frames folders not found.")
            return

        self.current_frame_index = 0

        self.show_loading_popup()

        self.frames = {'side': [], 'front': [], 'overhead': []}
        self.root.after(100, self.load_frames)

    def show_loading_popup(self):
        self.loading_popup = tk.Toplevel(self.root)
        self.loading_popup.geometry("300x100")
        self.loading_popup.title("Loading")
        label = tk.Label(self.loading_popup, text="Loading frames, please wait...")
        label.pack(pady=20, padx=20)
        self.root.update_idletasks()

    def extract_date_from_folder_path(self, folder_path):
        parts = folder_path.split(os.sep)
        for part in parts:
            if part.isdigit() and len(part) == 8:
                return part
        return None

    def load_frames(self):
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        # self.frame_numbers = {'side': [], 'front': [], 'overhead': []}

        for view in self.frames.keys():
            frame_files = sorted(
                (f for f in os.listdir(self.extracted_frames_path[view]) if
                 os.path.splitext(f)[1].lower() in valid_extensions),
                key=lambda x: os.path.getctime(os.path.join(self.extracted_frames_path[view], x))
            )
            for file in frame_files:
                frame = cv2.imread(os.path.join(self.extracted_frames_path[view], file))
                self.frames[view].append(frame)
                self.frame_names[view].append(file)
                image_number = int(os.path.splitext(file)[0].replace('img', ''))
                self.frame_numbers[view].append(image_number)

        min_frame_count = min(len(self.frames[view]) for view in self.frames if self.frames[view])
        for view in self.frames:
            self.frames[view] = self.frames[view][:min_frame_count]
            self.frame_numbers[view] = self.frame_numbers[view][:min_frame_count]

        print(f"Number of frames loaded for side: {len(self.frames['side'])}")
        print(f"Number of frames loaded for front: {len(self.frames['front'])}")
        print(f"Number of frames loaded for overhead: {len(self.frames['overhead'])}")

        if min_frame_count == 0:
            messagebox.showerror("Error", "No frames found in the directories.")
            self.loading_popup.destroy()
            return

        self.loading_popup.destroy()

        self.body_part_points = {
            frame_idx: {label: {"side": None, "front": None, "overhead": None} for label in self.labels}
            for frame_idx in range(min_frame_count)
        }

        self.match_frames()  # Call match_frames here

        self.setup_labeling_ui()

        video_names = {view: os.path.basename(self.extracted_frames_path[view]) for view in
                       ['side', 'front', 'overhead']}

        for view in ['side', 'front', 'overhead']:
            label_file_path = os.path.join(config.LABEL_SAVE_PATH_TEMPLATE[view].format(video_name=video_names[view]),
                                           "CollectedData_Holly.csv")
            if os.path.exists(label_file_path):
                self.load_existing_labels(label_file_path, view)

        # Load calibration data and populate body_part_points with calibration labels
        self.load_calibration_data(self.calibration_file_path)

        self.display_frame()

    def match_frames(self):
        min_frame_count = min(len(self.frames['side']), len(self.frames['front']), len(self.frames['overhead']))
        self.matched_frames = [(i, i, i) for i in range(min_frame_count)]
        print(f"Matched frames: {len(self.matched_frames)}")

    def setup_labeling_ui(self):
        self.main_tool.clear_root()

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        settings_frame = tk.Frame(control_frame)
        settings_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(settings_frame, text="Marker Size").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_MARKER_SIZE, to=config.MAX_MARKER_SIZE, orient=tk.HORIZONTAL,
                 resolution=config.MARKER_SIZE_STEP, variable=self.marker_size_var,
                 command=self.update_marker_size).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Contrast").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_CONTRAST, to=config.MAX_CONTRAST, orient=tk.HORIZONTAL,
                 resolution=config.CONTRAST_STEP, variable=self.contrast_var,
                 command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Brightness").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_BRIGHTNESS, to=config.MAX_BRIGHTNESS, orient=tk.HORIZONTAL,
                 resolution=config.BRIGHTNESS_STEP, variable=self.brightness_var,
                 command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(frame_control,
                                    text=f"Frame: {self.current_frame_index + 1}/{len(self.frames['side'])}")
        self.frame_label.pack()

        self.prev_button = tk.Button(frame_control, text="<<", command=lambda: self.skip_labeling_frames(-1))
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(frame_control, text=">>", command=lambda: self.skip_labeling_frames(1))
        self.next_button.pack(side=tk.LEFT, padx=5)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        home_button = tk.Button(button_frame, text="Home", command=self.reset_view)
        home_button.pack(pady=5)

        save_button = tk.Button(button_frame, text="Save Labels", command=self.save_labels)
        save_button.pack(pady=5)

        spacer_lines_button = tk.Button(button_frame, text="Spacer Lines", command=self.toggle_spacer_lines)
        spacer_lines_button.pack(pady=5)

        optimize_button = tk.Button(button_frame, text="Optimize Calibration", command=self.optimize_calibration)
        optimize_button.pack(pady=5)

        view_frame = tk.Frame(control_frame)
        view_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Label(view_frame, text="Label View").pack()
        self.current_view = tk.StringVar(value="side")
        for view in ["side", "front", "overhead"]:
            tk.Radiobutton(view_frame, text=view.capitalize(), variable=self.current_view, value=view).pack(side=tk.TOP,
                                                                                                            pady=2)

        projection_frame = tk.Frame(control_frame)
        projection_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Label(projection_frame, text="Projection View").pack()
        self.projection_view = tk.StringVar(value="side")
        for view in ["side", "front", "overhead"]:
            tk.Radiobutton(projection_frame, text=view.capitalize(), variable=self.projection_view, value=view).pack(
                side=tk.TOP, pady=2)

        control_frame_right = tk.Frame(control_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=20)

        exit_button = tk.Button(control_frame_right, text="Exit", command=self.confirm_exit)
        exit_button.pack(pady=5)

        back_button = tk.Button(control_frame_right, text="Back to Main Menu", command=self.main_tool.main_menu)
        back_button.pack(pady=5)

        control_frame_labels = tk.Frame(main_frame)
        control_frame_labels.pack(side=tk.RIGHT, fill=tk.Y, padx=3, pady=1)  # Reduce padding to minimize space

        self.labels = config.BODY_PART_LABELS  # + ['Door']  # Add 'Door' label here
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])

        self.label_canvas = tk.Canvas(control_frame_labels, width=100)  # Set a fixed width for the label canvas
        self.label_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)  # Do not expand to use only necessary space
        self.label_scrollbar = tk.Scrollbar(control_frame_labels, orient=tk.VERTICAL, command=self.label_canvas.yview)
        self.label_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.label_canvas.configure(yscrollcommand=self.label_scrollbar.set)

        self.label_frame = tk.Frame(self.label_canvas)
        self.label_canvas.create_window((0, 0), window=self.label_frame, anchor="nw")
        self.label_frame.bind("<Configure>",
                              lambda e: self.label_canvas.configure(scrollregion=self.label_canvas.bbox("all")))

        for label in self.labels:
            color = self.label_colors[label]
            if label != 'Door' and label in config.CALIBRATION_LABELS:
                continue  # Skip adding button for static calibration labels except 'Door'
            label_button = tk.Radiobutton(self.label_frame, text=label, variable=self.current_label, value=label,
                                          indicatoron=0, width=15, bg=color, font=("Helvetica", 7),
                                          command=lambda l=label: self.on_label_select(l))
            label_button.pack(fill=tk.X, pady=1)
            self.label_buttons.append(label_button)

        # Ensure "Nose" is selected by default
        self.current_label.set("Nose")
        self.update_label_button_selection()

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 10))  # Adjust figure size for better fit
        self.fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, wspace=0.01, hspace=0.005)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tooltip = self.fig.text(0, 0, "", va="bottom", ha="left", fontsize=8,
                                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1), zorder=10)
        self.tooltip.set_visible(False)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("motion_notify_event", self.update_crosshair)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self.show_tooltip)

        self.display_frame()

    def update_label_button_selection(self):
        for button in self.label_buttons:
            if button.cget('text') == self.current_label.get():
                button.select()

    def load_existing_labels(self, label_file_path, view):
        # Replace filepath with h5 file
        label_file_path = label_file_path.replace('.csv', '.h5')
        df = pd.read_hdf(label_file_path, key='df')

        for frame_idx_pos, frame_idx in enumerate(df.index):
            for label in self.labels:
                # Check if the required key exists in the DataFrame before accessing it
                if ('Holly', label, 'x') in df.columns and ('Holly', label, 'y') in df.columns:
                    x, y = df.loc[frame_idx, ('Holly', label, 'x')], df.loc[frame_idx, ('Holly', label, 'y')]
                    if not np.isnan(x) and not np.isnan(y):
                        self.body_part_points[frame_idx_pos][label][view] = (x, y)
                else:
                    print(f"Label '{label}' not found in the DataFrame for frame {frame_idx}.")

    def show_tooltip(self, event):
        if event.inaxes in self.axs:
            marker_size = self.marker_size_var.get() * 10  # Assuming marker size is scaled
            for label, views in self.body_part_points[self.current_frame_index].items():
                for view, coords in views.items():
                    if view == self.current_view.get() and coords is not None:
                        x, y = coords
                        if np.hypot(x - event.xdata, y - event.ydata) < marker_size:
                            widget = self.canvas.get_tk_widget()
                            self.show_custom_tooltip(widget, label)
                            return
        self.hide_custom_tooltip()

    def show_custom_tooltip(self, wdgt, text):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = tk.Toplevel(wdgt)
        self.tooltip_window.overrideredirect(True)

        tk.Label(self.tooltip_window, text=text, background='yellow').pack()
        self.tooltip_window.update_idletasks()

        x_center = wdgt.winfo_pointerx() + 20
        y_center = wdgt.winfo_pointery() + 20
        self.tooltip_window.geometry(f"+{x_center}+{y_center}")

        wdgt.bind('<Leave>', self.hide_custom_tooltip)

    def hide_custom_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def get_p(self, view):
        if self.calibration_data:
            # Camera intrinsics
            K = self.calibration_data['intrinsics'][view]

            # Camera extrinsics
            R = self.calibration_data['extrinsics'][view]['rotm']
            t = self.calibration_data['extrinsics'][view]['tvec']

            # Ensure t is a column vector
            if t.ndim == 1:
                t = t[:, np.newaxis]

            # Form the projection matrix
            self.P = np.dot(K, np.hstack((R, t)))

    def get_camera_center(self, view):
        if self.calibration_data:
            # Extract the rotation matrix and translation vector
            R = self.calibration_data['extrinsics'][view]['rotm']
            t = self.calibration_data['extrinsics'][view]['tvec']

            # Compute the camera center in world coordinates
            camera_center = -np.dot(np.linalg.inv(R), t)

            return camera_center.flatten()  # Flatten to make it a 1D array

    def find_projection(self, view, bp):
        self.get_p(view)  # Ensure projection matrix is updated
        # Find 3D point with self.P and the current 2D point
        if self.body_part_points[self.current_frame_index][bp][view] is not None:
            x, y = self.body_part_points[self.current_frame_index][bp][view]

            if x is not None and y is not None:
                # Create the homogeneous coordinates for the 2D point
                uv = np.array([x, y, 1.0])

                # Compute the pseudo-inverse of the projection matrix
                P_inv = np.linalg.pinv(self.P)

                # Find the 3D point in homogeneous coordinates
                X = np.dot(P_inv, uv)

                # Normalize to get the 3D point
                X /= X[-1]

                return X[:3]  # Return only the x, y, z components
        return None

    def get_line_equation(self, point_3d, camera_center):
        # Extract the coordinates of the points
        x1, y1, z1 = point_3d
        x2, y2, z2 = camera_center

        def line_at_t(t):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            z = z1 + t * (z2 - z1)
            return x, y, z

        return line_at_t

    def find_t_for_coordinate(self, val, coord_index, point_3d, camera_center):
        x1, y1, z1 = point_3d
        x2, y2, z2 = camera_center

        if coord_index == 0:  # x-coordinate
            t = (val - x1) / (x2 - x1)
        elif coord_index == 1:  # y-coordinate
            t = (val - y1) / (y2 - y1)
        elif coord_index == 2:  # z-coordinate
            t = (val - z1) / (z2 - z1)
        else:
            raise ValueError("coord_index must be 0 (x), 1 (y), or 2 (z)")

        return t

    def find_3d_edges(self, view, bp):
        # Get the 3D coordinates of the body part
        point_3d = self.find_projection(view, bp)

        # Get the camera center
        camera_center = self.get_camera_center(view)

        if point_3d is not None and camera_center is not None:
            # Get the line equation
            line_at_t = self.get_line_equation(point_3d, camera_center)

            # Determine the appropriate dimension based on the view
            if view == "side":
                coord_index = 1  # y-coordinate for side view
            elif view == "front":
                coord_index = 0  # x-coordinate for front view
            elif view == "overhead":
                coord_index = 2  # z-coordinate for overhead view

            # Get the 3D coordinates of the near edge
            near_edge = line_at_t(self.find_t_for_coordinate(0, coord_index, point_3d, camera_center))

            # Get the 3D coordinates of the far edge
            far_edge_value = self.calibration_data['belt points WCS'].T[coord_index].max()
            if view == 'front':
                far_edge_value += 140  # Add the length of the belt to the far edge
            if view == 'overhead':
                far_edge_value = 60  # Set the far edge to fixed height for overhead view
            far_edge = line_at_t(self.find_t_for_coordinate(far_edge_value, coord_index, point_3d, camera_center))

            return near_edge, far_edge

        return None, None

    def reproject_3d_to_2d(self):
        view = self.projection_view.get()
        bp = self.current_label.get()

        # reset the reprojected points
        self.cam_reprojected_points['near'] = {}
        self.cam_reprojected_points['far'] = {}

        near_edge, far_edge = self.find_3d_edges(view, bp)
        if near_edge is not None and far_edge is not None:
            # Define the views and exclude the projection view
            views = ['side', 'front', 'overhead']
            views.remove(view)

            for wcs in [near_edge, far_edge]:
                # Loop through the other views
                for other_view in views:
                    CCS_repr, _ = cv2.projectPoints(
                        wcs,
                        cv2.Rodrigues(self.calibration_data['extrinsics'][other_view]['rotm'])[0],
                        self.calibration_data['extrinsics'][other_view]['tvec'],
                        self.calibration_data['intrinsics'][other_view],
                        np.array([]),
                    )
                    self.cam_reprojected_points['near' if wcs is near_edge else 'far'][other_view] = CCS_repr[
                        0].flatten()

    def draw_reprojected_points(self):
        self.reproject_3d_to_2d()
        for view in ['side', 'front', 'overhead']:
            if view != self.projection_view.get():
                ax = self.axs[["side", "front", "overhead"].index(view)]
                # Clear previous lines if they exist
                if self.projection_lines[view] is not None:
                    self.projection_lines[view].remove()
                    self.projection_lines[view] = None

                frame = cv2.cvtColor(self.frames[view][self.current_frame_index], cv2.COLOR_BGR2RGB)
                frame = self.apply_contrast_brightness(frame)
                ax.imshow(frame)
                ax.set_title(f'{view.capitalize()} View', fontsize=8)
                ax.axis('off')

                if view in self.cam_reprojected_points['near'] and view in self.cam_reprojected_points['far']:
                    near_point = self.cam_reprojected_points['near'][view]
                    far_point = self.cam_reprojected_points['far'][view]

                    # Draw the line between near and far points and store it
                    self.projection_lines[view], = ax.plot([near_point[0], far_point[0]], [near_point[1], far_point[1]],
                                                           'r-', linewidth=0.5, linestyle='--')

        self.show_body_part_points()  # Redraw body part points to ensure they are displayed correctly
        self.canvas.draw_idle()

    def on_label_select(self, label):
        self.current_label.set(label)
        self.draw_reprojected_points()

    def skip_labeling_frames(self, step):
        self.current_frame_index += step
        self.current_frame_index = max(0, min(self.current_frame_index, len(self.frames['side']) - 1))
        self.frame_label.config(text=f"Frame: {self.current_frame_index + 1}/{len(self.frames['side'])}")
        self.display_frame()
        self.current_label.set("Nose")

    def display_frame(self):
        frame_side, frame_front, frame_overhead = self.matched_frames[self.current_frame_index]

        frame_side_img = self.frames['side'][frame_side]
        frame_front_img = self.frames['front'][frame_front]
        frame_overhead_img = self.frames['overhead'][frame_overhead]

        frame_side_img = self.apply_contrast_brightness(frame_side_img)
        frame_front_img = self.apply_contrast_brightness(frame_front_img)
        frame_overhead_img = self.apply_contrast_brightness(frame_overhead_img)

        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()

        self.axs[0].imshow(cv2.cvtColor(frame_side_img, cv2.COLOR_BGR2RGB))
        self.axs[1].imshow(cv2.cvtColor(frame_front_img, cv2.COLOR_BGR2RGB))
        self.axs[2].imshow(cv2.cvtColor(frame_overhead_img, cv2.COLOR_BGR2RGB))

        self.axs[0].set_title('Side View')
        self.axs[1].set_title('Front View')
        self.axs[2].set_title('Overhead View')

        self.show_body_part_points()
        self.canvas.draw()

        # Reset to 'Nose' label
        self.current_label.set("Nose")
        self.update_label_button_selection()

    def show_body_part_points(self, draw=True):
        for ax in self.axs:
            for collection in ax.collections:
                collection.remove()

        current_points = self.body_part_points[self.current_frame_index]
        for label, views in current_points.items():
            for view, coords in views.items():
                if coords is not None:
                    x, y = coords
                    ax = self.axs[["side", "front", "overhead"].index(view)]
                    color = self.label_colors[label]
                    if label in config.CALIBRATION_LABELS:
                        ax.scatter(x, y, c=color, s=self.marker_size_var.get() * 10, label=label, edgecolors='red',
                                   linewidths=1)
                    else:
                        ax.scatter(x, y, c=color, s=self.marker_size_var.get() * 10, label=label)

        if draw:
            self.canvas.draw_idle()

    def toggle_spacer_lines(self):
        self.spacer_lines_active = not self.spacer_lines_active
        if not self.spacer_lines_active:
            self.remove_spacer_lines()
            self.spacer_lines_points = []
        else:
            self.spacer_lines_points = []

    def remove_spacer_lines(self):
        for line in self.spacer_lines:
            line.remove()
        self.spacer_lines = []
        self.canvas.draw_idle()

    def draw_spacer_lines(self, ax, start_point, end_point):
        if len(self.spacer_lines) > 0:
            self.remove_spacer_lines()

        x_values = np.linspace(start_point[0], end_point[0], num=12)
        for x in x_values:
            line = ax.axvline(x=x, color='pink', linestyle=':', linewidth=1)
            self.spacer_lines.append(line)

        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[["side", "front", "overhead"].index(view)]
        label = self.current_label.get()
        color = self.label_colors[label]
        marker_size = self.marker_size_var.get()

        frame_points = self.body_part_points[self.current_frame_index]

        if event.button == MouseButton.RIGHT:
            if self.spacer_lines_active:
                if len(self.spacer_lines_points) < 2:
                    self.spacer_lines_points.append((event.xdata, event.ydata))
                    if len(self.spacer_lines_points) == 2:
                        self.draw_spacer_lines(ax, self.spacer_lines_points[0], self.spacer_lines_points[1])
                return
            if event.key == 'shift':
                self.delete_closest_point(ax, event, frame_points)
            else:
                if label == 'Door' or label not in config.CALIBRATION_LABELS:
                    if frame_points[label][view] is not None:
                        frame_points[label][view] = None
                    frame_points[label][view] = (event.xdata, event.ydata)
                    ax.scatter(event.xdata, event.ydata, c=color, s=marker_size * 10, label=label)
                    self.canvas.draw_idle()
                    self.advance_label()
                    self.draw_reprojected_points()
        elif event.button == MouseButton.LEFT:
            if label == 'Door' or label not in config.CALIBRATION_LABELS:
                self.dragging_point = self.find_closest_point(ax, event, frame_points)

    def on_drag(self, event):
        if self.dragging_point is None or event.inaxes not in self.axs:
            return

        label, view, _ = self.dragging_point
        ax = self.axs[["side", "front", "overhead"].index(view)]

        if event.button == MouseButton.LEFT:
            self.body_part_points[self.current_frame_index][label][view] = (event.xdata, event.ydata)
            self.show_body_part_points()
            self.draw_reprojected_points()  # Call to update reprojected points

    def find_closest_point(self, ax, event, frame_points):
        min_dist = float('inf')
        closest_point = None
        for label, views in frame_points.items():
            for view, coords in views.items():
                if coords is not None:
                    x, y = coords
                    dist = np.hypot(x - event.xdata, y - event.ydata)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = (label, view, coords)
        return closest_point if min_dist < 10 else None

    def delete_closest_point(self, ax, event, frame_points):
        min_dist = float('inf')
        closest_point_label = None
        closest_view = None
        for label, views in frame_points.items():
            for view, coords in views.items():
                if coords is not None:
                    x, y = coords
                    dist = np.hypot(x - event.xdata, y - event.ydata)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point_label = label
                        closest_view = view

        if closest_point_label and closest_view:
            if closest_point_label == 'Door' or closest_point_label not in config.CALIBRATION_LABELS:
                frame_points[closest_point_label][closest_view] = None
                self.display_frame()

    def load_calibration_data(self, calibration_data_path):
        try:
            calibration_coordinates = pd.read_csv(calibration_data_path)
            calib = BasicCalibration(calibration_coordinates)
            cameras_extrinsics = calib.estimate_cams_pose()
            cameras_intrinsics = calib.cameras_intrinsics
            belt_points_WCS = calib.belt_coords_WCS
            belt_points_CCS = calib.belt_coords_CCS

            self.calibration_data = {
                'extrinsics': cameras_extrinsics,
                'intrinsics': cameras_intrinsics,
                'belt points WCS': belt_points_WCS,
                'belt points CCS': belt_points_CCS
            }

            for label in config.CALIBRATION_LABELS:
                for view in ['side', 'front', 'overhead']:
                    x_vals = calibration_coordinates[
                        (calibration_coordinates['bodyparts'] == label) & (calibration_coordinates['coords'] == 'x')][
                        view].values
                    y_vals = calibration_coordinates[
                        (calibration_coordinates['bodyparts'] == label) & (calibration_coordinates['coords'] == 'y')][
                        view].values

                    if len(x_vals) > 0 and len(y_vals) > 0:
                        x = x_vals[0]
                        y = y_vals[0]
                        self.calibration_points_static[label][view] = (x, y)
                        if label != 'Door':
                            for frame in self.body_part_points.keys():
                                self.body_part_points[frame][label][view] = (x, y)
                    else:
                        self.calibration_points_static[label][view] = None
                        print(f"Missing data for {label} in {view} view")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration data: {e}")
            print(f"Error loading calibration data: {e}")

    def update_marker_size(self, val):
        current_xlim = [ax.get_xlim() for ax in self.axs]
        current_ylim = [ax.get_ylim() for ax in self.axs]
        self.marker_size = self.marker_size_var.get()
        self.display_frame()
        for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def update_contrast_brightness(self, val):
        current_time = time.time()
        if current_time - self.last_update_time > 0.1:  # 100ms debounce time
            current_xlim = [ax.get_xlim() for ax in self.axs]
            current_ylim = [ax.get_ylim() for ax in self.axs]
            self.display_frame()
            for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            self.canvas.draw_idle()
            self.last_update_time = current_time

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        frame_bytes = frame.tobytes()
        return self.cached_apply_contrast_brightness(frame_bytes, frame.shape, contrast, brightness)

    @lru_cache(maxsize=128)
    def cached_apply_contrast_brightness(self, frame_bytes, frame_shape, contrast, brightness):
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)  # Convert bytes back to numpy array
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        img_contrast = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_brightness = enhancer.enhance(brightness)
        return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)

    def on_scroll(self, event):
        if event.inaxes:
            ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            xdata = event.xdata
            ydata = event.ydata

            if xdata is not None and ydata is not None:
                zoom_factor = 0.9 if event.button == 'up' else 1.1

                new_xlim = [xdata + (x - xdata) * zoom_factor for x in xlim]
                new_ylim = [ydata + (y - ydata) * zoom_factor for y in ylim]

                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)

                self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.button == 2:
            self.panning = True
            self.pan_start = (event.x, event.y)

    def on_mouse_release(self, event):
        if event.button == 2:
            self.panning = False
            self.pan_start = None
        elif event.button == MouseButton.LEFT:
            self.dragging_point = None
        elif event.button == MouseButton.RIGHT and self.spacer_lines_active:
            if len(self.spacer_lines_points) == 2:
                self.spacer_lines_points = []
                self.spacer_lines_active = False

    def on_mouse_move(self, event):
        if self.panning and self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_start = (event.x, event.y)

            ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            scale_x = (xlim[1] - xlim[0]) / self.canvas.get_width_height()[0]
            scale_y = (ylim[1] - ylim[0]) / self.canvas.get_width_height()[1]

            ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
            ax.set_ylim(ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)
            self.canvas.draw_idle()

    def update_crosshair(self, event):
        for line in self.crosshair_lines:
            line.remove()
        self.crosshair_lines = []

        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.crosshair_lines.append(event.inaxes.axhline(y, color='cyan', linestyle='--', linewidth=0.5))
            self.crosshair_lines.append(event.inaxes.axvline(x, color='cyan', linestyle='--', linewidth=0.5))
            self.canvas.draw_idle()

    def advance_label(self):
        current_index = self.labels.index(self.current_label.get())
        next_index = (current_index + 1) % len(self.labels)
        if next_index != 0 or len(self.labels) == 1:
            self.current_label.set(self.labels[next_index])
        else:
            self.current_label.set('')  # No more labels to advance to
        self.draw_reprojected_points()  # Update the reprojected points for the new label

    def reset_view(self):
        for ax in self.axs:
            ax.set_xlim(0, ax.get_images()[0].get_array().shape[1])
            ax.set_ylim(ax.get_images()[0].get_array().shape[0], 0)
        self.contrast_var.set(config.DEFAULT_CONTRAST)
        self.brightness_var.set(config.DEFAULT_BRIGHTNESS)
        self.marker_size_var.set(config.DEFAULT_MARKER_SIZE)
        self.display_frame()

    def save_labels(self):
        video_names = {view: os.path.basename(self.extracted_frames_path[view]) for view in
                       ['side', 'front', 'overhead']}
        save_paths = {
            'side': os.path.join(config.LABEL_SAVE_PATH_TEMPLATE['side'].format(video_name=video_names['side']),
                                "CollectedData_Holly.csv"),
            'front': os.path.join(config.LABEL_SAVE_PATH_TEMPLATE['front'].format(video_name=video_names['front']),
                                "CollectedData_Holly.csv"),
            'overhead': os.path.join(config.LABEL_SAVE_PATH_TEMPLATE['overhead'].format(video_name=video_names['overhead']),
                                "CollectedData_Holly.csv")
        }
        for path in save_paths.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {view: [] for view in ['side', 'front', 'overhead']}
        for frame_idx, labels in self.body_part_points.items():
            for label, views in labels.items():
                for view, coords in views.items():
                    if coords is not None:
                        x, y = coords
                        frame_number = self.matched_frames[frame_idx][['side', 'front', 'overhead'].index(view)]
                        filename = self.frame_names[view][frame_number]
                        video_filename = os.path.basename(self.extracted_frames_path[view])
                        data[view].append((frame_idx, label, x, y, "Holly", video_filename, filename))

        for view, view_data in data.items():
            df_view = pd.DataFrame(view_data, columns=["frame_index", "label", "x", "y", "scorer", "video_filename",
                                                  "frame_filename"])

            # Initialize an empty DataFrame with the correct columns
            multi_cols = pd.MultiIndex.from_product([['Holly'], self.labels, ['x', 'y']],
                                                    names=['scorer', 'bodyparts', 'coords'])
            multi_idx = pd.MultiIndex.from_tuples(
                [('labeled_data', video_names[view], filename) for filename in df_view['frame_filename'].unique()])
            df_ordered = pd.DataFrame(index=multi_idx, columns=multi_cols)

            for _, row in df_view.iterrows():
                df_ordered.loc[('labeled_data', row.video_filename, row.frame_filename), ('Holly', row.label, 'x')] = row.x
                df_ordered.loc[('labeled_data', row.video_filename, row.frame_filename), ('Holly', row.label, 'y')] = row.y

            # Convert the DataFrame to numeric values to ensure saving works
            df_ordered = df_ordered.apply(pd.to_numeric)

            # Save the DataFrame
            save_path = save_paths[view]
            print(f"Saving to {save_path}")
            try:
                df_ordered.to_csv(save_path)
                df_ordered.to_hdf(save_path.replace(".csv", ".h5"), key='df', mode='w', format='fixed')
            except PermissionError as e:
                print(f"PermissionError: {e}")
                messagebox.showerror("Error",
                                     f"Unable to save the file at {save_path}. Please check the file permissions.")

        print("Labels saved successfully")
        messagebox.showinfo("Info", "Labels saved successfully")


    def parse_video_path(self, video_path):
        video_file = os.path.basename(video_path)
        parts = video_file.split('_')
        date = parts[1]
        camera_view = [part for part in parts if part in ['side', 'front', 'overhead']][0]
        name_parts = [part for part in parts if part not in ['side', 'front', 'overhead']]
        name = '_'.join(name_parts).replace('.avi', '')  # Remove the .avi extension if present
        name_with_camera = f"{name}_{camera_view}"
        return name_with_camera, date, camera_view

    def generate_label_colors(self, labels):
        colormap = plt.get_cmap('hsv')
        body_part_labels = [label for label in labels if label not in config.CALIBRATION_LABELS]
        colors = [colormap(i / len(body_part_labels)) for i in range(len(body_part_labels))]
        label_colors = {}
        for label in labels:
            if label in config.CALIBRATION_LABELS:
                label_colors[label] = '#ffffff'  # White color for calibration labels
            else:
                label_colors[label] = self.rgb_to_hex(
                    colors.pop(0))  # Assign colors from the colormap to body part labels
        return label_colors

    def rgb_to_hex(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    ##################################### Calibration enhancements ############################################
    def optimize_calibration(self):
        reference_points = ['Nose', 'ForepawToeR', 'ForepawToeL', 'Back6']
        optimized_points = {}

        missing_labels = [label for label in reference_points if
                          label not in self.body_part_points[self.current_frame_index]]
        if missing_labels:
            print(f"Warning: The following reference points are missing in the current frame: {missing_labels}")

        # Print initial reprojection error
        initial_total_error, initial_errors = self.compute_reprojection_error(reference_points)
        print(f"Initial total reprojection error for {reference_points}: {initial_total_error}")
        for label, views in initial_errors.items():
            print(f"Initial reprojection error for {label}: {views}")

        for label in config.CALIBRATION_LABELS:
            optimized_points[label] = {}
            for view in ["side", "front", "overhead"]:
                initial_point = self.calibration_points_static[label][view]
                if initial_point is not None:
                    optimized_points[label][view] = self.optimize_point(label, view, initial_point, reference_points)

        for label, views in optimized_points.items():
            for view, point in views.items():
                self.calibration_points_static[label][view] = point

        self.recalculate_camera_parameters()

        # Print new reprojection error
        new_total_error, new_errors = self.compute_reprojection_error(reference_points)
        print(f"New total reprojection error for {reference_points}: {new_total_error}")
        for label, views in new_errors.items():
            print(f"New reprojection error for {label}: {views}")

        # Print the change in error
        error_change = new_total_error - initial_total_error
        print(f"Change in total reprojection error: {error_change}")

        self.save_optimized_calibration_points()

        # Reload enhanced calibration points to use for drawing guidance lines
        enhanced_calibration_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name).replace(
            '.csv', '_enhanced.csv')
        self.load_calibration_data(enhanced_calibration_path)

        self.display_frame()  # Redraw the frame to reflect the changes

    def optimize_point(self, label, view, initial_point, reference_points):
        def reprojection_error(point):
            self.calibration_points_static[label][view] = point
            self.recalculate_camera_parameters()
            total_error, _ = self.compute_reprojection_error(reference_points)
            return total_error

        initial_point = np.array(initial_point, dtype=float)

        result = minimize(reprojection_error, initial_point, method='L-BFGS-B',
                          bounds=[(initial_point[0] - 3.0, initial_point[0] + 3.0),
                                  (initial_point[1] - 3.0, initial_point[1] + 3.0)],
                          options={'maxiter': 3000, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False})
        ## make multi dimensional ( all reference pts and x/y)
        ## increase tolerance OR lower iterations (3000 may already be really low, check defaults), because i dont need that low tolerance
        ## check if learning rate is a parameter to set


        return result.x

    def compute_reprojection_error(self, labels):
        errors = {label: {"side": 0, "front": 0, "overhead": 0} for label in labels}
        total_error = 0
        for label in labels:
            for view in ["side", "front", "overhead"]:
                if self.body_part_points[self.current_frame_index].get(label, {}).get(view) is not None:
                    projected_x, projected_y = self.project_to_view(label, view)
                    original_x, original_y = self.body_part_points[self.current_frame_index][label][view]
                    error = np.sqrt((projected_x - original_x) ** 2 + (projected_y - original_y) ** 2)
                    errors[label][view] = error
                    total_error += error
        return total_error, errors

    def project_to_view(self, label, view):
        point_3d = self.find_projection(view, label)
        if point_3d is not None:
            CCS_repr, _ = cv2.projectPoints(
                point_3d,
                cv2.Rodrigues(self.calibration_data['extrinsics'][view]['rotm'])[0],
                self.calibration_data['extrinsics'][view]['tvec'],
                self.calibration_data['intrinsics'][view],
                np.array([]),
            )
            return CCS_repr[0].flatten()
        return None, None

    def recalculate_camera_parameters(self):
        calibration_coordinates = pd.DataFrame([
            {'bodyparts': label, 'coords': coord, 'side': self.calibration_points_static[label]['side'][i],
             'front': self.calibration_points_static[label]['front'][i],
             'overhead': self.calibration_points_static[label]['overhead'][i]}
            for label in self.calibration_points_static
            for i, coord in enumerate(['x', 'y'])
        ])

        calib = BasicCalibration(calibration_coordinates)
        cameras_extrinsics = calib.estimate_cams_pose()
        cameras_intrinsics = calib.cameras_intrinsics
        belt_points_WCS = calib.belt_coords_WCS
        belt_points_CCS = calib.belt_coords_CCS

        self.calibration_data = {
            'extrinsics': cameras_extrinsics,
            'intrinsics': cameras_intrinsics,
            'belt points WCS': belt_points_WCS,
            'belt points CCS': belt_points_CCS
        }

    def save_optimized_calibration_points(self):
        calibration_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name).replace('.csv',
                                                                                                            '_enhanced.csv')
        os.makedirs(os.path.dirname(calibration_path), exist_ok=True)

        data = {"bodyparts": [], "coords": [], "side": [], "front": [], "overhead": []}
        for label, coords in self.calibration_points_static.items():
            if label in config.CALIBRATION_LABELS:
                for coord in ['x', 'y']:
                    data["bodyparts"].append(label)
                    data["coords"].append(coord)
                    for view in ["side", "front", "overhead"]:
                        if coords[view] is not None:
                            x, y = coords[view]
                            if coord == 'x':
                                data[view].append(x)
                            else:
                                data[view].append(y)
                        else:
                            data[view].append(None)

        df = pd.DataFrame(data)
        try:
            df.to_csv(calibration_path, index=False)
            messagebox.showinfo("Info", "Optimized calibration points saved successfully")
        except PermissionError:
            print(
                f"Permission denied: Unable to save the file at {calibration_path}. Please check the file permissions.")
            messagebox.showerror("Error",
                                 f"Unable to save the file at {calibration_path}. Please check the file permissions.")

        ############################################################################################################

    def confirm_exit(self):
        answer = messagebox.askyesnocancel("Exit", "Do you want to save the labels before exiting?")
        if answer is not None:
            if answer:  # Yes, save labels and exit
                self.save_labels()
            self.root.quit()  # Exit without saving if No or after saving if Yes


if __name__ == "__main__":
    root = tk.Tk()
    app = MainTool(root)
    root.mainloop()
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from PIL import Image, ImageTk, ImageEnhance

import Helpers.MultiCamLabelling_config as config
from Helpers.CalibrateCams import BasicCalibration


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
        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        self.cap_front = cv2.VideoCapture(self.get_corresponding_video_path('front'))
        self.cap_overhead = cv2.VideoCapture(self.get_corresponding_video_path('overhead'))

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, pady=10)

        self.frame_label = tk.Label(control_frame, text="Frame: 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        self.slider = tk.Scale(control_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, length=600,
                               command=self.update_frame_label)
        self.slider.pack(side=tk.LEFT, padx=5)

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

        self.show_frames_extraction()

    def update_frame_label(self, val):
        self.current_frame_index = int(val)
        self.show_frames_extraction()

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
        self.show_frames_extraction()

    def save_extracted_frames(self):
        frame_number = self.current_frame_index
        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_side, frame_side = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_front, frame_front = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_overhead, frame_overhead = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            side_path = os.path.join(config.FRAME_SAVE_PATH_TEMPLATE["side"].format(video_name=self.video_name),
                                     f"img{frame_number}.png")
            front_path = os.path.join(config.FRAME_SAVE_PATH_TEMPLATE["front"].format(video_name=self.video_name),
                                      f"img{frame_number}.png")
            overhead_path = os.path.join(config.FRAME_SAVE_PATH_TEMPLATE["overhead"].format(video_name=self.video_name),
                                         f"img{frame_number}.png")

            os.makedirs(os.path.dirname(side_path), exist_ok=True)
            os.makedirs(os.path.dirname(front_path), exist_ok=True)
            os.makedirs(os.path.dirname(overhead_path), exist_ok=True)

            cv2.imwrite(side_path, frame_side)
            cv2.imwrite(front_path, frame_front)
            cv2.imwrite(overhead_path, frame_overhead)

    def show_frames_extraction(self, val=None):
        frame_number = self.current_frame_index

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_side, frame_side = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_front, frame_front = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_overhead, frame_overhead = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            frame_side = self.apply_contrast_brightness(frame_side)
            frame_front = self.apply_contrast_brightness(frame_front)
            frame_overhead = self.apply_contrast_brightness(frame_overhead)

            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

            self.axs[0].imshow(cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB))
            self.axs[1].imshow(cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB))
            self.axs[2].imshow(cv2.cvtColor(frame_overhead, cv2.COLOR_BGR2RGB))

            self.axs[0].set_title('Side View')
            self.axs[1].set_title('Front View')
            self.axs[2].set_title('Overhead View')

            self.canvas.draw()

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
        name = '_'.join(name_parts)
        return name, date, camera_view

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace(self.camera_view, view)
        return os.path.join(base_path, corresponding_file)


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
                 resolution=config.MARKER_SIZE_STEP, variable=self.marker_size_var, command=self.update_marker_size).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Contrast").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_CONTRAST, to=config.MAX_CONTRAST, orient=tk.HORIZONTAL,
                 resolution=config.CONTRAST_STEP, variable=self.contrast_var, command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Brightness").pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_BRIGHTNESS, to=config.MAX_BRIGHTNESS, orient=tk.HORIZONTAL,
                 resolution=config.BRIGHTNESS_STEP, variable=self.brightness_var, command=self.update_contrast_brightness).pack(side=tk.LEFT, padx=5)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(frame_control, text="Frame: 0")
        self.frame_label.pack()

        self.slider = tk.Scale(frame_control, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, length=400,
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
            tk.Radiobutton(control_frame_right, text=view.capitalize(), variable=self.current_view, value=view).pack(pady=2)

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

        self.calibration_points_static = {label: {"side": None, "front": None, "overhead": None} for label in self.labels}

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
        name = '_'.join(name_parts)
        return name, date, camera_view

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace(self.camera_view, view)
        return os.path.join(base_path, corresponding_file)

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

    def update_marker_size(self, val):
        for label, coords in self.calibration_points_static.items():
            for view, point in coords.items():
                if point is not None:
                    point.set_sizes([self.marker_size_var.get() * 10])
        self.canvas.draw()

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
        ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

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
        self.frame_label.config(text=f"Frame: {frame_number}/{self.total_frames - 1}")

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_side, frame_side = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_front, frame_front = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret_overhead, frame_overhead = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            frame_side = self.apply_contrast_brightness(frame_side)
            frame_front = self.apply_contrast_brightness(frame_front)
            frame_overhead = self.apply_contrast_brightness(frame_overhead)

            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

            self.axs[0].imshow(cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB))
            self.axs[1].imshow(cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB))
            self.axs[2].imshow(cv2.cvtColor(frame_overhead, cv2.COLOR_BGR2RGB))

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
        self.current_label = tk.StringVar(value=self.labels[0])
        self.current_view = tk.StringVar(value="side")
        self.projection_view = tk.StringVar(value="side") # view to project points from
        self.body_part_points = {}
        self.cam_reprojected_points = {'near': {}, 'far': {}}
        self.frames = {'side': [], 'front': [], 'overhead': []}
        self.projection_lines = {'side': None, 'front': None, 'overhead': None}
        self.P = None

        self.crosshair_lines = []
        self.dragging_point = None
        self.panning = False
        self.pan_start = None

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

        self.extracted_frames_path = {
            'side': f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Side/{self.video_name}",
            'front': f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Front/{self.video_name}",
            'overhead': f"X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Overhead/{self.video_name}"
        }

        if not all(os.path.exists(path) for path in self.extracted_frames_path.values()):
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
        for view in self.frames.keys():
            frame_files = sorted(os.listdir(self.extracted_frames_path[view]),
                                 key=lambda x: os.path.getctime(os.path.join(self.extracted_frames_path[view], x)))
            self.frames[view] = [cv2.imread(os.path.join(self.extracted_frames_path[view], file)) for file in
                                 frame_files]

        min_frame_count = min(len(self.frames[view]) for view in self.frames)
        for view in self.frames:
            self.frames[view] = self.frames[view][:min_frame_count]

        self.loading_popup.destroy()

        self.load_calibration_data(self.calibration_file_path)
        self.setup_labeling_ui()

        self.body_part_points = {
            frame_idx: {label: {"side": None, "front": None, "overhead": None} for label in self.labels}
            for frame_idx in range(len(self.frames['side']))
        }

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

        back_button = tk.Button(button_frame, text="Back to Main Menu", command=self.main_tool.main_menu)
        back_button.pack(pady=5)

        # Move view selection buttons to the top right
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

        control_frame_right = tk.Frame(main_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        self.labels = config.BODY_PART_LABELS
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])

        self.body_part_points = {
            frame_idx: {label: {"side": None, "front": None, "overhead": None} for label in self.labels} for
            frame_idx
            in range(len(self.frames['side']))}

        # Create buttons for body part labels with colors
        self.label_buttons = []
        for label in self.labels:
            color = self.label_colors[label]
            label_button = tk.Radiobutton(control_frame_right, text=label, variable=self.current_label, value=label,
                                          indicatoron=0, width=20, bg=color, font=("Helvetica", 8), command=lambda l=label: self.on_label_select(l))
            label_button.pack(fill=tk.X, pady=2)
            self.label_buttons.append(label_button)

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

        self.display_frame()


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

            # Get the 3D coordinates of the near edge, ie y=0
            y = 0
            t_near = self.find_t_for_coordinate(y, 1, point_3d, camera_center)
            near_edge = line_at_t(t_near)

            # Get the 3D coordinates of the far edge
            y = self.calibration_data['belt points WCS'].T[1].max()
            t_far = self.find_t_for_coordinate(y, 1, point_3d, camera_center)
            far_edge = line_at_t(t_far)

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
                ax.set_title(f'{view.capitalize()} View')
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

    def display_frame(self):
        for i, (ax, view) in enumerate(zip(self.axs, ['side', 'front', 'overhead'])):
            ax.cla()  # Clear the axis before drawing
            frame = cv2.cvtColor(self.frames[view][self.current_frame_index], cv2.COLOR_BGR2RGB)
            frame = self.apply_contrast_brightness(frame)
            ax.imshow(frame)
            ax.set_title(f'{view.capitalize()} View')
            ax.axis('off')

        self.show_body_part_points()
        self.draw_reprojected_points()  # Call to update reprojected points
        self.canvas.draw()

    def show_body_part_points(self):
        current_points = self.body_part_points[self.current_frame_index]
        for label, views in current_points.items():
            for view, coords in views.items():
                if coords is not None:
                    x, y = coords
                    ax = self.axs[["side", "front", "overhead"].index(view)]
                    ax.scatter(x, y, c=self.label_colors[label], s=self.marker_size_var.get() * 10, label=label)
        self.canvas.draw()

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
            if event.key == 'shift':
                self.delete_closest_point(ax, event, frame_points)
            else:
                if frame_points[label][view] is not None:
                    frame_points[label][view] = None
                frame_points[label][view] = (event.xdata, event.ydata)
                ax.scatter(event.xdata, event.ydata, c=color, s=marker_size * 10, label=label)
                self.canvas.draw()
                self.advance_label()
                self.draw_reprojected_points()  # Call to update reprojected points
        elif event.button == MouseButton.LEFT:
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
            frame_points[closest_point_label][closest_view] = None
            self.display_frame()

    def load_calibration_data(self, calibration_data_path):
        try:
            calibration_coordinates = pd.read_csv(calibration_data_path)

            calib = BasicCalibration(calibration_coordinates)
            cameras_extrinsics = calib.estimate_cams_pose()
            cameras_intrisinics = calib.cameras_intrinsics
            belt_points_WCS = calib.belt_coords_WCS
            belt_points_CCS = calib.belt_coords_CCS

            self.calibration_data = {
                'extrinsics': cameras_extrinsics,
                'intrinsics': cameras_intrisinics,
                'belt points WCS': belt_points_WCS,
                'belt points CCS': belt_points_CCS
            }

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration data: {e}")

    def update_marker_size(self, val):
        self.marker_size = self.marker_size_var.get()
        self.show_body_part_points()

    # def update_contrast_brightness(self, val):
    #     self.display_frame()
    def update_contrast_brightness(self, val):
        current_xlim = [ax.get_xlim() for ax in self.axs]
        current_ylim = [ax.get_ylim() for ax in self.axs]
        self.display_frame()
        for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

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
        ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

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
        side_path = config.LABEL_SAVE_PATH_TEMPLATE['side'].format(video_name=self.video_name)
        front_path = config.LABEL_SAVE_PATH_TEMPLATE['front'].format(video_name=self.video_name)
        overhead_path = config.LABEL_SAVE_PATH_TEMPLATE['overhead'].format(video_name=self.video_name)

        os.makedirs(os.path.dirname(side_path), exist_ok=True)
        os.makedirs(os.path.dirname(front_path), exist_ok=True)
        os.makedirs(os.path.dirname(overhead_path), exist_ok=True)

        # Saving the body part coordinates
        data = {"frame_index": [], "label": [], "view": [], "x": [], "y": []}
        for frame_idx, labels in self.body_part_points.items():
            for label, views in labels.items():
                for view, coords in views.items():
                    if coords is not None:
                        data["frame_index"].append(frame_idx)
                        data["label"].append(label)
                        data["view"].append(view)
                        data["x"].append(coords[0])
                        data["y"].append(coords[1])

        df = pd.DataFrame(data)
        df.to_csv(side_path, index=False)
        df.to_csv(front_path, index=False)
        df.to_csv(overhead_path, index=False)

        messagebox.showinfo("Info", "Labels saved successfully")

    def generate_label_colors(self, labels):
        colormap = plt.get_cmap('hsv')
        colors = [colormap(i / len(labels)) for i in range(len(labels))]
        return {label: self.rgb_to_hex(color) for label, color in zip(labels, colors)}

    def rgb_to_hex(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
if __name__ == "__main__":
    root = tk.Tk()
    app = MainTool(root)
    root.mainloop()
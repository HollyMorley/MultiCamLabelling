"""
Pure utility functions shared across GUI tools.
"""

import time
import tkinter as tk


def make_scrollable(parent) -> tk.Frame:
    canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
    scrollbar = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    inner = tk.Frame(canvas)
    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _sync_scrollregion(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _sync_inner_width(event):
        canvas.itemconfigure(inner_id, width=event.width)

    inner.bind("<Configure>", _sync_scrollregion)
    canvas.bind("<Configure>", _sync_inner_width)

    # Mouse-wheel only while cursor is over this canvas.
    def _on_wheel(event):
        canvas.yview_scroll(int(-event.delta / 120), "units")

    def _bind_wheel(_event):
        canvas.bind_all("<MouseWheel>", _on_wheel)

    def _unbind_wheel(_event):
        canvas.unbind_all("<MouseWheel>")

    canvas.bind("<Enter>", _bind_wheel)
    canvas.bind("<Leave>", _unbind_wheel)

    return inner


def help_button(parent, title: str, body: str) -> tk.Button:
    """Build a small "?" button that opens a Toplevel popup containing `body`.
    """
    def _show():
        top = tk.Toplevel(parent)
        top.title(title)
        top.transient(parent.winfo_toplevel())
        # Center the popup roughly over the parent
        parent.update_idletasks()
        x = parent.winfo_rootx() + 40
        y = parent.winfo_rooty() + 40
        top.geometry(f"+{x}+{y}")

        tk.Label(
            top, text=body, wraplength=460, justify=tk.LEFT,
            padx=14, pady=12,
        ).pack(fill=tk.BOTH, expand=True)
        tk.Button(top, text="OK", command=top.destroy, width=8).pack(pady=(0, 12))
        top.bind("<Escape>", lambda e: top.destroy())
        top.focus_set()

    return tk.Button(parent, text="?", width=2, command=_show)


def rgb_to_hex(color):
    """Convert an (r, g, b, ...) tuple with 0-1 floats to a hex color string."""
    return "#{:02x}{:02x}{:02x}".format(
        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
    )


def generate_label_colors(labels, calibration_labels=None):
    """Generate a distinct color for each label using the HSV colormap.

    If calibration_labels is provided, those labels get white and the colormap
    is distributed only across the remaining body-part labels.
    """
    from matplotlib import pyplot as plt
    colormap = plt.get_cmap("hsv")
    if calibration_labels:
        body_part_labels = [l for l in labels if l not in calibration_labels]
        colors = [colormap(i / len(body_part_labels)) for i in range(len(body_part_labels))]
        label_colors = {}
        color_idx = 0
        for label in labels:
            if label in calibration_labels:
                label_colors[label] = "#ffffff"
            else:
                label_colors[label] = rgb_to_hex(colors[color_idx])
                color_idx += 1
        return label_colors
    else:
        colors = [colormap(i / len(labels)) for i in range(len(labels))]
        return {label: rgb_to_hex(color) for label, color in zip(labels, colors)}


def apply_contrast_brightness(frame, contrast, brightness):
    """Apply contrast and brightness adjustments to a BGR frame."""
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    img_contrast = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Brightness(img_contrast)
    img_brightness = enhancer.enhance(brightness)
    return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)


def clip_ray_to_aabb(origin, direction, aabb):
    """Clip a 3D ray to an axis-aligned bounding box (AABB) using the slab method.

    The ray is parameterised as ``point(t) = origin + t * direction``. For each
    world axis we find the t-values where the ray enters and exits the box's
    slab on that axis; the ray is inside the box when it's inside all three
    slabs simultaneously, so t_near = max(per-axis enters) and
    t_far = min(per-axis exits). If t_near > t_far the ray misses the box.

    origin, direction: array-likes with 3 entries.
    aabb: dict with keys 'x', 'y', 'z' each mapping to [min, max].

    Returns (point_near, point_far) as numpy arrays, or (None, None) if the
    ray misses the box.
    """
    import numpy as np

    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)

    t_enters: list[float] = []
    t_exits: list[float] = []
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        a_min, a_max = aabb[axis_name]
        d = direction[axis_idx]
        if abs(d) < 1e-12:
            # Ray is parallel to this axis: it's either always or never inside
            # the slab. If origin is outside, the ray misses the box.
            if origin[axis_idx] < a_min or origin[axis_idx] > a_max:
                return None, None
            continue  # this axis doesn't constrain t
        t1 = (a_min - origin[axis_idx]) / d
        t2 = (a_max - origin[axis_idx]) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_enters.append(t1)
        t_exits.append(t2)

    if not t_enters:
        # Ray parallel to every axis (degenerate): no segment to draw.
        return None, None

    t_near = max(t_enters)
    t_far = min(t_exits)
    if t_near > t_far:
        return None, None  # ray misses the box

    return origin + t_near * direction, origin + t_far * direction


def debounce(wait):
    """Decorator that suppresses calls within `wait` seconds of the last call."""
    def decorator(fn):
        last_call = [0]

        def debounced(*args, **kwargs):
            current_time = time.time()
            if current_time - last_call[0] >= wait:
                last_call[0] = current_time
                return fn(*args, **kwargs)

        return debounced

    return decorator

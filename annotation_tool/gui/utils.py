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


def attach_tooltip(widget, text):
    """Show `text` as a hover tooltip below `widget`, unless no text.

    Used to explain why a disabled button is disabled. Bindings still fire on
    state=DISABLED widgets, so this works for greyed-out buttons.
    """
    if not text:
        return
    state = {"window": None}

    def show(_event):
        if state["window"] is not None:
            return
        x = widget.winfo_rootx() + 8
        y = widget.winfo_rooty() + widget.winfo_height() + 4
        win = tk.Toplevel(widget)
        win.wm_overrideredirect(True)
        win.geometry(f"+{x}+{y}")
        tk.Label(
            win, text=text, bg="#ffffe0", fg="#333",
            relief=tk.SOLID, borderwidth=1,
            padx=6, pady=3, font=("Helvetica", 9), justify=tk.LEFT,
            wraplength=320,
        ).pack()
        state["window"] = win

    def hide(_event):
        if state["window"] is not None:
            state["window"].destroy()
            state["window"] = None

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)


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
    """Apply contrast and brightness adjustments to a BGR frame.
    """
    import cv2

    mean = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
    alpha = brightness * contrast
    beta = brightness * (1 - contrast) * mean
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


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

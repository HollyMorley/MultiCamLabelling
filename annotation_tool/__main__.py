import tkinter as tk

from annotation_tool.gui.app import Navigator


def main():
    root = tk.Tk()
    Navigator(root)
    root.mainloop()


if __name__ == "__main__":
    main()

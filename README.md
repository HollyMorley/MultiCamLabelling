# MultiCamLabelling
### GUI for manual labelling across multiple calibrated cameras

Click [here](https://www.dropbox.com/scl/fo/ifppb8f3ss8z1ijvun3ry/AAAgY5bUVrrvuz4W0q-rYWA?rlkey=1tjdthbuqur7kuqb4a1t0bcza&dl=0) 
for link to dropbox folder with example file structure and pre-made calibration file for provided video.
(N.B. update the paths in config file to match your local directory structure.)

Create environment:
```bash
conda env create -f [your_path]/MultiCamLabelling.yml
```
Run the GUI:
```bash
# First set the "dir" path in MultiCamLabelling_config.py to the main directory where 
# the data is stored after download, e.g. "C:/Users/hmorl/Downloads/MultiCamLabelling"

# Then activate the environment and run the GUI
conda activate MultiCamLabelling
python [your_path]/labelling/MultiCamLabelling.py
```

### Main menu:
**Extraction:**
*Choose video frames to be labeled*
- Choose a video file from the pop-up file manager
- Scroll/skip through the synced videos to extract frame trios for labelling.
- In the case of frame misalignment across cameras, timestamps are used.

**Calibration:**
*Label known landmarks across all three camera views for camera pose estimation*
- Choose a video file from the pop-up file manager (N.B. pick a video which you have/will have extracted frames from)
- Scroll through the video and label the calibration points in each camera view.
- Calibration landmarks:
  - 4 corners of the first belt, corners of the starting step edge, and the 'x' sticker on the door
- Labelling controls:
  - Right-click --> place label
  - Right-click + shift --> delete nearest label
  - Left-click + mouse drag --> drag label

**Label:**
*Label pre-defined bodyparts with projection estimates of each point displayed between the 3 camera views*
- Choose a video folder under the CameraCalibration directory from the pop-up file manager to open the files from a video for which you have both extracted frames and added calibration labels. 
- Label view --> select the camera view to label
- Projection view --> select the camera view to calculate the estimated projection lines from 
  - If labels are present in the selected 'Projection View', estimated projection lines (from cam centre and crossing through the platform edges) will display on the other two views to guide labelling.
- Spacer lines --> Press once and right-click two points on the active labelling frame. 12 equally spaced lines along the x-axis will be displayed 
- Optimize calibration --> Selecting will adjust manually labeled calibration labels to minimize reprojection error between camera views and therefore improve estimated projections
- Save labels --> Save the labels to the video folder under the respective camera names (e.g. 'Side', 'Front', 'Overhead')
- Labelling controls:
  - Right-click --> place label
  - Right-click + shift --> delete nearest label
  - Left-click + mouse drag --> drag label
  - Hover --> label name

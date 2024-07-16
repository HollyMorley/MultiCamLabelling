# MultiCamLabelling
GUI for manual labelling across multiple calibrated cameras

Extraction:
- Select synced frames from videos from all 3 cameras ready for labelling. In the case of frame misalignment, timestamps are used.

Calibration:
- Label known landmarks across all three camera views for camera pose estimation

Label:
- Label predefined bodyparts.
- Labelling controls:
  - Right-click --> place label
  - Right-click + shift --> delete nearest label
  - Left-click + mouse drag --> drag label
  - Hover --> label name
- If labels are present in the selected 'Projection View', estimated projection lines (from cam centre and crossing through the platform edges) will display on the other two viewsto guide labelling.
- Spacer lines:
  - Press once, right-click two points on the active frame. 12 equally spaced lines will be displayed along the x-axis
- Optimize calibration:
  - Selecting will adjust manually labeled calibration labels to minimize reprojection error and therefore improve estimated projections

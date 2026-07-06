# Grid Step

Grid Step is a Python desktop app that watches a floor with a camera, maps the floor into a calibrated top-down grid, and reports which cells are being stepped on in real time.

The MVP prioritizes latency: camera capture runs on a dedicated latest-frame thread, detection uses OpenCV and NumPy over the calibrated floor ROI, and optional ONNX inference runs asynchronously only when the AI switch is enabled.

## Features

- SteamVR-style setup wizard: camera, floor area, floor plane, grid, background, detection
- Four-corner floor selection directly on the live camera image
- Homography-based top-down floor preview
- Configurable rows, columns, floor dimensions, thresholds, smoothing, and detection mode
- Empty-floor background capture
- Touch, pixel difference, shadow detection, and hybrid confidence modes
- Per-cell `IDLE`, `CANDIDATE`, `PRESSED`, `RELEASED` state management
- Real-time FPS and processing latency display
- Optional ONNX Runtime detector that fails gracefully when disabled, missing, or unavailable
- JSON settings saved in `grid_step/settings.json`

## Install

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not plan to use AI detection, the app still runs with AI switched off. `onnxruntime` is listed so the optional path is ready when a model is added.

## Run

```bash
python -m grid_step.main
```

## Basic Setup Flow

1. Click `Start Camera`.
2. Open `Setup Wizard`.
3. Select or confirm camera, resolution, and FPS.
4. Click the four corners of the usable floor area on the left camera view.
5. Drag the circular handles until the right preview looks like a clean top-down floor.
6. Set grid rows and columns.
7. Click `Pick Shoe Color`, then click the shoe in the left camera view.
8. Make sure nobody is on the floor, then click `Capture Background`.
9. Choose `Touch` for footstep-style input, or `Pixel`, `Shadow`, or `Hybrid` for raw testing.
10. Click `Start Detection`.

Pressed and released events are printed to the console:

```text
PRESSED: [(1, 2), (2, 2)]
RELEASED: [(1, 2)]
```

The current state is also held as a JSON-compatible object in `OutputManager`, so UDP or WebSocket output can be added later without changing the detector.

## Detection Modes

`Pixel` compares the current calibrated frame to the empty-floor background using `absdiff`, grayscale conversion, blur, thresholding, and morphology.

`Shadow` compares LAB lightness against the background with global brightness compensation, then filters small regions.

`Touch` is the default mode for footstep-style input. It separates "foot candidate inside a cell" from "touchdown-like timing" by combining shoe-color occupancy, background occupancy, local contact shadow, frame-to-frame motion, and short motion-stop history. Blue means `HOVER`, yellow means `TOUCH_CANDIDATE`, and green means `PRESSED`.

When a shoe color is picked, `Touch` mode uses that color as the first foot candidate. Shadow pixels are only counted near the detected shoe-color region, which helps reject body shadows far away from the actual foot.

`Hybrid` combines pixel and shadow confidence. If AI detection is off, AI weight is removed and the remaining weights are normalized. If AI detection is on, the app combines pixel, shadow, and the latest asynchronous ONNX result.

## AI ONNX Detection

AI detection is off by default.

To enable it:

1. Put a model at `models/foot_detector.onnx`, or choose another `.onnx` file with `Browse`.
2. Check `AI Detection`.
3. Start detection.

If the model is missing or ONNX Runtime cannot load, Grid Step shows a warning and continues using OpenCV detection. ONNX inference never blocks the camera or pixel/shadow pipeline; stale frames are dropped and only the latest AI result is used.

The current parser supports common box-like outputs shaped like `N x 5+` (`x0, y0, x1, y1, confidence`) and simple 2D masks. A real model may need a small adapter in `grid_step/onnx_detector.py`.

## Performance Tips

- Use 60 FPS camera input where possible.
- Keep the floor ROI tightly around the usable area.
- Start with `Touch` mode and AI off.
- If cells flicker green, raise `Press Score` to `0.75` or raise `Min Area`.
- If feet are visible but never press, lower `Press Score` to `0.60` or lower `Pixel Threshold`.
- Start `Shoe Threshold` around `24`. If the shoe is not detected, raise it to `30-40`. If pants or floor are mistaken for the shoe, lower it to `12-20` and pick the shoe color again.
- Lower camera resolution if the CPU cannot keep up.
- Keep `smoothing_frames` at 1 or 2 for fastest response.
- Recapture background after major lighting changes.

## Troubleshooting

- `Camera could not be opened`: check camera permissions, camera ID, and whether another app is using it.
- No pressed cells: lower `Pixel Threshold`, `Shadow Threshold`, or `Min Area`.
- Too many false positives: raise thresholds or recapture the empty-floor background.
- Skew warning: move the camera or adjust the four floor handles until the top-down preview is less distorted.
- AI warning: leave AI off, install `onnxruntime`, or provide a valid `.onnx` model path.

## Project Structure

```text
grid_step/
  main.py
  gui.py
  setup_wizard.py
  camera_manager.py
  calibration.py
  floor_plane.py
  detector.py
  onnx_detector.py
  grid_mapper.py
  state_manager.py
  output_manager.py
  config.py
  utils.py
  settings.json
requirements.txt
README.md
```

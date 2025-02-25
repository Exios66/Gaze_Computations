# Gaze Tracking System

This system allows you to track and analyze user gaze data using the GazeCloud API. It includes both tracking and analysis capabilities.

## Features

- Real-time gaze tracking using GazeCloud API
- Track gaze on any website or the current page
- Record eye positions, blinks, and head pose
- Generate comprehensive analysis reports including:
  - Fixation detection and metrics
  - Saccade detection and metrics
  - Blink detection and metrics
  - Head pose stability analysis
  - Visualization of gaze patterns

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/gaze-tracking-system.git
cd gaze-tracking-system
```

2. Install required Python packages:
```bash
pip install pandas numpy matplotlib seaborn
```

## Usage

### Recording Gaze Data

1. Start the tracking server:
```bash
python server.py
```

2. The gaze tracking interface will open in your default browser.

3. To track gaze on another website:
   - Enter the URL in the input field
   - Click "Start Tracking"
   - The tracking will begin on the specified website

4. To track gaze on the current page:
   - Leave the URL field empty
   - Click "Start Tracking"

5. When finished:
   - Click "Stop Tracking"
   - Click "Save Data" to download the recorded data as CSV

### Analyzing Gaze Data

1. Place your recorded gaze data CSV file in the `data/examples/` directory.

2. Run the analysis script:
```bash
python gaze.py
```

3. The script will generate:
   - Metrics in JSON format
   - Visualization plots including:
     - Gaze trajectory with events
     - Head pose over time
     - Event timeline

4. Results will be saved in the `reports/` directory.

## Data Format

The recorded CSV files contain the following columns:
- timestamp: ISO format timestamp
- frame_number: Sequential frame counter
- left_eye_x, left_eye_y: Left eye position
- right_eye_x, right_eye_y: Right eye position
- left_pupil_size, right_pupil_size: Pupil sizes
- left_blink, right_blink: Blink detection (0/1)
- gaze_x, gaze_y: Computed gaze position
- head_pose_x, head_pose_y, head_pose_z: Head orientation
- marker: Optional event marker

## Analysis Parameters

You can adjust various parameters in the `gaze.py` script:
- Saccade detection thresholds
- Fixation detection parameters
- Blink detection sensitivity
- Visualization settings

## Contributing

Feel free to submit issues and enhancement requests!

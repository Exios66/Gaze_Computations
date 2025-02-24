import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_reading_pattern(n_samples, screen_width, screen_height):
    """Generate a realistic reading pattern with line sweeps and returns."""
    t = np.linspace(0, 1, n_samples)
    
    # Number of lines to read
    n_lines = 10
    line_height = screen_height / (n_lines + 2)  # Leave margins
    line_start = screen_width * 0.1  # Left margin
    line_end = screen_width * 0.9   # Right margin
    
    # Initialize coordinates
    x_coords = np.zeros(n_samples)
    y_coords = np.zeros(n_samples)
    
    # Generate reading pattern
    samples_per_line = n_samples // n_lines
    for i in range(n_lines):
        start_idx = i * samples_per_line
        end_idx = (i + 1) * samples_per_line
        
        # Forward reading sweep
        x_coords[start_idx:end_idx] = np.linspace(
            line_start, line_end, samples_per_line
        )
        
        # Add small vertical variations within line
        y_coords[start_idx:end_idx] = (i + 1) * line_height + np.random.normal(
            0, line_height * 0.05, samples_per_line
        )
        
        # Add quick return saccade at end of line
        if i < n_lines - 1:
            x_coords[end_idx-5:end_idx] = np.linspace(line_end, line_start, 5)
            y_coords[end_idx-5:end_idx] = np.linspace(
                (i + 1) * line_height, (i + 2) * line_height, 5
            )
    
    return x_coords, y_coords

def generate_sample_gaze_data(duration_seconds=60, sampling_rate=60):
    """
    Generate synthetic gaze data for testing.
    
    Args:
        duration_seconds (int): Duration of recording in seconds
        sampling_rate (int): Samples per second
    """
    n_samples = duration_seconds * sampling_rate
    
    # Generate timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(milliseconds=i*(1000/sampling_rate)) 
                 for i in range(n_samples)]
    
    # Screen dimensions (assuming 1920x1080)
    screen_width = 1920
    screen_height = 1080
    
    # Generate reading pattern
    gaze_x, gaze_y = generate_reading_pattern(n_samples, screen_width, screen_height)
    
    # Add natural jitter/noise
    gaze_x += np.random.normal(0, 5, n_samples)  # 5 pixels standard deviation
    gaze_y += np.random.normal(0, 5, n_samples)
    
    # Generate eye positions (slightly offset from gaze position)
    eye_distance = 60  # pixels
    left_eye_x = gaze_x - eye_distance/2 + np.random.normal(0, 2, n_samples)
    left_eye_y = gaze_y + np.random.normal(0, 2, n_samples)
    right_eye_x = gaze_x + eye_distance/2 + np.random.normal(0, 2, n_samples)
    right_eye_y = gaze_y + np.random.normal(0, 2, n_samples)
    
    # Generate pupil sizes with natural fluctuations
    t = np.linspace(0, duration_seconds, n_samples)
    base_pupil_size = 3 + 0.2 * np.sin(2*np.pi*0.1*t)  # Slow natural fluctuation
    
    # Add occasional pupil dilations
    dilation_events = np.random.rand(n_samples) > 0.95
    base_pupil_size[dilation_events] += np.random.uniform(1, 2, sum(dilation_events))
    
    # Add high-frequency noise
    pupil_noise = np.random.normal(0, 0.1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
        'left_eye_x': left_eye_x,
        'left_eye_y': left_eye_y,
        'right_eye_x': right_eye_x,
        'right_eye_y': right_eye_y,
        'left_pupil_size': base_pupil_size + pupil_noise,
        'right_pupil_size': base_pupil_size + np.random.normal(0, 0.1, n_samples),
        'confidence': np.random.uniform(0.8, 1.0, n_samples)
    })
    
    return df

def generate_test_datasets():
    """Generate various test datasets with different characteristics."""
    # Normal reading pattern
    normal_data = generate_sample_gaze_data(60, 60)
    normal_data.to_csv('data/examples/normal_reading.csv', index=False)
    
    # Quick reading (faster saccades)
    quick_data = generate_sample_gaze_data(30, 120)
    quick_data.to_csv('data/examples/quick_reading.csv', index=False)
    
    # Low quality data (more noise and lower confidence)
    low_quality = generate_sample_gaze_data(60, 60)
    low_quality['confidence'] = np.random.uniform(0.3, 0.7, len(low_quality))
    noise = np.random.normal(0, 50, len(low_quality))
    low_quality['gaze_x'] += noise
    low_quality['gaze_y'] += noise
    low_quality.to_csv('data/examples/low_quality.csv', index=False)

if __name__ == "__main__":
    generate_test_datasets()
    print("Test datasets generated successfully in data/examples/") 
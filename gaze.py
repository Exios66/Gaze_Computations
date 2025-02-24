import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

class GazeAnalyzer:
    def __init__(self, file_path, min_confidence=0.5):
        """
        Initialize GazeAnalyzer with data and configuration.
        
        Args:
            file_path (str): Path to the gaze data CSV file
            min_confidence (float): Minimum confidence threshold for valid data
        """
        self.data = pd.read_csv(file_path, parse_dates=['timestamp'])
        self.file_name = os.path.basename(file_path)
        self.min_confidence = min_confidence
        self.metrics = {}
        self.clean_data()
    
    def clean_data(self):
        """Clean and validate gaze data."""
        # Remove low confidence data
        self.data = self.data[self.data['confidence'] >= self.min_confidence]
        
        # Remove physiologically impossible values
        self.data = self.data[
            (self.data['left_eye_x'] > 0) & 
            (self.data['right_eye_x'] > 0) &
            (self.data['left_pupil_size'] > 0) &
            (self.data['right_pupil_size'] > 0)
        ]
        
        # Calculate time differences
        self.data['timestamp_diff'] = self.data['timestamp'].diff().dt.total_seconds()
        self.sampling_rate = 1 / self.data['timestamp_diff'].median()
        
        # Compute vergence and version
        self.data['vergence'] = np.abs(self.data['left_eye_x'] - self.data['right_eye_x'])
        self.data['version'] = (self.data['left_eye_x'] + self.data['right_eye_x']) / 2
    
    def compute_saccades(self, velocity_threshold=30, acceleration_threshold=8000):
        """
        Detects saccades using velocity and acceleration thresholds.
        
        Args:
            velocity_threshold (float): Minimum velocity for saccade detection (degrees/second)
            acceleration_threshold (float): Minimum acceleration for saccade detection (degrees/second²)
        """
        # Calculate velocity and acceleration
        self.data['velocity'] = np.sqrt(
            (self.data['gaze_x'].diff() / self.data['timestamp_diff'])**2 +
            (self.data['gaze_y'].diff() / self.data['timestamp_diff'])**2
        )
        
        self.data['acceleration'] = self.data['velocity'].diff() / self.data['timestamp_diff']
        
        # Detect saccades with more conservative thresholds
        self.data['saccade'] = (
            (self.data['velocity'] > velocity_threshold) &
            (np.abs(self.data['acceleration']) > acceleration_threshold)
        )
        
        # Apply minimum duration filter (remove very short saccades)
        saccade_groups = self.data['saccade'].ne(self.data['saccade'].shift()).cumsum()
        for group_id in saccade_groups.unique():
            group_data = self.data[saccade_groups == group_id]
            if group_data['saccade'].iloc[0] and len(group_data) < 3:  # Minimum 3 samples
                self.data.loc[group_data.index, 'saccade'] = False
        
        # Recompute metrics
        saccade_groups = self.data['saccade'].ne(self.data['saccade'].shift()).cumsum()
        saccade_data = self.data[self.data['saccade']].groupby(saccade_groups)
        
        self.metrics['saccades'] = {
            'count': len(saccade_data),
            'mean_velocity': self.data[self.data['saccade']]['velocity'].mean(),
            'mean_duration': saccade_data.size().mean() / self.sampling_rate if len(saccade_data) > 0 else 0
        }
        
        return self.data[['timestamp', 'saccade', 'velocity', 'acceleration']]
    
    def compute_fixations(self, dispersion_threshold=50, duration_threshold=0.1):
        """
        Detects fixations using dispersion-based algorithm.
        
        Args:
            dispersion_threshold (float): Maximum dispersion for fixation detection (pixels)
            duration_threshold (float): Minimum duration for fixation detection (seconds)
        """
        # Start with non-saccadic periods
        self.data['fixation'] = ~self.data['saccade']
        
        # Compute fixation groups
        fixation_groups = self.data['fixation'].ne(self.data['fixation'].shift()).cumsum()
        
        # Calculate dispersion for each potential fixation
        for group_id in fixation_groups.unique():
            group_data = self.data[fixation_groups == group_id]
            if not group_data['fixation'].iloc[0]:
                continue
                
            duration = group_data['timestamp_diff'].sum()
            dispersion = np.sqrt(
                (group_data['gaze_x'].max() - group_data['gaze_x'].min())**2 +
                (group_data['gaze_y'].max() - group_data['gaze_y'].min())**2
            )
            
            # Mark as not fixation if criteria not met
            if duration < duration_threshold or dispersion > dispersion_threshold:
                self.data.loc[group_data.index, 'fixation'] = False
        
        # Merge very close fixations (gaps less than 50ms)
        fixation_groups = self.data['fixation'].ne(self.data['fixation'].shift()).cumsum()
        for i in range(1, fixation_groups.max()):
            gap_start = self.data[fixation_groups == i].index[-1] + 1
            gap_end = self.data[fixation_groups == (i + 1)].index[0] - 1
            
            if gap_end > gap_start:
                gap_duration = self.data.loc[gap_start:gap_end, 'timestamp_diff'].sum()
                if gap_duration < 0.05:  # 50ms threshold
                    self.data.loc[gap_start:gap_end, 'fixation'] = True
        
        # Recompute fixation groups and metrics
        fixation_groups = self.data['fixation'].ne(self.data['fixation'].shift()).cumsum()
        fixation_data = self.data[self.data['fixation']].groupby(fixation_groups)
        
        self.metrics['fixations'] = {
            'count': len(fixation_data),
            'mean_duration': fixation_data.size().mean() / self.sampling_rate if len(fixation_data) > 0 else 0,
            'total_duration': self.data['fixation'].sum() / self.sampling_rate
        }
        
        return self.data[['timestamp', 'fixation']]
    
    def compute_pupil_metrics(self):
        """Compute pupil-related metrics."""
        self.metrics['pupil'] = {
            'mean_left_size': self.data['left_pupil_size'].mean(),
            'mean_right_size': self.data['right_pupil_size'].mean(),
            'std_left_size': self.data['left_pupil_size'].std(),
            'std_right_size': self.data['right_pupil_size'].std(),
            'dilation_events': sum(
                (self.data['left_pupil_size'] > self.data['left_pupil_size'].mean() + 2 * self.data['left_pupil_size'].std()) |
                (self.data['right_pupil_size'] > self.data['right_pupil_size'].mean() + 2 * self.data['right_pupil_size'].std())
            )
        }
        
        return self.data[['timestamp', 'left_pupil_size', 'right_pupil_size']]
    
    def compute_aois(self, aois):
        """
        Identifies whether gaze falls within predefined Areas of Interest (AOIs).
        
        Args:
            aois (dict): Dictionary of AOI names and their boundaries (x_min, x_max, y_min, y_max)
        """
        self.data['aoi'] = 'none'
        aoi_durations = {}
        
        for aoi_name, (x_min, x_max, y_min, y_max) in aois.items():
            mask = (
                (self.data['gaze_x'] >= x_min) & 
                (self.data['gaze_x'] <= x_max) & 
                (self.data['gaze_y'] >= y_min) & 
                (self.data['gaze_y'] <= y_max)
            )
            self.data.loc[mask, 'aoi'] = aoi_name
            aoi_durations[aoi_name] = sum(mask) / self.sampling_rate
        
        self.metrics['aois'] = aoi_durations
        return self.data[['timestamp', 'aoi']]
    
    def generate_report(self, output_dir='reports'):
        """Generate a comprehensive report of all metrics."""
        # Create report directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Gaze trajectory plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['gaze_x'], self.data['gaze_y'], 'b-', alpha=0.5)
        plt.scatter(
            self.data[self.data['fixation']]['gaze_x'],
            self.data[self.data['fixation']]['gaze_y'],
            c='r', alpha=0.5, label='Fixations'
        )
        plt.xlabel('Gaze X')
        plt.ylabel('Gaze Y')
        plt.title('Gaze Trajectory with Fixations')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plot_dir, f'{self.file_name}_trajectory.png'))
        plt.close()
        
        # Heatmap
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=self.data[self.data['fixation']], 
            x='gaze_x', y='gaze_y',
            cmap='hot', fill=True
        )
        plt.title('Fixation Heatmap')
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plot_dir, f'{self.file_name}_heatmap.png'))
        plt.close()
        
        # Pupil size over time
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['timestamp'], self.data['left_pupil_size'], 
                label='Left Pupil')
        plt.plot(self.data['timestamp'], self.data['right_pupil_size'], 
                label='Right Pupil')
        plt.xlabel('Time')
        plt.ylabel('Pupil Size')
        plt.title('Pupil Size Over Time')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{self.file_name}_pupil.png'))
        plt.close()
        
        # Save metrics to JSON
        report_file = os.path.join(output_dir, f'{self.file_name}_metrics.json')
        with open(report_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        return report_file

# Example usage:
if __name__ == "__main__":
    # Generate example data if it doesn't exist
    if not os.path.exists('data/examples/normal_reading.csv'):
        from data.examples.generate_test_data import generate_test_datasets
        generate_test_datasets()
    
    # Analyze example data
    analyzer = GazeAnalyzer('data/examples/normal_reading.csv')
    
    # Compute all metrics
    analyzer.compute_saccades()
    analyzer.compute_fixations()
    analyzer.compute_pupil_metrics()
    analyzer.compute_aois({
        "top_left": (0, 640, 0, 360),
        "top_right": (640, 1280, 0, 360),
        "bottom_left": (0, 640, 360, 720),
        "bottom_right": (640, 1280, 360, 720)
    })
    
    # Generate report
    report_file = analyzer.generate_report()
    print(f"Analysis complete. Report saved to {report_file}")

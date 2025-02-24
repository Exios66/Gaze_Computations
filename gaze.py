import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

class GazeAnalyzer:
    def __init__(self, file_path):
        """
        Initialize GazeAnalyzer with data and configuration.
        
        Args:
            file_path (str): Path to the gaze data CSV file
        """
        self.data = pd.read_csv(file_path, parse_dates=['timestamp'])
        self.file_name = os.path.basename(file_path)
        self.metrics = {}
        self.clean_data()
    
    def clean_data(self):
        """Clean and validate gaze data."""
        # Calculate time differences
        self.data['timestamp_diff'] = self.data['timestamp'].diff().dt.total_seconds()
        self.sampling_rate = 1 / self.data['timestamp_diff'].median()
        
        # Compute gaze position as average of left and right eye when not provided
        mask_no_gaze = (self.data['gaze_x'] == 0) & (self.data['gaze_y'] == 0)
        self.data.loc[mask_no_gaze, 'gaze_x'] = (self.data.loc[mask_no_gaze, 'left_eye_x'] + 
                                                self.data.loc[mask_no_gaze, 'right_eye_x']) / 2
        self.data.loc[mask_no_gaze, 'gaze_y'] = (self.data.loc[mask_no_gaze, 'left_eye_y'] + 
                                                self.data.loc[mask_no_gaze, 'right_eye_y']) / 2
        
        # Compute vergence and version
        self.data['vergence'] = np.sqrt(
            (self.data['left_eye_x'] - self.data['right_eye_x'])**2 +
            (self.data['left_eye_y'] - self.data['right_eye_y'])**2
        )
        
        # Detect blinks if not provided
        if self.data['left_blink'].sum() == 0 and self.data['right_blink'].sum() == 0:
            self.detect_blinks()
    
    def detect_blinks(self, vergence_threshold=200):
        """Detect blinks based on sudden changes in vergence."""
        # Detect rapid changes in vergence
        vergence_velocity = np.abs(self.data['vergence'].diff() / self.data['timestamp_diff'])
        
        # Mark as blinks when vergence changes rapidly
        self.data['left_blink'] = vergence_velocity > vergence_threshold
        self.data['right_blink'] = self.data['left_blink']  # Assuming binocular blinks
        
        # Extend blink duration
        blink_groups = self.data['left_blink'].ne(self.data['left_blink'].shift()).cumsum()
        for group_id in blink_groups[self.data['left_blink']].unique():
            group_data = self.data[blink_groups == group_id]
            if len(group_data) < 3:  # Extend very short blinks
                idx_before = group_data.index[0] - 1
                idx_after = group_data.index[-1] + 1
                if idx_before in self.data.index:
                    self.data.loc[idx_before, ['left_blink', 'right_blink']] = True
                if idx_after in self.data.index:
                    self.data.loc[idx_after, ['left_blink', 'right_blink']] = True
    
    def compute_saccades(self, velocity_threshold=300, acceleration_threshold=8000):
        """
        Detects saccades using velocity and acceleration thresholds.
        
        Args:
            velocity_threshold (float): Minimum velocity for saccade detection (pixels/second)
            acceleration_threshold (float): Minimum acceleration for saccade detection (pixels/second²)
        """
        # Don't detect saccades during blinks
        self.data['velocity'] = np.where(
            self.data['left_blink'] | self.data['right_blink'],
            0,
            np.sqrt(
                (self.data['gaze_x'].diff() / self.data['timestamp_diff'])**2 +
                (self.data['gaze_y'].diff() / self.data['timestamp_diff'])**2
            )
        )
        
        self.data['acceleration'] = self.data['velocity'].diff() / self.data['timestamp_diff']
        
        # Detect saccades
        self.data['saccade'] = (
            ~(self.data['left_blink'] | self.data['right_blink']) &
            (self.data['velocity'] > velocity_threshold) &
            (np.abs(self.data['acceleration']) > acceleration_threshold)
        )
        
        # Apply minimum duration filter
        saccade_groups = self.data['saccade'].ne(self.data['saccade'].shift()).cumsum()
        for group_id in saccade_groups.unique():
            group_data = self.data[saccade_groups == group_id]
            if group_data['saccade'].iloc[0] and len(group_data) < 3:
                self.data.loc[group_data.index, 'saccade'] = False
        
        # Compute metrics
        saccade_groups = self.data['saccade'].ne(self.data['saccade'].shift()).cumsum()
        saccade_data = self.data[self.data['saccade']].groupby(saccade_groups)
        
        self.metrics['saccades'] = {
            'count': len(saccade_data),
            'mean_velocity': self.data[self.data['saccade']]['velocity'].mean(),
            'mean_duration': saccade_data.size().mean() / self.sampling_rate if len(saccade_data) > 0 else 0,
            'mean_amplitude': saccade_data.apply(
                lambda x: np.sqrt(
                    (x['gaze_x'].iloc[-1] - x['gaze_x'].iloc[0])**2 +
                    (x['gaze_y'].iloc[-1] - x['gaze_y'].iloc[0])**2
                )
            ).mean() if len(saccade_data) > 0 else 0
        }
        
        return self.data[['timestamp', 'saccade', 'velocity', 'acceleration']]
    
    def compute_fixations(self, dispersion_threshold=50, duration_threshold=0.1):
        """
        Detects fixations using dispersion-based algorithm.
        
        Args:
            dispersion_threshold (float): Maximum dispersion for fixation detection (pixels)
            duration_threshold (float): Minimum duration for fixation detection (seconds)
        """
        # Start with non-saccadic and non-blink periods
        self.data['fixation'] = ~(self.data['saccade'] | self.data['left_blink'] | self.data['right_blink'])
        
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
            
            if duration < duration_threshold or dispersion > dispersion_threshold:
                self.data.loc[group_data.index, 'fixation'] = False
        
        # Compute metrics
        fixation_groups = self.data['fixation'].ne(self.data['fixation'].shift()).cumsum()
        fixation_data = self.data[self.data['fixation']].groupby(fixation_groups)
        
        self.metrics['fixations'] = {
            'count': len(fixation_data),
            'mean_duration': fixation_data.size().mean() / self.sampling_rate if len(fixation_data) > 0 else 0,
            'total_duration': self.data['fixation'].sum() / self.sampling_rate,
            'mean_dispersion': fixation_data.apply(
                lambda x: np.sqrt(
                    (x['gaze_x'].max() - x['gaze_x'].min())**2 +
                    (x['gaze_y'].max() - x['gaze_y'].min())**2
                )
            ).mean() if len(fixation_data) > 0 else 0
        }
        
        return self.data[['timestamp', 'fixation']]
    
    def compute_blink_metrics(self):
        """Compute blink-related metrics."""
        blink_groups = self.data['left_blink'].ne(self.data['left_blink'].shift()).cumsum()
        blink_data = self.data[self.data['left_blink']].groupby(blink_groups)
        
        self.metrics['blinks'] = {
            'count': len(blink_data),
            'mean_duration': blink_data.size().mean() / self.sampling_rate if len(blink_data) > 0 else 0,
            'total_duration': self.data['left_blink'].sum() / self.sampling_rate,
            'rate_per_minute': len(blink_data) / (self.data['timestamp_diff'].sum() / 60)
        }
        
        return self.data[['timestamp', 'left_blink', 'right_blink']]
    
    def compute_head_pose_metrics(self):
        """Compute head pose stability metrics."""
        self.metrics['head_pose'] = {
            'mean_x': self.data['head_pose_x'].mean(),
            'mean_y': self.data['head_pose_y'].mean(),
            'mean_z': self.data['head_pose_z'].mean(),
            'std_x': self.data['head_pose_x'].std(),
            'std_y': self.data['head_pose_y'].std(),
            'std_z': self.data['head_pose_z'].std()
        }
        
        return self.data[['timestamp', 'head_pose_x', 'head_pose_y', 'head_pose_z']]
    
    def generate_report(self, output_dir='reports'):
        """Generate a comprehensive report of all metrics."""
        os.makedirs(output_dir, exist_ok=True)
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Gaze trajectory with events
        plt.figure(figsize=(12, 8))
        plt.plot(self.data['gaze_x'], self.data['gaze_y'], 'b-', alpha=0.3, label='Gaze Path')
        plt.scatter(
            self.data[self.data['fixation']]['gaze_x'],
            self.data[self.data['fixation']]['gaze_y'],
            c='g', alpha=0.5, s=50, label='Fixations'
        )
        plt.scatter(
            self.data[self.data['saccade']]['gaze_x'],
            self.data[self.data['saccade']]['gaze_y'],
            c='r', alpha=0.5, s=20, label='Saccades'
        )
        plt.scatter(
            self.data[self.data['left_blink']]['gaze_x'],
            self.data[self.data['left_blink']]['gaze_y'],
            c='k', alpha=0.5, s=30, label='Blinks'
        )
        plt.xlabel('Gaze X (pixels)')
        plt.ylabel('Gaze Y (pixels)')
        plt.title('Gaze Trajectory with Events')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plot_dir, f'{self.file_name}_trajectory.png'))
        plt.close()
        
        # Head pose over time
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['timestamp'], self.data['head_pose_x'], label='X')
        plt.plot(self.data['timestamp'], self.data['head_pose_y'], label='Y')
        plt.plot(self.data['timestamp'], self.data['head_pose_z'], label='Z')
        plt.xlabel('Time')
        plt.ylabel('Head Pose (degrees)')
        plt.title('Head Pose Over Time')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{self.file_name}_head_pose.png'))
        plt.close()
        
        # Event timeline
        plt.figure(figsize=(12, 4))
        plt.plot(self.data['timestamp'], self.data['fixation'], label='Fixation')
        plt.plot(self.data['timestamp'], self.data['saccade'], label='Saccade')
        plt.plot(self.data['timestamp'], self.data['left_blink'], label='Blink')
        plt.xlabel('Time')
        plt.ylabel('Event')
        plt.title('Eye Movement Events Timeline')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{self.file_name}_events.png'))
        plt.close()
        
        # Save metrics to JSON
        report_file = os.path.join(output_dir, f'{self.file_name}_metrics.json')
        with open(report_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        return report_file

# Example usage:
if __name__ == "__main__":
    analyzer = GazeAnalyzer('data/examples/gaze_recording.csv')
    
    # Compute all metrics
    analyzer.compute_saccades()
    analyzer.compute_fixations()
    analyzer.compute_blink_metrics()
    analyzer.compute_head_pose_metrics()
    
    # Generate report
    report_file = analyzer.generate_report()
    print(f"Analysis complete. Report saved to {report_file}")

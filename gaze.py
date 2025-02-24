import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GazeAnalyzer:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, parse_dates=['timestamp'])
        self.clean_data()
    
    def clean_data(self):
        # Handling missing values and incorrect data
        self.data.dropna(inplace=True)
        self.data = self.data[(self.data['left_eye_x'] > 0) & (self.data['right_eye_x'] > 0)]
    
    def compute_saccades(self, threshold=30):
        """Detects saccades based on gaze shift threshold."""
        self.data['gaze_shift'] = np.sqrt((self.data['gaze_x'].diff() ** 2) + (self.data['gaze_y'].diff() ** 2))
        self.data['saccade'] = self.data['gaze_shift'] > threshold
        return self.data[['timestamp', 'saccade']]
    
    def compute_fixations(self, duration_threshold=100):
        """Detects fixations by checking stable gaze regions."""
        self.data['fixation'] = ~self.data['saccade']
        self.data['fixation_duration'] = self.data['fixation'].astype(int).groupby(self.data['fixation'].ne(self.data['fixation'].shift()).cumsum()).cumsum()
        self.data['fixation'] = self.data['fixation_duration'] > duration_threshold
        return self.data[['timestamp', 'fixation']]
    
    def compute_aois(self, aois):
        """Identifies whether gaze falls within predefined Areas of Interest (AOIs)."""
        self.data['aoi'] = 'None'
        for aoi_name, (x_min, x_max, y_min, y_max) in aois.items():
            mask = (self.data['gaze_x'] >= x_min) & (self.data['gaze_x'] <= x_max) & (self.data['gaze_y'] >= y_min) & (self.data['gaze_y'] <= y_max)
            self.data.loc[mask, 'aoi'] = aoi_name
        return self.data[['timestamp', 'aoi']]
    
    def plot_gaze_trajectory(self):
        """Plots gaze trajectory over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['gaze_x'], self.data['gaze_y'], marker='o', linestyle='-', alpha=0.5)
        plt.xlabel("Gaze X")
        plt.ylabel("Gaze Y")
        plt.title("Gaze Trajectory")
        plt.gca().invert_yaxis()
        plt.show()
    
    def generate_heatmap(self):
        """Generates a heatmap of gaze fixations."""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(x=self.data['gaze_x'], y=self.data['gaze_y'], cmap="Reds", fill=True, alpha=0.5)
        plt.xlabel("Gaze X")
        plt.ylabel("Gaze Y")
        plt.title("Gaze Heatmap")
        plt.gca().invert_yaxis()
        plt.show()

# Usage Example
# analyzer = GazeAnalyzer('gaze_data.csv')
# analyzer.compute_saccades()
# analyzer.compute_fixations()
# analyzer.compute_aois({"Center": (500, 700, 200, 400)})
# analyzer.plot_gaze_trajectory()
# analyzer.generate_heatmap()

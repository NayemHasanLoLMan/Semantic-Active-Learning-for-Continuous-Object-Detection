import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

class MetricsTracker:
    def __init__(self):
        self.history_file = "models/training_history.json"
        self.log_file = "logs/training_logs/training_log.json"
    
    def load_history(self):
        """Load training history."""
        if not os.path.exists(self.history_file):
            return []
        
        with open(self.history_file, 'r') as f:
            return json.load(f)
    
    def get_latest_metrics(self):
        """Get metrics from the latest training."""
        history = self.load_history()
        if not history:
            return None
        
        return history[-1]['metrics']
    
    def get_metric_trend(self, metric_name='map50'):
        """Get trend of a specific metric over time."""
        history = self.load_history()
        
        timestamps = []
        values = []
        
        for entry in history:
            timestamps.append(entry['timestamp'])
            values.append(entry['metrics'].get(metric_name, 0))
        
        return timestamps, values
    
    def plot_metrics(self, save_path="logs/metrics_plot.png"):
        """Plot all metrics over time."""
        history = self.load_history()
        
        if not history:
            print(" No training history found")
            return
        
        metrics = ['precision', 'recall', 'map50', 'map']
        data = {m: [] for m in metrics}
        timestamps = []
        
        for entry in history:
            timestamps.append(entry['timestamp'])
            for metric in metrics:
                data[metric].append(entry['metrics'].get(metric, 0))
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Over Time', fontsize=16)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            ax.plot(range(len(data[metric])), data[metric], marker='o')
            ax.set_title(metric.upper())
            ax.set_xlabel('Training Session')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved metrics plot to: {save_path}")
        
        plt.close()
    
    def generate_report(self):
        """Generate a text report of model performance."""
        history = self.load_history()
        
        if not history:
            print(" No training history found")
            return
        
        print("\n" + "="*60)
        print(" MODEL PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\nTotal Training Sessions: {len(history)}")
        
        # Latest metrics
        latest = history[-1]
        print(f"\n Latest Model ({latest['timestamp']}):")
        print("-"*60)
        for metric, value in latest['metrics'].items():
            print(f"  {metric:<15}: {value:.4f}")
        
        # Best metrics
        print(f"\n Best Performance:")
        print("-"*60)
        for metric in ['precision', 'recall', 'map50', 'map']:
            values = [h['metrics'].get(metric, 0) for h in history]
            best_value = max(values)
            best_idx = values.index(best_value)
            best_session = history[best_idx]['timestamp']
            print(f"  {metric:<15}: {best_value:.4f} (Session: {best_session})")
        
        # Improvement
        if len(history) > 1:
            first = history[0]
            print(f"\n Overall Improvement:")
            print("-"*60)
            for metric in ['precision', 'recall', 'map50', 'map']:
                first_val = first['metrics'].get(metric, 0)
                latest_val = latest['metrics'].get(metric, 0)
                improvement = latest_val - first_val
                improvement_pct = (improvement / first_val * 100) if first_val > 0 else 0
                symbol = "📈" if improvement > 0 else "📉"
                print(f"  {metric:<15}: {improvement:+.4f} ({improvement_pct:+.1f}%) {symbol}")
        
        print("="*60)

if __name__ == "__main__":
    tracker = MetricsTracker()
    tracker.generate_report()
    tracker.plot_metrics()
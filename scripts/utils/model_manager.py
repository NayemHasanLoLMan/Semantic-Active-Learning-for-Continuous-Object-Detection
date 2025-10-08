import os
import shutil
from datetime import datetime
import json
from pathlib import Path

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.versions_dir = f"{models_dir}/previous_versions"
        self.history_file = f"{models_dir}/training_history.json"
    
    def get_all_versions(self):
        """Get list of all model versions."""
        if not os.path.exists(self.versions_dir):
            return []
        
        versions = []
        for f in os.listdir(self.versions_dir):
            if f.endswith('.pt'):
                timestamp = f.replace('model_', '').replace('.pt', '')
                versions.append({
                    'filename': f,
                    'timestamp': timestamp,
                    'path': f"{self.versions_dir}/{f}"
                })
        
        return sorted(versions, key=lambda x: x['timestamp'], reverse=True)
    
    def get_training_history(self):
        """Load training history."""
        if not os.path.exists(self.history_file):
            return []
        
        with open(self.history_file, 'r') as f:
            return json.load(f)
    
    def rollback_to_version(self, version_filename):
        """Rollback to a previous model version."""
        version_path = f"{self.versions_dir}/{version_filename}"
        
        if not os.path.exists(version_path):
            print(f" Version not found: {version_filename}")
            return False
        
        # Backup current model
        current_model = f"{self.models_dir}/current_best.pt"
        if os.path.exists(current_model):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.versions_dir}/backup_{timestamp}.pt"
            shutil.copy2(current_model, backup_path)
            print(f" Backed up current model to: {backup_path}")
        
        # Restore version
        shutil.copy2(version_path, current_model)
        print(f" Rolled back to: {version_filename}")
        
        return True
    
    def compare_versions(self, version1, version2):
        """Compare metrics between two versions."""
        history = self.get_training_history()
        
        v1_metrics = None
        v2_metrics = None
        
        for entry in history:
            if version1 in entry.get('timestamp', ''):
                v1_metrics = entry['metrics']
            if version2 in entry.get('timestamp', ''):
                v2_metrics = entry['metrics']
        
        if not v1_metrics or not v2_metrics:
            print(" Could not find metrics for comparison")
            return
        
        print("\n Version Comparison:")
        print("="*60)
        print(f"{'Metric':<15} {version1:<20} {version2:<20}")
        print("-"*60)
        
        for metric in ['precision', 'recall', 'map50', 'map']:
            v1 = v1_metrics.get(metric, 0)
            v2 = v2_metrics.get(metric, 0)
            diff = v2 - v1
            symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            print(f"{metric:<15} {v1:<20.4f} {v2:<20.4f} {symbol} {diff:+.4f}")
    
    def clean_old_versions(self, keep_last=10):
        """Remove old model versions, keep only recent ones."""
        versions = self.get_all_versions()
        
        if len(versions) <= keep_last:
            print(f"Only {len(versions)} versions, nothing to clean")
            return
        
        to_remove = versions[keep_last:]
        
        for version in to_remove:
            os.remove(version['path'])
            print(f"  Removed old version: {version['filename']}")
        
        print(f" Cleaned {len(to_remove)} old versions")

if __name__ == "__main__":
    manager = ModelManager()
    versions = manager.get_all_versions()
    
    print(" Available Model Versions:")
    print("="*50)
    for v in versions:
        print(f"  {v['filename']}")
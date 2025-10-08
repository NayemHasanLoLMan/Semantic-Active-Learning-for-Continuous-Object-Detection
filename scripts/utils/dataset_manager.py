import os
import shutil
from pathlib import Path
import json

class DatasetManager:
    def __init__(self):
        self.base_dir = "datasets"
    
    def get_dataset_stats(self):
        """Get statistics about all datasets."""
        stats = {
            'initial_dataset': self.count_images('datasets/initial_dataset/images/train'),
            'pending_verification': self.count_images('datasets/captured_data/pending_verification/images'),
            'verified_positive': self.count_images('datasets/captured_data/verified_positive/images'),
            'verified_negative': self.count_images('datasets/captured_data/verified_negative/images'),
        }
        return stats
    
    def count_images(self, directory):
        """Count images in a directory."""
        if not os.path.exists(directory):
            return 0
        return len([f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    def clean_empty_labels(self, dataset_path):
        """Remove images that don't have corresponding labels."""
        images_dir = f"{dataset_path}/images"
        labels_dir = f"{dataset_path}/labels"
        
        removed = 0
        for split in ['train', 'val']:
            img_path = f"{images_dir}/{split}"
            lbl_path = f"{labels_dir}/{split}"
            
            if not os.path.exists(img_path):
                continue
            
            for img_file in Path(img_path).glob('*.jpg'):
                lbl_file = Path(lbl_path) / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    os.remove(img_file)
                    removed += 1
                    print(f"Removed: {img_file.name} (no label)")
        
        return removed
    
    def archive_old_batches(self, keep_last=5):
        """Archive old training batches, keep only recent ones."""
        batches_dir = "datasets/training_batches"
        if not os.path.exists(batches_dir):
            return
        
        batches = sorted([d for d in os.listdir(batches_dir) if d.startswith('batch_')])
        
        if len(batches) > keep_last:
            archive_dir = "datasets/archived_batches"
            os.makedirs(archive_dir, exist_ok=True)
            
            for old_batch in batches[:-keep_last]:
                old_path = f"{batches_dir}/{old_batch}"
                archive_path = f"{archive_dir}/{old_batch}"
                shutil.move(old_path, archive_path)
                print(f"Archived: {old_batch}")

if __name__ == "__main__":
    manager = DatasetManager()
    stats = manager.get_dataset_stats()
    
    print(" Dataset Statistics:")
    print("="*50)
    for name, count in stats.items():
        print(f"  {name}: {count} images")
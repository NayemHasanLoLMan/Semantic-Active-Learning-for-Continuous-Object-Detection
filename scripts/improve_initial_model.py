import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO
from datetime import datetime
import json

class ModelImprover:
    def __init__(self, base_model_path, output_dir="models"):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/previous_versions", exist_ok=True)
        
    def combine_datasets(self):
        """Combine existing COCO dataset with new Roboflow datasets."""
        print("="*60)
        print(" Combining Datasets")
        print("="*60)
        
        combined_dir = "datasets/initial_dataset"
        os.makedirs(f"{combined_dir}/images/train", exist_ok=True)
        os.makedirs(f"{combined_dir}/images/val", exist_ok=True)
        os.makedirs(f"{combined_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{combined_dir}/labels/val", exist_ok=True)
        
        datasets = [
            {
                'name': 'COCO',
                'path': 'yolo_cellphone_dataset',
                'train_img': 'images/train',
                'val_img': 'images/val',
                'train_lbl': 'labels/train',
                'val_lbl': 'labels/val'
            },
            {
                'name': 'MobilePhone',
                'path': 'dataset_mobile_phone',
                'train_img': 'train/images',
                'val_img': 'valid/images',
                'train_lbl': 'train/labels',
                'val_lbl': 'valid/labels'
            },
            {
                'name': 'PhoneCall',
                'path': 'dataset_phone_call',
                'train_img': 'train/images',
                'val_img': 'valid/images',
                'train_lbl': 'train/labels',
                'val_lbl': 'valid/labels'
            },
            {
                'name': 'PhoneHand',
                'path': 'dataset_phone_hand',
                'train_img': 'train/images',
                'val_img': 'valid/images',
                'train_lbl': 'train/labels',
                'val_lbl': 'valid/labels'
            }
        ]
        
        total_train = 0
        total_val = 0
        
        for dataset in datasets:
            if not os.path.exists(dataset['path']):
                print(f"  {dataset['name']} not found, skipping...")
                continue
            
            print(f"\n Adding {dataset['name']} dataset...")
            
            # Copy training data
            train_img_dir = os.path.join(dataset['path'], dataset['train_img'])
            train_lbl_dir = os.path.join(dataset['path'], dataset['train_lbl'])
            
            if os.path.exists(train_img_dir):
                for img_file in Path(train_img_dir).glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        new_name = f"{dataset['name']}_{img_file.name}"
                        shutil.copy2(img_file, f"{combined_dir}/images/train/{new_name}")
                        
                        lbl_file = Path(train_lbl_dir) / f"{img_file.stem}.txt"
                        if lbl_file.exists():
                            shutil.copy2(lbl_file, f"{combined_dir}/labels/train/{Path(new_name).stem}.txt")
                        total_train += 1
            
            # Copy validation data
            val_img_dir = os.path.join(dataset['path'], dataset['val_img'])
            val_lbl_dir = os.path.join(dataset['path'], dataset['val_lbl'])
            
            if os.path.exists(val_img_dir):
                for img_file in Path(val_img_dir).glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        new_name = f"{dataset['name']}_{img_file.name}"
                        shutil.copy2(img_file, f"{combined_dir}/images/val/{new_name}")
                        
                        lbl_file = Path(val_lbl_dir) / f"{img_file.stem}.txt"
                        if lbl_file.exists():
                            shutil.copy2(lbl_file, f"{combined_dir}/labels/val/{Path(new_name).stem}.txt")
                        total_val += 1
            
            print(f"   Added {dataset['name']}")
        
        print("\n" + "="*60)
        print(f" Combined Dataset Summary:")
        print(f"   Training images:   {total_train}")
        print(f"   Validation images: {total_val}")
        print(f"   Total images:      {total_train + total_val}")
        print("="*60)
        
        # Create YAML config
        config = {
            'path': os.path.abspath(combined_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['cell phone']
        }
        
        yaml_path = f"{combined_dir}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n Created dataset config: {yaml_path}")
        return yaml_path, total_train, total_val
    
    def train_improved_model(self, data_yaml):
        """Train improved model from existing best model."""
        print("\n" + "="*60)
        print(" Training Improved Model")
        print("="*60)
        
        # Load existing model
        print(f"\n Loading model: {self.base_model_path}")
        model = YOLO(self.base_model_path)
        
        # Train
        print("\n Starting training (50 epochs)...")
        results = model.train(
            data=data_yaml,
            epochs=50,
            imgsz=640,
            batch=16,
            name='improved_model',
            project='runs/detect',
            patience=30,
            save=True,
            device=0,
            workers=8,
            optimizer='AdamW',
            lr0=0.0001,
            
            # Enhanced augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            
            single_cls=True,
            amp=True,
            plots=True,
            verbose=True
        )
        
        # Validate
        print("\n Validating improved model...")
        metrics = model.val()
        
        print("\n" + "="*60)
        print(" Model Performance:")
        print("="*60)
        print(f"  Precision:    {metrics.box.mp:.4f}")
        print(f"  Recall:       {metrics.box.mr:.4f}")
        print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print("="*60)
        
        # Save as current best
        best_path = 'runs/detect/improved_model/weights/best.pt'
        new_model_path = f"{self.output_dir}/current_best.pt"
        shutil.copy2(best_path, new_model_path)
        
        # Archive old version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f"{self.output_dir}/previous_versions/model_{timestamp}.pt"
        if os.path.exists(self.base_model_path):
            shutil.copy2(self.base_model_path, archive_path)
        
        # Save training history
        history = {
            'timestamp': timestamp,
            'base_model': self.base_model_path,
            'new_model': new_model_path,
            'metrics': {
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'map50': float(metrics.box.map50),
                'map': float(metrics.box.map)
            }
        }
        
        history_file = f"{self.output_dir}/training_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                all_history = json.load(f)
        else:
            all_history = []
        
        all_history.append(history)
        
        with open(history_file, 'w') as f:
            json.dump(all_history, f, indent=2)
        
        print(f"\n New model saved: {new_model_path}")
        print(f" Old model archived: {archive_path}")
        
        return new_model_path, metrics
    
    def run(self):
        """Run the complete improvement process."""
        print("="*60)
        print(" Model Improvement Pipeline")
        print("="*60)
        
        # Step 1: Combine datasets
        data_yaml, train_count, val_count = self.combine_datasets()
        
        # Step 2: Train improved model
        new_model, metrics = self.train_improved_model(data_yaml)
        
        print("\n" + "="*60)
        print(" Model Improvement Complete!")
        print("="*60)
        print(f"\n Improved model ready: {new_model}")
        print("\nNext step: Run webcam capture system")
        print("  python scripts/webcam_capture.py")
        print("="*60)
        
        return new_model

if __name__ == "__main__":
    # Run improvement
    improver = ModelImprover(
        base_model_path="runs/detect/cellphone_yolo11m/weights/best.pt"
    )
    improver.run()
import os
import shutil
from pathlib import Path
import yaml
import json
from datetime import datetime
from ultralytics import YOLO

class ModelRetrainer:
    def __init__(self, current_model_path, config):
        self.current_model_path = current_model_path
        self.config = config
        
        # Directories
        self.verified_dir = "datasets/captured_data/verified_positive"
        self.initial_dataset = "datasets/initial_dataset"
        self.training_batch_dir = "datasets/training_batches/current"
        
        self.models_dir = "models"
        self.log_dir = "logs/training_logs"
        
        os.makedirs(self.log_dir, exist_ok=True)
    
    def check_if_ready(self):
        """Check if we have enough verified data for retraining."""
        verified_images = list(Path(f"{self.verified_dir}/images").glob('*.jpg'))
        
        print(f"\n Verified samples available: {len(verified_images)}")
        
        if len(verified_images) < self.config['batch_size']:
            print(f" Need {self.config['batch_size'] - len(verified_images)} more samples")
            return False
        
        return True
    
    def prepare_training_data(self):
        """Combine initial dataset with new verified data."""
        print("\n" + "="*60)
        print(" Preparing Training Data")
        print("="*60)
        
        # Create training batch directory
        os.makedirs(f"{self.training_batch_dir}/images/train", exist_ok=True)
        os.makedirs(f"{self.training_batch_dir}/images/val", exist_ok=True)
        os.makedirs(f"{self.training_batch_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{self.training_batch_dir}/labels/val", exist_ok=True)
        
        # Copy initial dataset
        print("\n Copying initial dataset...")
        initial_train_img = f"{self.initial_dataset}/images/train"
        initial_val_img = f"{self.initial_dataset}/images/val"
        initial_train_lbl = f"{self.initial_dataset}/labels/train"
        initial_val_lbl = f"{self.initial_dataset}/labels/val"
        
        train_count = 0
        val_count = 0
        
        # Copy training data
        if os.path.exists(initial_train_img):
            for img in Path(initial_train_img).glob('*.*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img, f"{self.training_batch_dir}/images/train/{img.name}")
                    lbl = Path(initial_train_lbl) / f"{img.stem}.txt"
                    if lbl.exists():
                        shutil.copy2(lbl, f"{self.training_batch_dir}/labels/train/{img.stem}.txt")
                    train_count += 1
        
        # Copy validation data
        if os.path.exists(initial_val_img):
            for img in Path(initial_val_img).glob('*.*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img, f"{self.training_batch_dir}/images/val/{img.name}")
                    lbl = Path(initial_val_lbl) / f"{img.stem}.txt"
                    if lbl.exists():
                        shutil.copy2(lbl, f"{self.training_batch_dir}/labels/val/{img.stem}.txt")
                    val_count += 1
        
        print(f"   Copied {train_count} training images")
        print(f"   Copied {val_count} validation images")
        
        # Add verified data (split 80/20 train/val)
        print("\n Adding verified data...")
        verified_imgs = list(Path(f"{self.verified_dir}/images").glob('*.jpg'))
        
        split_idx = int(len(verified_imgs) * (1 - self.config['validation_split']))
        train_imgs = verified_imgs[:split_idx]
        val_imgs = verified_imgs[split_idx:]
        
        new_train = 0
        new_val = 0
        
        # Add to training
        for img in train_imgs:
            shutil.copy2(img, f"{self.training_batch_dir}/images/train/verified_{img.name}")
            lbl = Path(f"{self.verified_dir}/labels") / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, f"{self.training_batch_dir}/labels/train/verified_{img.stem}.txt")
            new_train += 1
        
        # Add to validation
        for img in val_imgs:
            shutil.copy2(img, f"{self.training_batch_dir}/images/val/verified_{img.name}")
            lbl = Path(f"{self.verified_dir}/labels") / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, f"{self.training_batch_dir}/labels/val/verified_{img.stem}.txt")
            new_val += 1
        
        print(f"   Added {new_train} new training images")
        print(f"   Added {new_val} new validation images")
        
        # Create YAML config
        config = {
            'path': os.path.abspath(self.training_batch_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['cell phone']
        }
        
        yaml_path = f"{self.training_batch_dir}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("\n" + "="*60)
        print(" Training Data Summary:")
        print("="*60)
        print(f"  Total training:   {train_count + new_train}")
        print(f"  Total validation: {val_count + new_val}")
        print(f"  New samples:      {new_train + new_val}")
        print("="*60)
        
        return yaml_path, new_train + new_val
    
    def retrain_model(self, data_yaml):
        """Retrain model with combined data."""
        print("\n" + "="*60)
        print(" Retraining Model")
        print("="*60)
        
        # Load current model
        print(f"\n Loading current model: {self.current_model_path}")
        model = YOLO(self.current_model_path)
        
        # Get baseline metrics
        print("\n Baseline performance:")
        baseline_metrics = model.val(data=data_yaml)
        print(f"  mAP@0.5: {baseline_metrics.box.map50:.4f}")
        print(f"  Recall:  {baseline_metrics.box.mr:.4f}")
        
        # Train
        print(f"\n Training for {self.config['epochs']} epochs...")
        results = model.train(
            data=data_yaml,
            epochs=self.config['epochs'],
            imgsz=640,
            batch=16,
            name='retrained_model',
            project='runs/detect',
            patience=15,
            save=True,
            device=0,
            workers=8,
            optimizer='AdamW',
            lr0=self.config['learning_rate'],
            
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            
            single_cls=True,
            amp=True,
            plots=True,
            verbose=True
        )
        
        # Validate new model
        print("\n Evaluating new model...")
        new_model = YOLO('runs/detect/retrained_model/weights/best.pt')
        new_metrics = new_model.val(data=data_yaml)
        
        print("\n" + "="*60)
        print(" Performance Comparison")
        print("="*60)
        print(f"{'Metric':<15} {'Baseline':<12} {'New Model':<12} {'Change'}")
        print("-"*60)
        
        metrics_to_compare = [
            ('Precision', baseline_metrics.box.mp, new_metrics.box.mp),
            ('Recall', baseline_metrics.box.mr, new_metrics.box.mr),
            ('mAP@0.5', baseline_metrics.box.map50, new_metrics.box.map50),
            ('mAP@0.5:0.95', baseline_metrics.box.map, new_metrics.box.map)
        ]
        
        improvement = False
        for name, baseline, new_val in metrics_to_compare:
            change = new_val - baseline
            symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"{name:<15} {baseline:<12.4f} {new_val:<12.4f} {symbol} {change:+.4f}")
            
            if name == 'mAP@0.5' and change >= self.config['min_map_improvement']:
                improvement = True
        
        print("="*60)
        
        return new_metrics, improvement
    
    def deploy_model(self, metrics):
        """Deploy new model if improved."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Archive current model
        archive_path = f"{self.models_dir}/previous_versions/model_{timestamp}.pt"
        shutil.copy2(self.current_model_path, archive_path)
        print(f"\n Archived old model: {archive_path}")
        
        # Deploy new model
        new_model_path = 'runs/detect/retrained_model/weights/best.pt'
        shutil.copy2(new_model_path, self.current_model_path)
        print(f" Deployed new model: {self.current_model_path}")
        
        # Log training
        log_entry = {
            'timestamp': timestamp,
            'metrics': {
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'map50': float(metrics.box.map50),
                'map': float(metrics.box.map)
            },
            'deployed': True
        }
        
        log_file = f"{self.log_dir}/training_log.json"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        log_data.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Clean up verified data (move to archive)
        archive_batch = f"datasets/training_batches/batch_{timestamp}"
        shutil.move(self.verified_dir, archive_batch)
        os.makedirs(f"{self.verified_dir}/images", exist_ok=True)
        os.makedirs(f"{self.verified_dir}/labels", exist_ok=True)
        
        print(f" Archived training batch: {archive_batch}")
    
    def run(self):
        """Run complete retraining pipeline."""
        print("="*60)
        print(" Automated Retraining Pipeline")
        print("="*60)
        
        # Check if ready
        if not self.check_if_ready():
            print("\n Not enough verified data for retraining")
            print("Continue capturing more samples")
            return False
        
        # Prepare data
        data_yaml, new_samples = self.prepare_training_data()
        
        # Retrain
        new_metrics, improved = self.retrain_model(data_yaml)
        
        # Deploy if improved
        if improved:
            print("\n Model improved! Deploying...")
            self.deploy_model(new_metrics)
            
            print("\n" + "="*60)
            print(" Retraining Complete & Deployed!")
            print("="*60)
            print("\nNext step: Continue learning loop")
            print("  python scripts/5_continuous_learning.py")
            print("="*60)
            
            return True
        else:
            print("\n  No significant improvement")
            print(f"   Required: +{self.config['min_map_improvement']:.3f} mAP@0.5")
            print("   Keeping current model")
            print("\n Tips:")
            print("   - Collect more diverse samples")
            print("   - Check data quality")
            print("   - Adjust training parameters")
            
            return False

if __name__ == "__main__":
    # Configuration
    config = {
        'batch_size': 50,
        'epochs': 20,
        'learning_rate': 0.0001,
        'validation_split': 0.2,
        'min_map_improvement': 0.01
    }
    
    # Run retraining
    retrainer = ModelRetrainer(
        current_model_path="models/current_best.pt",
        config=config
    )
    retrainer.run()
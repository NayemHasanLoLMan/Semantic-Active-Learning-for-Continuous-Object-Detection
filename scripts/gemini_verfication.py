import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from PIL import Image
import time

from scripts.utils.vector_db_manager import VectorDBManager


class GeminiVerifier:
    def __init__(self, api_key, config):
        self.config = config
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config['model'])

        self.vector_db = VectorDBManager()
        
        # Directories
        self.pending_dir = "datasets/captured_data/pending_verification"
        self.verified_positive_dir = "datasets/captured_data/verified_positive"
        self.verified_negative_dir = "datasets/captured_data/verified_negative"
        
        # Create directories
        os.makedirs(f"{self.verified_positive_dir}/images", exist_ok=True)
        os.makedirs(f"{self.verified_positive_dir}/labels", exist_ok=True)
        os.makedirs(f"{self.verified_negative_dir}/images", exist_ok=True)
        
        # Load detections
        self.detections_file = f"{self.pending_dir}/detections.json"
        self.detections = self.load_detections()
        
        # Verification log
        self.log_file = "logs/verification_logs/verification_log.json"
        os.makedirs("logs/verification_logs", exist_ok=True)
        self.verification_log = self.load_log()
        
    def load_detections(self):
        """Load pending detections."""
        if os.path.exists(self.detections_file):
            with open(self.detections_file, 'r') as f:
                return json.load(f)
        return []
    
    def load_log(self):
        """Load verification log."""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_log(self):
        """Save verification log."""
        with open(self.log_file, 'w') as f:
            json.dump(self.verification_log, f, indent=2)
    
    def verify_image(self, image_path):
        """Verify if image contains a cell phone using Gemini."""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Create prompt
            prompt = self.config['verification_prompt']
            
            # Generate response
            response = self.model.generate_content([prompt, img])
            
            # Parse response
            result_text = response.text.strip().lower()
            
            # Determine if it's a phone
            is_phone = 'true' in result_text
            
            return is_phone, result_text
        
        except Exception as e:
            print(f"  Error verifying {image_path}: {e}")
            return None, str(e)
    
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """Convert bbox to YOLO format."""
        x1, y1, x2, y2 = bbox
        
        # Calculate center and dimensions
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Normalize
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    def process_verified_image(self, detection, is_phone):
            """Move verified image to appropriate directory and create label."""
            image_path = f"{self.pending_dir}/images/{detection['image_name']}"
            
            if not os.path.exists(image_path):
                print(f"  Image not found: {image_path}")
                return
            
            # Get image dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            if is_phone:
                # Move to verified positive
                dest_img_path = f"{self.verified_positive_dir}/images/{detection['image_name']}"
                shutil.move(image_path, dest_img_path)
                
                # Create YOLO label file
                label_path = f"{self.verified_positive_dir}/labels/{Path(detection['image_name']).stem}.txt"
                
                # Get the confidence score for metadata
                confidence = detection['detections'][0]['confidence'] if detection['detections'] else 0.0

                with open(label_path, 'w') as f:
                    for det in detection['detections']:
                        bbox = det['bbox']
                        yolo_bbox = self.convert_bbox_to_yolo(bbox, img_width, img_height)
                        # Class 0 for cell phone
                        f.write(f"0 {' '.join(map(str, yolo_bbox))}\n")
                
                # --- NEW: Add to Vector Memory ---
                # This is the Critical Step for Q1 Standards
                try:
                    self.vector_db.add_to_memory(
                        image_path=dest_img_path,
                        label="cellphone",
                        confidence=confidence
                    )
                    print(f"   [Memory] Vector embedding stored.")
                except Exception as e:
                    print(f"   [Warning] Failed to add to Vector DB: {e}")
                # ---------------------------------

                print(f"   Verified POSITIVE: {detection['image_name']}")
            else:
                # Move to verified negative
                dest_img_path = f"{self.verified_negative_dir}/images/{detection['image_name']}"
                shutil.move(image_path, dest_img_path)
                print(f"   Verified NEGATIVE: {detection['image_name']}")
    
    def run(self):
        """Run verification on all pending images."""
        print("="*60)
        print(" Gemini Verification System")
        print("="*60)
        
        if not self.detections:
            print("\n  No pending detections to verify")
            return
        
        print(f"\n Found {len(self.detections)} detections to verify")
        print("="*60)
        
        verified_count = 0
        positive_count = 0
        negative_count = 0
        
        for i, detection in enumerate(self.detections):
            print(f"\n[{i+1}/{len(self.detections)}] Verifying: {detection['image_name']}")
            
            image_path = f"{self.pending_dir}/images/{detection['image_name']}"
            
            if not os.path.exists(image_path):
                print(f"    Image not found, skipping...")
                continue
            
            # Verify with Gemini
            is_phone, gemini_response = self.verify_image(image_path)
            
            if is_phone is None:
                print(f"    Verification failed, skipping...")
                continue
            
            # Process result
            self.process_verified_image(detection, is_phone)
            
            # Log verification
            log_entry = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'image_name': detection['image_name'],
                'is_phone': is_phone,
                'gemini_response': gemini_response,
                'confidence': detection['detections'][0]['confidence'] if detection['detections'] else 0
            }
            self.verification_log.append(log_entry)
            
            verified_count += 1
            if is_phone:
                positive_count += 1
            else:
                negative_count += 1
            
            # Rate limiting (Gemini free tier)
            time.sleep(1)
        
        # Save log
        self.save_log()
        
        # Clear processed detections
        if os.path.exists(self.detections_file):
            os.remove(self.detections_file)
        
        print("\n" + "="*60)
        print(" Verification Summary")
        print("="*60)
        print(f"  Total verified:     {verified_count}")
        print(f"  Positive (phones):  {positive_count}")
        print(f"  Negative (not):     {negative_count}")
        print(f"  Accuracy estimate:  {(positive_count/verified_count*100):.1f}%")
        print("="*60)
        print(f"\n Verified data saved:")
        print(f"   Positive: {self.verified_positive_dir}/")
        print(f"   Negative: {self.verified_negative_dir}/")
        print(f"   Log: {self.log_file}")
        print("="*60)
        
        # Check if ready for retraining
        verified_images = len(list(Path(f"{self.verified_positive_dir}/images").glob('*.jpg')))
        print(f"\n Training Progress: {verified_images} verified samples")
        
        if verified_images >= 50:
            print(" Ready for retraining! (≥50 samples)")
            print("\nNext step: Run retraining")
            print("  python scripts/retrain_model.py")
        else:
            print(f" Need {50 - verified_images} more samples for retraining")
            print("\nNext step: Continue capturing")
            print("  python scripts/webcam_capture.py")
        
        print("="*60)
        
        return verified_count, positive_count, negative_count

if __name__ == "__main__":
    # Get API key
    api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print(" Error: API key required")
        print("Get your free key: https://makersuite.google.com/app/apikey")
        exit(1)
    
    # Configuration
    config = {
        'model': 'gemini-1.5-flash',
        'verification_prompt': """Is there a cell phone (mobile phone, smartphone) visible in this image?
Analyze carefully and respond with only 'true' or 'false'.
- true: if you can see a cell phone/smartphone
- false: if there is no cell phone or you're not sure"""
    }
    
    # Run verification
    verifier = GeminiVerifier(api_key=api_key, config=config)
    verifier.run()

import cv2
import os
import json
import time
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np
from scripts.utils.vector_db_manager import VectorDBManager

class WebcamCapture:
    def __init__(self, model_path, config):
        self.model = YOLO(model_path)
        self.config = config
        
        # Create directories
        self.pending_dir = "datasets/captured_data/pending_verification"
        os.makedirs(f"{self.pending_dir}/images", exist_ok=True)
        
        # Detection metadata
        self.detections_file = f"{self.pending_dir}/detections.json"
        self.detections = self.load_detections()
        
        # Capture statistics
        self.capture_count = 0
        self.last_capture_time = 0
        
        # Vector DB Manager for deduplication
        self.vector_db = VectorDBManager() # Initialize DB
        self.use_vector_db = config.get('use_vector_deduplication', True)
        self.threshold = config.get('similarity_threshold', 0.15)

    def load_detections(self):
        """Load existing detections."""
        if os.path.exists(self.detections_file):
            with open(self.detections_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_detections(self):
        """Save detections to file."""
        with open(self.detections_file, 'w') as f:
            json.dump(self.detections, f, indent=2)
    
    def should_capture(self):
        """Check if we should capture based on interval and limits."""
        current_time = time.time()
        
        # Check time interval
        if current_time - self.last_capture_time < self.config['capture_interval']:
            return False
        
        # Check max captures
        if self.capture_count >= self.config['max_captures_per_session']:
            return False
        
        return True
    
    def capture_detection(self, frame, boxes, confidences):
        """Capture frame with detection information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_name = f"capture_{timestamp}.jpg"
        image_path = f"{self.pending_dir}/images/{image_name}"
        
        # Save image
        cv2.imwrite(image_path, frame)
        
        # Save detection metadata
        detection_info = {
            'image_name': image_name,
            'timestamp': timestamp,
            'detections': []
        }
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            detection_info['detections'].append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class': 'cell phone'
            })
        
        self.detections.append(detection_info)
        self.save_detections()
        
        self.capture_count += 1
        self.last_capture_time = time.time()
        
        print(f" Captured: {image_name} (Count: {self.capture_count})")
        
        return image_path
    
    def run(self):
            """Run webcam capture system with Semantic Deduplication."""
            print("="*60)
            print(" Webcam Capture System (Active Learning + Vector Memory)")
            print("="*60)
            print(f"\nConfiguration:")
            print(f"  Capture interval: {self.config['capture_interval']}s")
            print(f"  Max captures: {self.config['max_captures_per_session']}")
            print(f"  Min confidence: {self.config['min_confidence_for_capture']}")
            print(f"\nControls:")
            print("  Press 'q' to quit")
            print("  Press 's' to skip capture")
            print("  Press 'c' to force capture")
            print("="*60)
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print(" Error: Could not open webcam")
                return
            
            print("\n Webcam started. Detecting...")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run detection
                    results = self.model(frame, 
                                        conf=self.config['min_confidence_for_capture'],
                                        verbose=False)
                    
                    # Get detections
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    # Draw detections
                    annotated_frame = results[0].plot()
                    
                    # Display capture status
                    status_text = f"Captures: {self.capture_count}/{self.config['max_captures_per_session']}"
                    cv2.putText(annotated_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # --- MODIFIED: Auto capture with Semantic Check ---
                    if len(boxes) > 0 and self.should_capture():
                        
                        should_save = True
                        
                        # 1. Perform Semantic Deduplication
                        if self.use_vector_db:
                            # Crop the primary detection (highest confidence)
                            # Box format is [x1, y1, x2, y2]
                            x1, y1, x2, y2 = map(int, boxes[0]) 
                            
                            # Ensure coordinates are within frame bounds
                            h, w, _ = frame.shape
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                crop = frame[y1:y2, x1:x2]
                                # Convert BGR (OpenCV) to RGB (PIL) for the VectorDB
                                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                
                                is_unique, _ = self.vector_db.is_semantically_unique(pil_crop)
                                
                                if not is_unique:
                                    print(f"  [Skipped] Semantic duplicate detected (Similarity too high)")
                                    should_save = False
                                    # Visual feedback for skipped
                                    cv2.putText(annotated_frame, "Skipped: Duplicate", (10, 60), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # 2. Save if unique
                        if should_save:
                            self.capture_detection(frame, boxes, confidences)
                            # Visual feedback for saved
                            cv2.rectangle(annotated_frame, (0, 0), 
                                        (annotated_frame.shape[1], annotated_frame.shape[0]),
                                        (0, 255, 0), 10)
                    
                    # Show frame
                    cv2.imshow('Cell Phone Detection - Capture System', annotated_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("\n Stopping capture...")
                        break
                    elif key == ord('c') and len(boxes) > 0:
                        # Force capture bypasses vector check (User Override)
                        self.capture_detection(frame, boxes, confidences)
                    elif key == ord('s'):
                        # Skip next capture
                        self.last_capture_time = time.time()
                    
                    # Auto-stop at max captures
                    if self.capture_count >= self.config['max_captures_per_session']:
                        print(f"\n Reached maximum captures ({self.capture_count})")
                        break
            
            except KeyboardInterrupt:
                print("\n  Interrupted by user")
            
            finally:
                cap.release()
                cv2.destroyAllWindows()
                
                print("\n" + "="*60)
                print(" Capture Session Summary")
                print("="*60)
                print(f"  Total captures: {self.capture_count}")
                print(f"  Saved to: {self.pending_dir}/")
                print("="*60)

if __name__ == "__main__":
    # Configuration
    config = {
        'capture_interval': 2.0,
        'min_confidence_for_capture': 0.15,
        'max_captures_per_session': 50,
        'save_format': 'jpg'
    }
    
    # Run capture
    capturer = WebcamCapture(
        model_path="models/current_best.pt",
        config=config
    )
    capturer.run()
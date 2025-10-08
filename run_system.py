import os
import sys

def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print(" CONTINUOUS LEARNING CELL PHONE DETECTION SYSTEM")
    print("="*70)

def check_setup():
    """Check if system is properly set up."""
    issues = []
    
    if not os.path.exists("models/current_best.pt"):
        issues.append(" No model found at models/current_best.pt")
    
    if not os.path.exists("config/config.yaml"):
        issues.append(" No config found at config/config.yaml")
    
    if not os.path.exists("scripts"):
        issues.append(" Scripts folder not found")
    
    return issues

def main_menu():
    """Display main menu and handle user selection."""
    
    print_banner()
    
    # Check setup
    issues = check_setup()
    if issues:
        print("\n  Setup Issues Detected:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 Run: python create_project.py")
        print()
        return
    
    print("\n System ready!")
    print("\n" + "-"*70)
    print("SELECT OPERATION:")
    print("-"*70)
    
    print("\n Quick Actions:")
    print("  1. Run Full Continuous Learning System (Recommended)")
    print("  2. Run Single Learning Cycle")
    
    print("\n Data Collection:")
    print("  3. Capture Data from Webcam")
    print("  4. Verify Captured Data with Gemini")
    
    print("\n Model Training:")
    print("  5. Improve Initial Model (Add Roboflow Data)")
    print("  6. Retrain with Verified Data")
    
    print("\n Monitoring:")
    print("  7. View System Statistics")
    print("  8. Check Model Performance")
    
    print("\n  Utilities:")
    print("  9. Test Model on Image")
    print("  10. Test Model on Webcam")
    
    print("\n  0. Exit")
    print("-"*70)
    
    choice = input("\nEnter your choice (0-10): ").strip()
    
    return choice

def run_continuous_learning():
    """Run full continuous learning system."""
    print("\n Starting Continuous Learning System...")
    print("This will run indefinitely. Press Ctrl+C to stop.\n")
    
    cycles = input("Enter number of cycles (leave empty for infinite): ").strip()
    
    if cycles:
        os.system(f"python scripts/5_continuous_learning.py --cycles {cycles}")
    else:
        os.system("python scripts/5_continuous_learning.py")

def run_single_cycle():
    """Run single learning cycle."""
    print("\n Running Single Learning Cycle...")
    os.system("python scripts/5_continuous_learning.py --cycles 1")

def run_webcam_capture():
    """Run webcam capture."""
    print("\n Starting Webcam Capture...")
    print("Press 'q' to quit, 'c' to force capture, 's' to skip\n")
    os.system("python scripts/2_webcam_capture.py")

def run_gemini_verification():
    """Run Gemini verification."""
    print("\n Starting Gemini Verification...")
    os.system("python scripts/3_gemini_verification.py")

def run_improve_model():
    """Improve initial model with Roboflow data."""
    print("\n Improving Initial Model...")
    print("This will add Roboflow datasets and train for 50 epochs.\n")
    
    confirm = input("Continue? This may take 2-4 hours. (y/n): ").lower()
    if confirm == 'y':
        os.system("python scripts/1_improve_initial_model.py")
    else:
        print("Cancelled.")

def run_retrain():
    """Run retraining with verified data."""
    print("\n Starting Retraining...")
    os.system("python scripts/4_retrain_model.py")

def view_statistics():
    """View system statistics."""
    print("\n System Statistics")
    print("="*70)
    
    # Check captures
    pending_dir = "datasets/captured_data/pending_verification/images"
    if os.path.exists(pending_dir):
        pending = len([f for f in os.listdir(pending_dir) if f.endswith('.jpg')])
        print(f"  Pending verification: {pending}")
    
    # Check verified
    verified_pos_dir = "datasets/captured_data/verified_positive/images"
    if os.path.exists(verified_pos_dir):
        verified = len([f for f in os.listdir(verified_pos_dir) if f.endswith('.jpg')])
        print(f"  Verified positives:   {verified}")
    
    # Check model versions
    versions_dir = "models/previous_versions"
    if os.path.exists(versions_dir):
        versions = len([f for f in os.listdir(versions_dir) if f.endswith('.pt')])
        print(f"  Model versions:       {versions}")
    
    # Check training history
    history_file = "models/training_history.json"
    if os.path.exists(history_file):
        import json
        with open(history_file, 'r') as f:
            history = json.load(f)
        print(f"  Training sessions:    {len(history)}")
        
        if history:
            latest = history[-1]
            print(f"\n  Latest Model Performance:")
            print(f"    Precision:    {latest['metrics']['precision']:.4f}")
            print(f"    Recall:       {latest['metrics']['recall']:.4f}")
            print(f"    mAP@0.5:      {latest['metrics']['map50']:.4f}")
    
    print("="*70)

def check_performance():
    """Check model performance."""
    print("\n Model Performance Check")
    print("="*70)
    
    from ultralytics import YOLO
    
    model_path = "models/current_best.pt"
    if not os.path.exists(model_path):
        print(" Model not found")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check if validation data exists
    val_data = "datasets/initial_dataset/data.yaml"
    if os.path.exists(val_data):
        print("Running validation...")
        metrics = model.val(data=val_data)
        
        print(f"\n  Precision:    {metrics.box.mp:.4f}")
        print(f"  Recall:       {metrics.box.mr:.4f}")
        print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    else:
        print("  Validation dataset not found")
    
    print("="*70)

def test_on_image():
    """Test model on single image."""
    print("\n Test Model on Image")
    print("="*70)
    
    image_path = input("Enter image path: ").strip()
    
    if not os.path.exists(image_path):
        print("Image not found")
        return
    
    from ultralytics import YOLO
    
    model = YOLO("models/current_best.pt")
    results = model(image_path)
    
    print(f"\nDetections: {len(results[0].boxes)}")
    results[0].show()
    
    save = input("\nSave result? (y/n): ").lower()
    if save == 'y':
        results[0].save("test_result.jpg")
        print("Saved to test_result.jpg")

def test_on_webcam():
    """Test model on webcam."""
    print("\n Test Model on Webcam")
    print("="*70)
    print("Press 'q' to quit\n")
    
    from ultralytics import YOLO
    import cv2
    
    model = YOLO("models/current_best.pt")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.25, verbose=False)
        annotated = results[0].plot()
        
        cv2.imshow('Model Test', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main menu loop."""
    while True:
        choice = main_menu()
        
        if choice == '0':
            print("\n Goodbye!")
            break
        elif choice == '1':
            run_continuous_learning()
        elif choice == '2':
            run_single_cycle()
        elif choice == '3':
            run_webcam_capture()
        elif choice == '4':
            run_gemini_verification()
        elif choice == '5':
            run_improve_model()
        elif choice == '6':
            run_retrain()
        elif choice == '7':
            view_statistics()
        elif choice == '8':
            check_performance()
        elif choice == '9':
            test_on_image()
        elif choice == '10':
            test_on_webcam()
        else:
            print("\n Invalid choice!")
        
        if choice != '0':
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
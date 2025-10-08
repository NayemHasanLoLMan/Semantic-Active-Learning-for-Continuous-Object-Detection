import os
import sys
import time
import yaml
from datetime import datetime

class ContinuousLearningSystem:
    def __init__(self, config_path="config/config.yaml"):
        self.config = self.load_config(config_path)
        self.running = True
        self.cycle_count = 0
        
        # Statistics
        self.stats = {
            'total_captures': 0,
            'total_verified': 0,
            'total_retrains': 0,
            'successful_deployments': 0
        }
        
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            # Create default config
            default_config = {
                'model': {
                    'current_model_path': 'models/current_best.pt',
                    'confidence_threshold': 0.25,
                    'iou_threshold': 0.45
                },
                'data_collection': {
                    'capture_interval': 2.0,
                    'min_confidence_for_capture': 0.15,
                    'max_captures_per_session': 50,
                    'save_format': 'jpg'
                },
                'gemini': {
                    'api_key': 'YOUR_API_KEY',
                    'model': 'gemini-1.5-flash',
                    'verification_prompt': 'Is there a cell phone visible? Answer only true or false.',
                    'batch_size': 10
                },
                'retraining': {
                    'trigger_mode': 'batch',
                    'batch_size': 50,
                    'epochs': 20,
                    'learning_rate': 0.0001,
                    'validation_split': 0.2,
                    'min_map_improvement': 0.01
                },
                'performance': {
                    'min_map_improvement': 0.01,
                    'track_metrics': True
                }
            }
            
            os.makedirs('config', exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            print(f"  Created default config: {config_path}")
            print("  Please update your Gemini API key in the config file!")
            
            return default_config
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def capture_phase(self):
        """Run webcam capture phase."""
        print("\n" + "="*60)
        print(f" PHASE 1: Capture (Cycle {self.cycle_count})")
        print("="*60)
        
        # Import the class directly
        import sys
        sys.path.insert(0, 'scripts')
        
        # Import webcam capture module
        import importlib.util
        spec = importlib.util.spec_from_file_location("webcam_capture", "scripts/2_webcam_capture.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        capturer = module.WebcamCapture(
            model_path=self.config['model']['current_model_path'],
            config=self.config['data_collection']
        )
        
        capturer.run()
        self.stats['total_captures'] += capturer.capture_count
        
        return capturer.capture_count
    
    def verification_phase(self):
        """Run Gemini verification phase."""
        print("\n" + "="*60)
        print(f" PHASE 2: Verification (Cycle {self.cycle_count})")
        print("="*60)
        
        if self.config['gemini']['api_key'] == 'YOUR_API_KEY':
            print(" Gemini API key not configured!")
            print("Please update config/config.yaml with your API key")
            return 0, 0, 0
        
        sys.path.insert(0, 'scripts')
        from gemini_verification import GeminiVerifier
        
        verifier = GeminiVerifier(
            api_key=self.config['gemini']['api_key'],
            config=self.config['gemini']
        )
        
        verified, positive, negative = verifier.run()
        self.stats['total_verified'] += verified
        
        return verified, positive, negative
    
    def retraining_phase(self):
        """Run retraining phase if ready."""
        print("\n" + "="*60)
        print(f" PHASE 3: Retraining (Cycle {self.cycle_count})")
        print("="*60)
        
        sys.path.insert(0, 'scripts')
        from retrain_model import ModelRetrainer
        
        retrainer = ModelRetrainer(
            current_model_path=self.config['model']['current_model_path'],
            config=self.config['retraining']
        )
        
        success = retrainer.run()
        
        if success:
            self.stats['total_retrains'] += 1
            self.stats['successful_deployments'] += 1
        
        return success
    
    def run_cycle(self):
        """Run one complete learning cycle."""
        self.cycle_count += 1
        
        print("\n" + "="*70)
        print(f" CONTINUOUS LEARNING CYCLE #{self.cycle_count}")
        print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        try:
            # Phase 1: Capture
            captures = self.capture_phase()
            
            if captures == 0:
                print("\n  No captures made, skipping this cycle")
                return False
            
            # Phase 2: Verification
            verified, positive, negative = self.verification_phase()
            
            if verified == 0:
                print("\n  No verifications made, skipping this cycle")
                return False
            
            # Phase 3: Retraining (if ready)
            retrained = self.retraining_phase()
            
            # Cycle summary
            print("\n" + "="*70)
            print(f" CYCLE #{self.cycle_count} SUMMARY")
            print("="*70)
            print(f"  Captures:     {captures}")
            print(f"  Verified:     {verified}")
            print(f"  Positive:     {positive}")
            print(f"  Negative:     {negative}")
            print(f"  Retrained:    {' Yes' if retrained else ' No'}")
            print("="*70)
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n  Cycle interrupted by user")
            self.running = False
            return False
        except Exception as e:
            print(f"\n Error in cycle: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def display_stats(self):
        """Display overall statistics."""
        print("\n" + "="*70)
        print(" OVERALL STATISTICS")
        print("="*70)
        print(f"  Total cycles:         {self.cycle_count}")
        print(f"  Total captures:       {self.stats['total_captures']}")
        print(f"  Total verified:       {self.stats['total_verified']}")
        print(f"  Total retrains:       {self.stats['total_retrains']}")
        print(f"  Successful deploys:   {self.stats['successful_deployments']}")
        print("="*70)
    
    def run(self, max_cycles=None):
        """Run continuous learning system."""
        print("="*70)
        print(" CONTINUOUS LEARNING SYSTEM")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Model: {self.config['model']['current_model_path']}")
        print(f"  Captures per session: {self.config['data_collection']['max_captures_per_session']}")
        print(f"  Retrain threshold: {self.config['retraining']['batch_size']} samples")
        print(f"  Mode: {'Batch' if self.config['retraining']['trigger_mode'] == 'batch' else 'Continuous'}")
        
        if max_cycles:
            print(f"\nRunning {max_cycles} cycles...")
        else:
            print(f"\nRunning indefinitely (Ctrl+C to stop)...")
        
        print("="*70)
        
        try:
            while self.running:
                # Run one cycle
                success = self.run_cycle()
                
                # Check if we should continue
                if max_cycles and self.cycle_count >= max_cycles:
                    print(f"\n Completed {max_cycles} cycles")
                    break
                
                if not self.running:
                    break
                
                # Wait before next cycle
                if success:
                    print(f"\n  Waiting 10 seconds before next cycle...")
                    time.sleep(10)
                else:
                    print(f"\n  Cycle failed, waiting 30 seconds...")
                    time.sleep(30)
        
        except KeyboardInterrupt:
            print("\n\n  System stopped by user")
        
        finally:
            # Display final statistics
            self.display_stats()
            
            print("\n" + "="*70)
            print(" Continuous Learning System Stopped")
            print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Learning System')
    parser.add_argument('--cycles', type=int, default=None,
                       help='Number of cycles to run (default: infinite)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Run system
    system = ContinuousLearningSystem(config_path=args.config)
    system.run(max_cycles=args.cycles)
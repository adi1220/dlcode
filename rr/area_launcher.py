#!/usr/bin/env python3
"""
Response Rewriter Arena Launcher
Manages all components of the arena system
"""

import os
import sys
import time
import subprocess
import argparse
import signal
import requests
from pathlib import Path
from datetime import datetime
import json

from config import Config


class ArenaLauncher:
    """Manages the arena system components."""
    
    def __init__(self):
        self.processes = {}
        self.start_time = datetime.now()
        
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'gradio', 'flask', 'flask-cors', 'requests', 
            'pandas', 'pydantic', 'python-dotenv'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"❌ Missing packages: {', '.join(missing)}")
            print(f"Run: pip install {' '.join(missing)}")
            return False
        
        print("✅ All dependencies installed")
        return True
    
    def check_configuration(self):
        """Validate configuration."""
        print("\n🔧 Checking configuration...")
        
        if not Config.validate():
            return False
        
        # Check if agent files exist
        agent_path = Path("core/agents/response_rewriter/agent.py")
        if not agent_path.exists():
            print(f"❌ Agent file not found: {agent_path}")
            print("Make sure the agent files are in the correct location")
            return False
        
        print("✅ Configuration valid")
        return True
    
    def start_backend(self):
        """Start the Flask backend."""
        print("\n🚀 Starting Flask backend...")
        
        try:
            self.processes['backend'] = subprocess.Popen(
                [sys.executable, "backend.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for backend to start
            for i in range(10):
                try:
                    response = requests.get(Config.FLASK_HEALTH_URL, timeout=1)
                    if response.status_code == 200:
                        print(f"✅ Backend started on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
                        return True
                except:
                    time.sleep(1)
            
            print("❌ Backend failed to start")
            return False
            
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the Gradio frontend."""
        print("\n🎨 Starting Gradio frontend...")
        
        try:
            self.processes['frontend'] = subprocess.Popen(
                [sys.executable, "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print(f"✅ Frontend starting on {Config.GRADIO_HOST}:{Config.GRADIO_PORT}")
            time.sleep(3)  # Give Gradio time to start
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting frontend: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes."""
        print("\n📊 Monitoring processes...")
        print("Press Ctrl+C to stop all services\n")
        
        try:
            while True:
                # Check if processes are still running
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        print(f"⚠️  {name} has stopped unexpectedly!")
                        return False
                
                # Show status every 30 seconds
                if int(time.time()) % 30 == 0:
                    uptime = datetime.now() - self.start_time
                    print(f"✅ System running for {uptime}")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            return True
    
    def stop_all(self):
        """Stop all processes."""
        for name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
        
        print("✅ All services stopped")
    
    def run_web_mode(self):
        """Run the full web interface mode."""
        print("""
╔═══════════════════════════════════════════════════════════╗
║          Response Rewriter Arena - Web Mode               ║
╚═══════════════════════════════════════════════════════════╝
        """)
        
        # Check everything
        if not self.check_dependencies():
            return
        
        if not self.check_configuration():
            return
        
        # Start services
        if not self.start_backend():
            self.stop_all()
            return
        
        if not self.start_frontend():
            self.stop_all()
            return
        
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║                  Arena is Ready! 🎉                       ║
║                                                           ║
║  Frontend: http://localhost:{Config.GRADIO_PORT:<38} ║
║  Backend:  http://localhost:{Config.FLASK_PORT:<38} ║
║                                                           ║
║  Press Ctrl+C to stop all services                       ║
╚═══════════════════════════════════════════════════════════╝
        """)
        
        # Monitor
        self.monitor_processes()
        self.stop_all()
    
    def run_test_mode(self):
        """Run test mode to verify everything works."""
        print("\n🧪 Running system test...")
        
        # Start backend only
        if not self.start_backend():
            return
        
        # Test endpoints
        tests = [
            ("Health Check", Config.FLASK_HEALTH_URL, "GET", None),
            ("Strategies", f"http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/strategies", "GET", None),
            ("Test Rewrite", Config.FLASK_URL, "POST", {
                "user_text": "Hello",
                "bix_response": "Hi there",
                "prompt_type": "default"
            })
        ]
        
        passed = 0
        for test_name, url, method, data in tests:
            try:
                if method == "GET":
                    response = requests.get(url, timeout=5)
                else:
                    response = requests.post(url, json=data, timeout=10)
                
                if response.status_code == 200:
                    print(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_name}: FAILED (Status: {response.status_code})")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
        
        print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
        
        self.stop_all()
    
    def export_data(self):
        """Export all data from the system."""
        print("\n📤 Exporting data...")
        
        from ratings_db import RatingsDatabase
        
        db = RatingsDatabase(Config.RATINGS_DB_PATH, Config.ARENA_DB_PATH)
        
        export_dir = Path("exports") / datetime.now().strftime('%Y%m%d_%H%M%S')
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export database data
        files = db.export_data(str(export_dir))
        
        # Export configuration
        config_file = export_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(Config.to_dict(), f, indent=2)
        
        # Get summary statistics
        summary = db.get_summary_statistics()
        summary_file = export_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Data exported to: {export_dir}")
        print(f"   - Ratings: {files['ratings']}")
        print(f"   - Battles: {files['battles']}")
        print(f"   - ELO Ratings: {files['elo']}")
        print(f"   - Configuration: {config_file}")
        print(f"   - Summary: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Response Rewriter Arena Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_arena.py              # Run full web interface
  python run_arena.py --test       # Run system tests
  python run_arena.py --export     # Export all data
  python run_arena.py --config     # Show configuration
        """
    )
    
    parser.add_argument('--test', action='store_true', help='Run system tests')
    parser.add_argument('--export', action='store_true', help='Export all data')
    parser.add_argument('--config', action='store_true', help='Show configuration')
    parser.add_argument('--backend-only', action='store_true', help='Run backend only')
    parser.add_argument('--frontend-only', action='store_true', help='Run frontend only')
    
    args = parser.parse_args()
    
    launcher = ArenaLauncher()
    
    try:
        if args.test:
            launcher.run_test_mode()
        elif args.export:
            launcher.export_data()
        elif args.config:
            Config.print_config()
        elif args.backend_only:
            print("Starting backend only...")
            subprocess.run([sys.executable, "backend.py"])
        elif args.frontend_only:
            print("Starting frontend only...")
            subprocess.run([sys.executable, "app.py"])
        else:
            launcher.run_web_mode()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        launcher.stop_all()
        sys.exit(1)


if __name__ == "__main__":
    main()

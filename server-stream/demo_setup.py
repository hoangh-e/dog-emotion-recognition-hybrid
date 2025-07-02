#!/usr/bin/env python3
"""
Demo setup script for Dog Emotion Recognition Web Interface
Suitable for both local and Google Colab environments
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    requirements = [
        "flask==2.3.3",
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "werkzeug>=2.3.0"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            continue
    
    print("✅ Requirements installed!")

def download_demo_models():
    """Download demo models for testing"""
    print("📥 Downloading demo models...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    demo_models = {
        "yolo_head_demo.pt": "1gK51jAz1gzYad7-UcDMmuH7bq849DOjz",
        "pure34_demo.pth": "11Oy8lqKF7MeMWV89SR-kN6sNLwNi-jjQ",
        "resnet50_demo.pth": "1s5KprrhHWkbhjRWCb3OK48I-OriDLR_S"
    }
    
    for filename, file_id in demo_models.items():
        model_path = models_dir / filename
        if not model_path.exists():
            try:
                print(f"📥 Downloading {filename}...")
                download_from_gdrive(file_id, str(model_path))
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                continue
        else:
            print(f"✅ {filename} already exists")

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive"""
    try:
        # Try gdown first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "gdown"
        ])
        subprocess.check_call([
            "gdown", f"https://drive.google.com/uc?id={file_id}", 
            "-O", output_path
        ])
    except:
        # Fallback to direct download
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        urllib.request.urlretrieve(url, output_path)

def create_demo_data():
    """Create demo data structure"""
    print("📁 Creating demo data structure...")
    
    directories = [
        "models",
        "uploads", 
        "results",
        "templates",
        "static/css",
        "static/js",
        "static/images"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    print("✅ Demo data structure created!")

def setup_demo_models_in_db():
    """Setup demo models in the model database"""
    print("🗄️ Setting up demo models in database...")
    
    try:
        from model_manager import ModelManager
        
        model_manager = ModelManager("models")
        
        # Add demo models if they exist
        demo_models = [
            {
                "name": "YOLO_Head_Demo",
                "type": "Head-object-detection", 
                "filename": "yolo_head_demo.pt"
            },
            {
                "name": "Pure34_Demo",
                "type": "Classification",
                "filename": "pure34_demo.pth"
            },
            {
                "name": "ResNet50_Demo", 
                "type": "Classification",
                "filename": "resnet50_demo.pth"
            }
        ]
        
        for model_info in demo_models:
            model_path = Path("models") / model_info["filename"]
            if model_path.exists():
                if not model_manager.model_exists(model_info["name"]):
                    model_manager.add_model(
                        model_info["name"],
                        model_info["type"],
                        str(model_path),
                        model_info["filename"]
                    )
                    print(f"✅ Added {model_info['name']} to database")
                else:
                    print(f"✅ {model_info['name']} already in database")
        
        print("✅ Demo models setup complete!")
        
    except ImportError:
        print("⚠️ ModelManager not available, skipping database setup")
    except Exception as e:
        print(f"❌ Failed to setup demo models: {e}")

def check_environment():
    """Check if running in Colab or local environment"""
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "local"

def setup_colab_environment():
    """Special setup for Google Colab"""
    print("🔧 Setting up Google Colab environment...")
    
    # Install ngrok for public URL
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
        print("✅ ngrok installed")
    except:
        print("❌ Failed to install ngrok")
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '0'
    
    print("✅ Colab environment setup complete!")

def start_app_colab():
    """Start the Flask app in Colab with ngrok tunnel"""
    print("🚀 Starting Flask app in Colab...")
    
    try:
        from pyngrok import ngrok
        import threading
        import time
        
        # Start Flask app in background thread
        def run_flask():
            import app
            app.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        
        # Wait for Flask to start
        time.sleep(5)
        
        # Create ngrok tunnel
        public_url = ngrok.connect(5000)
        
        print("🌐" + "="*50)
        print(f"🎉 Flask app is running!")
        print(f"🔗 Public URL: {public_url}")
        print(f"📱 Access your app at: {public_url}")
        print("🌐" + "="*50)
        
        return public_url
        
    except Exception as e:
        print(f"❌ Failed to start app in Colab: {e}")
        return None

def start_app_local():
    """Start the Flask app locally"""
    print("🚀 Starting Flask app locally...")
    print("🌐 Access your app at: http://localhost:5000")
    print("🔧 Press Ctrl+C to stop the server")
    
    try:
        import app
        app.app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")
    except Exception as e:
        print(f"❌ Failed to start local server: {e}")

def main():
    """Main setup function"""
    print("🐕 Dog Emotion Recognition - Demo Setup")
    print("="*50)
    
    # Check environment
    env = check_environment()
    print(f"🔍 Detected environment: {env.upper()}")
    
    # Install requirements
    install_requirements()
    
    # Create data structure
    create_demo_data()
    
    # Download demo models
    download_demo_models()
    
    # Setup demo models in database
    setup_demo_models_in_db()
    
    # Environment specific setup
    if env == "colab":
        setup_colab_environment()
        print("\n🎯 Setup complete! Use start_app_colab() to run the app.")
        return "colab"
    else:
        print("\n🎯 Setup complete! Use start_app_local() to run the app.")
        return "local"

def quick_start():
    """Quick start function"""
    env = main()
    
    # Auto-start based on environment
    if env == "colab":
        return start_app_colab()
    else:
        start_app_local()

if __name__ == "__main__":
    # Check if running as script or imported
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-start":
        quick_start()
    else:
        main()
        
        # Ask user if they want to start the app
        choice = input("\n🚀 Start the app now? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            env = check_environment()
            if env == "colab":
                start_app_colab()
            else:
                start_app_local() 
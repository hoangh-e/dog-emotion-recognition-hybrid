"""
Colab Setup Script for Dog Emotion Recognition Server-Stream

This script provides proper setup for running the Flask application in Google Colab
with ngrok tunneling and proper threading handling.
"""

import os
import sys
import threading
import time
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages for Colab environment"""
    print("📦 Installing requirements...")
    
    # Install pyngrok for tunneling
    try:
        import pyngrok
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
    
    # Install other requirements
    requirements = [
        "flask",
        "werkzeug",
        "pillow",
        "opencv-python",
        "torch",
        "torchvision",
        "ultralytics",
        "numpy",
        "pandas"
    ]
    
    for req in requirements:
        try:
            __import__(req.replace("-", "_"))
        except ImportError:
            print(f"Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
    
    print("✅ Requirements installed successfully!")

def setup_ngrok():
    """Setup ngrok for public URL access"""
    try:
        from pyngrok import ngrok, conf
        
        # Set ngrok path for Colab
        conf.get_default().ngrok_path = "/usr/local/bin/ngrok"
        
        # Kill existing tunnels
        ngrok.kill()
        
        return ngrok
    except Exception as e:
        print(f"❌ Error setting up ngrok: {e}")
        return None

def run_flask_app():
    """Run Flask app with proper error handling"""
    try:
        # Add current directory to Python path
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import Flask app
        from app import app
        
        # Configure for Colab
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
        
        print("🚀 Starting Flask application...")
        
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
        
    except Exception as e:
        print(f"❌ Error running Flask app: {e}")
        import traceback
        traceback.print_exc()

def start_server_with_ngrok():
    """Start server with ngrok tunnel in Colab"""
    print("🌐 Setting up Dog Emotion Recognition Server for Colab...")
    
    # Install requirements
    install_requirements()
    
    # Setup ngrok
    ngrok = setup_ngrok()
    if ngrok is None:
        print("❌ Failed to setup ngrok. Please install ngrok manually.")
        return None
    
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Wait for Flask to start
    print("⏳ Waiting for Flask to start...")
    time.sleep(3)
    
    try:
        # Create ngrok tunnel
        public_url = ngrok.connect(5000)
        print(f"✅ Server is running!")
        print(f"🌍 Public URL: {public_url}")
        print(f"📱 Local URL: http://localhost:5000")
        print("\n" + "="*50)
        print("🎯 Access your Dog Emotion Recognition Server at:")
        print(f"   {public_url}")
        print("="*50)
        
        return public_url
        
    except Exception as e:
        print(f"❌ Error creating ngrok tunnel: {e}")
        return None

def colab_demo():
    """Complete demo setup for Colab"""
    print("🐕 Dog Emotion Recognition - Colab Setup")
    print("="*50)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("⚠️  Not running in Google Colab, but continuing...")
    
    # Change to server-stream directory if it exists
    if os.path.exists('server-stream'):
        os.chdir('server-stream')
        print("📁 Changed to server-stream directory")
    elif os.path.exists('/content/dog-emotion-recognition-hybrid/server-stream'):
        os.chdir('/content/dog-emotion-recognition-hybrid/server-stream')
        print("📁 Changed to server-stream directory")
    
    # Start server
    public_url = start_server_with_ngrok()
    
    if public_url:
        print("\n🎉 Setup completed successfully!")
        print(f"🔗 Your server is available at: {public_url}")
        print("\n📋 Next steps:")
        print("1. Click the public URL above to access the web interface")
        print("2. Upload your models in the 'Add Model' section")
        print("3. Use 'Single Prediction' or 'Batch Prediction' features")
        print("\n⚠️  Keep this cell running to maintain the server!")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(60)
                print(f"🔄 Server still running at: {public_url}")
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
            import pyngrok
            pyngrok.ngrok.kill()
    else:
        print("❌ Failed to start server. Please check the error messages above.")

if __name__ == "__main__":
    colab_demo() 
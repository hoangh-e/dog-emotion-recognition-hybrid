# Google Colab Setup Instructions

## Quick Start - Copy and Paste This Code

### Step 1: Clone Repository and Setup
```python
# Clone the repository if not already done
!git clone https://github.com/your-repo/dog-emotion-recognition-hybrid.git
%cd dog-emotion-recognition-hybrid/server-stream

# Install requirements
!pip install flask werkzeug pillow opencv-python torch torchvision ultralytics numpy pandas pyngrok

# Run the Colab setup
exec(open('colab_setup.py').read())
```

### Step 2: Alternative Manual Setup (if Step 1 fails)
```python
import os
import sys
import threading
import time
from pyngrok import ngrok

# Change to server-stream directory
%cd /content/dog-emotion-recognition-hybrid/server-stream

# Add to Python path
sys.path.insert(0, os.getcwd())

# Import and configure Flask app
from app import app
app.config['DEBUG'] = False
app.config['TESTING'] = False

# Function to run Flask
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

# Start Flask in background thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Wait for Flask to start
time.sleep(3)

# Create ngrok tunnel
public_url = ngrok.connect(5000)
print(f"🌍 Public URL: {public_url}")
print(f"📱 Local URL: http://localhost:5000")

# Keep the tunnel alive
try:
    while True:
        time.sleep(60)
        print(f"🔄 Server still running at: {public_url}")
except KeyboardInterrupt:
    print("🛑 Server stopped")
    ngrok.kill()
```

## Troubleshooting Common Issues

### 1. Threading Error
If you get a threading error, try this safer approach:
```python
# Kill any existing processes
!pkill -f flask
!pkill -f ngrok

# Wait a moment
import time
time.sleep(2)

# Then run the setup again
exec(open('colab_setup.py').read())
```

### 2. Import Errors
If you get import errors for the dog_emotion packages:
```python
# Add the parent directory to Python path
import sys
import os
sys.path.insert(0, '/content/dog-emotion-recognition-hybrid')

# Verify the packages are accessible
try:
    import dog_emotion_classification
    import dog_emotion_ml
    print("✅ Packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and the packages exist")
```

### 3. Model Loading Issues
If models fail to load:
```python
# Check if model files exist
import os
model_dir = '/content/dog-emotion-recognition-hybrid/server-stream/models'
if os.path.exists(model_dir):
    print("📁 Models directory exists")
    print("Models found:", os.listdir(model_dir))
else:
    print("📁 Creating models directory")
    os.makedirs(model_dir, exist_ok=True)
```

### 4. ngrok Authentication (if required)
If ngrok requires authentication:
```python
from pyngrok import ngrok

# Set your ngrok auth token (get from https://ngrok.com/)
ngrok.set_auth_token("your_token_here")

# Then run the setup
exec(open('colab_setup.py').read())
```

## Step-by-Step Manual Process

### 1. Environment Setup
```python
# Install all requirements
!pip install flask werkzeug pillow opencv-python torch torchvision ultralytics numpy pandas pyngrok

# Navigate to the server directory
%cd /content/dog-emotion-recognition-hybrid/server-stream
```

### 2. Import Required Modules
```python
import os
import sys
import threading
import time
from pyngrok import ngrok

# Add current directory to Python path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '/content/dog-emotion-recognition-hybrid')
```

### 3. Configure Flask App
```python
# Import Flask app
from app import app

# Configure for Colab environment
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)
```

### 4. Start Flask Server
```python
def run_flask_server():
    """Run Flask server in a thread-safe way"""
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"Flask error: {e}")

# Start Flask in a daemon thread
flask_thread = threading.Thread(target=run_flask_server, daemon=True)
flask_thread.start()

print("⏳ Starting Flask server...")
time.sleep(3)
print("✅ Flask server started")
```

### 5. Create Public URL with ngrok
```python
# Kill any existing ngrok processes
ngrok.kill()

# Create tunnel
try:
    public_url = ngrok.connect(5000)
    print(f"🌍 Public URL: {public_url}")
    print(f"📱 Local URL: http://localhost:5000")
    print("\n" + "="*60)
    print("🎯 Your Dog Emotion Recognition Server is ready!")
    print(f"   Access it at: {public_url}")
    print("="*60)
except Exception as e:
    print(f"ngrok error: {e}")
```

### 6. Keep Server Running
```python
# Keep the server alive
print("⚠️  Keep this cell running to maintain the server!")
print("   Press Ctrl+C to stop the server")

try:
    while True:
        time.sleep(60)
        print(f"🔄 Server still running at: {public_url}")
except KeyboardInterrupt:
    print("\n🛑 Stopping server...")
    ngrok.kill()
    print("✅ Server stopped")
```

## Usage Instructions

1. **Upload Models**: Use the "Add Model" page to upload your `.pt` or `.pth` model files
2. **Single Prediction**: Upload a single image for emotion prediction
3. **Batch Prediction**: Upload multiple images via folder, ZIP file, or Roboflow integration

## Notes for Colab Users

- The server will automatically stop when you close the notebook or restart the runtime
- ngrok provides a temporary public URL that changes each time you restart
- Keep the cell running to maintain the server connection
- Upload models through the web interface rather than manually copying files

## Common File Paths in Colab

- Repository: `/content/dog-emotion-recognition-hybrid/`
- Server: `/content/dog-emotion-recognition-hybrid/server-stream/`
- Models: `/content/dog-emotion-recognition-hybrid/server-stream/models/`
- Uploads: `/content/dog-emotion-recognition-hybrid/server-stream/uploads/` 
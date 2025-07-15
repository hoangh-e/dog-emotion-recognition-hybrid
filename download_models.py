import os
import gdown
import zipfile

def download_file(url, output):
    """Download a file from Google Drive"""
    gdown.download(url, output, quiet=False)

def main():
    # Create models directory if it doesn't exist
    os.makedirs('trained', exist_ok=True)
    
    print("Downloading model files...")
    
    # Download and extract the main models zip
    models_zip = 'trained.zip'
    if not os.path.exists(models_zip) and not os.path.exists('trained/resnet'):
        print("Downloading trained models...")
        download_file('https://drive.google.com/uc?id=1rq1rXfjCmxVljg-kHvrzbILqKDy-HyVf', models_zip)
        
        print("Extracting models...")
        with zipfile.ZipFile(models_zip, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Extraction complete!")
    
    # Download individual model files
    models_to_download = [
        ('1Id2PaMxcU1YIoCH-ZxxD6qemX23t16sp', 'efficient_netb2.pt'),
        ('1uKw2fQ-Atb9zzFT4CRo4-F2O1N5504_m', 'yolo11n_dog_emotion_4cls_50epoch.pt'),
        ('1h3Wg_mzEhx7jip7OeXcfh2fZkvYfuvqf', 'vit_fold_1_best.pth')
    ]
    
    for file_id, filename in models_to_download:
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            download_file(f'https://drive.google.com/uc?id={file_id}', filename)
    
    print("\nAll model files have been downloaded successfully!")
    print("You can now run the notebook with the models in place.")

if __name__ == "__main__":
    main()

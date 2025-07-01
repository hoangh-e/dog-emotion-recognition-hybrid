# ==========================================
# UPDATED CELL 7: Load Emotion Models (Multi-Model Support)
# ==========================================

# Add the package to path
sys.path.append('/content/dog-emotion-recognition-hybrid')

# Import ALL model loading functions
from dog_emotion_classification import (
    load_pure34_model, predict_emotion_pure34,
    load_pure50_model, predict_emotion_pure50,
    load_resnet_emotion_model, predict_emotion_resnet
)

# Storage for loaded models
loaded_models = {}
model_transforms = {}
model_predict_functions = {}

print("🔄 Loading emotion classification models...")
print("=" * 50)

for model_name, config in ENABLED_MODELS.items():
    try:
        print(f"Loading {model_name} ({config['type']})...")
        
        if config['type'] == 'pure34':
            # Load Pure34 model
            model, transform = load_pure34_model(
                model_path=config['path'],
                num_classes=len(config['classes']),
                device=device
            )
            predict_func = predict_emotion_pure34
            print(f"✅ {model_name} loaded successfully (Pure34)")
            
        elif config['type'] == 'pure50':
            # Load Pure50 model
            model, transform = load_pure50_model(
                model_path=config['path'],
                num_classes=len(config['classes']),
                input_size=config['input_size'],
                device=device
            )
            predict_func = predict_emotion_pure50
            print(f"✅ {model_name} loaded successfully (Pure50)")
            
        elif config['type'] == 'resnet':
            # Load ResNet model
            model, transform = load_resnet_emotion_model(
                model_path=config['path'],
                architecture=config['architecture'],  # resnet50, resnet101
                num_classes=len(config['classes']),
                input_size=config['input_size'],
                device=device
            )
            predict_func = predict_emotion_resnet
            print(f"✅ {model_name} loaded successfully ({config['architecture'].upper()})")
            
        else:
            print(f"❌ Unknown model type for {model_name}: {config['type']}")
            continue
            
        # Store successfully loaded models
        loaded_models[model_name] = model
        model_transforms[model_name] = transform
        model_predict_functions[model_name] = predict_func
            
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        # Remove from enabled models if loading fails
        if model_name in ENABLED_MODELS:
            del ENABLED_MODELS[model_name]

print("=" * 50)
print(f"📊 LOADING SUMMARY:")
print(f"   Successfully loaded: {len(loaded_models)} models")
print(f"   Available for processing: {list(loaded_models.keys())}")

# Update multi-model mode based on actually loaded models
MULTI_MODEL_MODE = len(loaded_models) >= 2

if MULTI_MODEL_MODE:
    print(f"✅ Multi-model mode ENABLED with {len(loaded_models)} models")
else:
    print(f"⚠️ Multi-model mode DISABLED - only {len(loaded_models)} model(s) loaded")

EMOTION_CLASSES = ['sad', 'angry', 'happy', 'relaxed']
print(f"🎭 Emotion classes: {EMOTION_CLASSES}")
print("=" * 50)


# ==========================================
# UPDATED EMOTION PREDICTION FUNCTION
# ==========================================

def predict_emotion_classification_multi(image_path, model_name, head_bbox=None):
    """
    🧠 Predict emotion using specified model from loaded models
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    model_name : str
        Name of the model to use (from loaded_models)
    head_bbox : list, optional
        Head bounding box [x1, y1, x2, y2]
        
    Returns:
    --------
    dict: {'sad': float, 'angry': float, 'happy': float, 'relaxed': float, 'predicted': bool}
    """
    try:
        if model_name not in loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
            
        model = loaded_models[model_name]
        transform = model_transforms[model_name]
        predict_func = model_predict_functions[model_name]
        
        # Use the appropriate prediction function
        emotion_scores = predict_func(
            image_path=image_path,
            model=model,
            transform=transform,
            head_bbox=head_bbox,
            device=device
        )
        
        return emotion_scores
        
    except Exception as e:
        print(f"❌ Error in emotion classification with {model_name} for {image_path}: {e}")
        return {'sad': 0.0, 'angry': 0.0, 'happy': 0.0, 'relaxed': 0.0, 'predicted': False}


# ==========================================
# UPDATED MAIN PROCESSING LOOP FOR MULTI-MODEL
# ==========================================

print("🔄 Starting multi-model image processing pipeline...")
print("=" * 60)

# Results storage for each model
model_results = {model_name: [] for model_name in loaded_models.keys()}
processed_count = 0
skipped_count = 0
error_count = 0

# Progress tracking
total_images = len(image_files)
progress_interval = max(1, total_images // 20)

for i, image_path in enumerate(image_files):
    # Progress indicator
    if i % progress_interval == 0 or i == total_images - 1:
        progress = (i + 1) / total_images * 100
        print(f"📊 Progress: {i+1}/{total_images} ({progress:.1f}%) - {image_path.name}")

    try:
        # 1. Head detection (shared for all models)
        head_result = predict_head_detection(image_path, yolo_head_model)

        # 2. Tail detection (shared for all models)
        tail_result = predict_tail_detection(image_path, yolo_tail_model)

        # 3. ⚠️ FILTERING: Skip image if head or tail not detected
        if not head_result['detected']:
            skipped_count += 1
            print(f"   ⚠️  Skipped {image_path.name}: HEAD not detected")
            continue

        if not tail_result['detected']:
            skipped_count += 1
            print(f"   ⚠️  Skipped {image_path.name}: TAIL not detected")
            continue

        # 4. Get manual label
        manual_label = get_manual_label_from_filename(image_path)
        
        # 5. PROCESS WITH ALL MODELS
        image_processed = False
        
        for model_name in loaded_models.keys():
            try:
                # Emotion classification with current model
                emotion_result = predict_emotion_classification_multi(
                    image_path,
                    model_name,
                    head_bbox=head_result['bbox']
                )

                if emotion_result['predicted']:
                    # Create row for this model
                    row = {
                        'filename': image_path.name,
                        'sad': emotion_result['sad'],
                        'angry': emotion_result['angry'],
                        'happy': emotion_result['happy'],
                        'relaxed': emotion_result['relaxed'],
                        'down': tail_result['down'],
                        'up': tail_result['up'],
                        'mid': tail_result['mid'],
                        'label': manual_label,
                        # Additional metadata
                        'head_confidence': head_result['confidence'],
                        'head_bbox': str(head_result['bbox']),
                        'model_name': model_name
                    }

                    model_results[model_name].append(row)
                    image_processed = True

            except Exception as e:
                print(f"   ❌ Error with {model_name} on {image_path.name}: {e}")
                continue

        if image_processed:
            processed_count += 1
            # Show successful processing for first few images
            if processed_count <= 3:
                print(f"   ✅ Processed {image_path.name} with {len([m for m in loaded_models.keys() if model_results[m]])} models")
        else:
            skipped_count += 1
            print(f"   ⚠️  Skipped {image_path.name}: All models failed")

    except Exception as e:
        error_count += 1
        print(f"   ❌ Error processing {image_path.name}: {e}")
        continue

print("\n" + "=" * 60)
print("📊 MULTI-MODEL PROCESSING SUMMARY")
print("=" * 60)
print(f"📂 Total images found: {total_images}")
print(f"✅ Successfully processed: {processed_count}")
print(f"⚠️  Skipped (filtering): {skipped_count}")
print(f"❌ Errors: {error_count}")
print(f"📈 Success rate: {processed_count/total_images*100:.1f}%")

print(f"\n📊 MODEL-SPECIFIC RESULTS:")
for model_name, results in model_results.items():
    print(f"   {model_name:15s}: {len(results):4d} predictions")

print("=" * 60) 
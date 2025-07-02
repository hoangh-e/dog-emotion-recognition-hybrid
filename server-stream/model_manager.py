import os
import json
from datetime import datetime
from pathlib import Path

class ModelManager:
    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.models_db_path = os.path.join(model_folder, 'models.json')
        self.models = self._load_models()
    
    def _load_models(self):
        """Load models database from JSON file"""
        if os.path.exists(self.models_db_path):
            try:
                with open(self.models_db_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_models(self):
        """Save models database to JSON file"""
        with open(self.models_db_path, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def add_model(self, name, model_type, file_path, filename):
        """Add a new model to the database"""
        self.models[name] = {
            'name': name,
            'type': model_type,  # 'Head-object-detection' or 'Classification'
            'file_path': file_path,
            'filename': filename,
            'added_date': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        self._save_models()
    
    def get_models(self):
        """Get all models as a list"""
        return list(self.models.values())
    
    def get_model(self, name):
        """Get a specific model by name"""
        return self.models.get(name)
    
    def model_exists(self, name):
        """Check if a model with given name exists"""
        return name in self.models
    
    def delete_model(self, name):
        """Delete a model from database and file system"""
        if name in self.models:
            model = self.models[name]
            # Delete file if exists
            if os.path.exists(model['file_path']):
                os.remove(model['file_path'])
            # Remove from database
            del self.models[name]
            self._save_models()
            return True
        return False
    
    def get_models_by_type(self, model_type):
        """Get models filtered by type"""
        return [model for model in self.models.values() if model['type'] == model_type] 
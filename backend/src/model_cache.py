import torch
import logging
from transformers import ViTImageProcessor, ViTForImageClassification
from typing import Optional, Tuple
import threading

class ModelCache:
    """Singleton cache for ML models to avoid reloading on each processing session"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        
        # Model cache
        self._processor: Optional[ViTImageProcessor] = None
        self._model: Optional[ViTForImageClassification] = None
        self._device: Optional[torch.device] = None
        self._emotion_labels = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
        
        # Thread safety
        self._model_lock = threading.Lock()
        self._is_loading = False
        self._load_error = None
    
    def get_models(self) -> Tuple[ViTImageProcessor, ViTForImageClassification, torch.device]:
        """
        Get cached models, loading them if not already loaded.
        
        Returns:
            Tuple of (processor, model, device)
            
        Raises:
            RuntimeError: If model loading fails
        """
        with self._model_lock:
            # Check if we have a previous loading error
            if self._load_error:
                raise RuntimeError(f"Model loading failed: {self._load_error}")
            
            # Return cached models if available
            if self._processor is not None and self._model is not None:
                return self._processor, self._model, self._device
            
            # Prevent multiple simultaneous loading attempts
            if self._is_loading:
                raise RuntimeError("Models are currently being loaded by another thread")
            
            # Load models
            self._is_loading = True
            
        try:
            self.logger.info("Loading emotion detection models...")
            
            # Load processor and model
            self._processor = ViTImageProcessor.from_pretrained('dima806/facial_emotions_image_detection')
            self._model = ViTForImageClassification.from_pretrained('dima806/facial_emotions_image_detection')
            
            # Determine device
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Move model to device
            self._model.to(self._device)
            self._model.eval()
            
            self.logger.info(f"Emotion detection models loaded successfully on {self._device}")
            
            return self._processor, self._model, self._device
            
        except Exception as e:
            self._load_error = str(e)
            self.logger.error(f"Failed to load emotion detection models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
        
        finally:
            with self._model_lock:
                self._is_loading = False
    
    def get_emotion_labels(self) -> list:
        """Get the emotion labels used by the model"""
        return self._emotion_labels.copy()
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._processor is not None and self._model is not None
    
    def clear_cache(self):
        """Clear the model cache (useful for testing or memory management)"""
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
            
            if self._processor is not None:
                del self._processor
                self._processor = None
            
            self._device = None
            self._load_error = None
            self.logger.info("Model cache cleared")
    
    def get_device_info(self) -> dict:
        """Get information about the current device setup"""
        return {
            'device': str(self._device) if self._device else None,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'models_loaded': self.is_loaded()
        }

# Global instance
model_cache = ModelCache() 
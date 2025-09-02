#!/usr/bin/env python3
"""
Clear Model Cache Script
Clears the HuggingFace model cache to force fresh model loading
"""

import sys
import os
import shutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import with absolute imports
try:
    from model_cache import model_cache
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative import method...")
    
    # Try importing from src directory directly
    import importlib.util
    
    def load_module_from_path(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # Load modules manually
    model_cache_module = load_module_from_path('model_cache', 'src/model_cache.py')
    model_cache = model_cache_module.model_cache

def clear_all_caches():
    """Clear all caches to force fresh loading"""
    
    print("🧹 Clearing Model Cache")
    print("=" * 40)
    
    # Clear the model cache
    try:
        model_cache.clear_cache()
        print("✅ Model cache cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing model cache: {e}")
    
    # Clear HuggingFace cache
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(hf_cache_dir):
        try:
            shutil.rmtree(hf_cache_dir)
            print("✅ HuggingFace cache cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing HuggingFace cache: {e}")
    else:
        print("ℹ️  HuggingFace cache directory not found")
    
    # Clear any Python cache files
    cache_dirs = [
        "__pycache__",
        "src/__pycache__",
        ".pytest_cache"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Cleared {cache_dir}")
            except Exception as e:
                print(f"❌ Error clearing {cache_dir}: {e}")
    
    print("\n🎉 All caches cleared!")
    print("💡 Restart your server to load fresh models")

if __name__ == "__main__":
    clear_all_caches() 
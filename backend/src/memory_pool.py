import numpy as np
import torch
import cv2
from typing import List, Dict, Optional, Tuple
from collections import deque
import threading
import logging
from dataclasses import dataclass

@dataclass
class PooledBuffer:
    """A pooled buffer with metadata"""
    data: np.ndarray
    in_use: bool = False
    last_used: float = 0.0
    use_count: int = 0

class MemoryPool:
    """Memory pool for reusing image buffers and tensors"""
    
    def __init__(self, max_pool_size: int = 10):
        self.max_pool_size = max_pool_size
        self.logger = logging.getLogger(__name__)
        
        # Image buffers pool (for face ROIs)
        self.image_pools: Dict[Tuple[int, int, int], deque[PooledBuffer]] = {}
        
        # Tensor pools (for model inputs)
        self.tensor_pools: Dict[Tuple[int, int, int], deque[torch.Tensor]] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
    
    def get_image_buffer(self, height: int, width: int, channels: int = 3) -> np.ndarray:
        """Get a reusable image buffer"""
        key = (height, width, channels)
        
        with self.lock:
            if key in self.image_pools and self.image_pools[key]:
                # Reuse existing buffer
                pooled_buffer = self.image_pools[key].popleft()
                pooled_buffer.in_use = True
                pooled_buffer.use_count += 1
                self.stats['reuses'] += 1
                self.stats['pool_hits'] += 1
                return pooled_buffer.data
            else:
                # Create new buffer
                buffer = np.empty((height, width, channels), dtype=np.uint8)
                self.stats['allocations'] += 1
                self.stats['pool_misses'] += 1
                return buffer
    
    def return_image_buffer(self, buffer: np.ndarray):
        """Return an image buffer to the pool"""
        if buffer is None:
            return
        
        height, width, channels = buffer.shape
        key = (height, width, channels)
        
        with self.lock:
            if key not in self.image_pools:
                self.image_pools[key] = deque()
            
            # Only keep buffers if pool isn't full
            if len(self.image_pools[key]) < self.max_pool_size:
                pooled_buffer = PooledBuffer(data=buffer)  # Store original buffer, not a copy
                self.image_pools[key].append(pooled_buffer)
    
    def get_tensor_buffer(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        """Get a reusable tensor buffer for model inputs"""
        key = (batch_size, height, width)
        
        with self.lock:
            if key in self.tensor_pools and self.tensor_pools[key]:
                # Reuse existing tensor
                tensor = self.tensor_pools[key].popleft()
                self.stats['reuses'] += 1
                self.stats['pool_hits'] += 1
                return tensor
            else:
                # Create new tensor
                tensor = torch.empty((batch_size, 3, height, width), dtype=torch.float32)
                self.stats['allocations'] += 1
                self.stats['pool_misses'] += 1
                return tensor
    
    def return_tensor_buffer(self, tensor: torch.Tensor):
        """Return a tensor buffer to the pool"""
        if tensor is None:
            return
        
        batch_size, channels, height, width = tensor.shape
        key = (batch_size, height, width)
        
        with self.lock:
            if key not in self.tensor_pools:
                self.tensor_pools[key] = deque()
            
            # Only keep tensors if pool isn't full
            if len(self.tensor_pools[key]) < self.max_pool_size:
                # Clear the tensor data
                tensor.zero_()
                self.tensor_pools[key].append(tensor)
    
    def get_face_roi_buffer(self, face_size: Tuple[int, int]) -> np.ndarray:
        """Get a buffer specifically for face ROI extraction"""
        height, width = face_size
        return self.get_image_buffer(height, width, 3)
    
    def cleanup_unused_buffers(self, max_age_seconds: float = 60.0):
        """Clean up old unused buffers to prevent memory leaks"""
        import time
        current_time = time.time()
        
        with self.lock:
            # Clean up image pools
            for key in list(self.image_pools.keys()):
                pool = self.image_pools[key]
                # Remove old buffers
                pool[:] = [buf for buf in pool if current_time - buf.last_used < max_age_seconds]
                
                # Remove empty pools
                if not pool:
                    del self.image_pools[key]
            
            # Clean up tensor pools
            for key in list(self.tensor_pools.keys()):
                pool = self.tensor_pools[key]
                if not pool:
                    del self.tensor_pools[key]
    
    def get_stats(self) -> Dict:
        """Get memory pool statistics"""
        with self.lock:
            total_buffers = sum(len(pool) for pool in self.image_pools.values())
            total_tensors = sum(len(pool) for pool in self.tensor_pools.values())
            
            return {
                **self.stats,
                'total_image_buffers': total_buffers,
                'total_tensor_buffers': total_tensors,
                'pool_types': len(self.image_pools) + len(self.tensor_pools),
                'hit_rate': self.stats['pool_hits'] / max(1, self.stats['pool_hits'] + self.stats['pool_misses'])
            }
    
    def reset_stats(self):
        """Reset statistics"""
        with self.lock:
            self.stats = {
                'allocations': 0,
                'reuses': 0,
                'pool_hits': 0,
                'pool_misses': 0
            }
    
    def clear_all(self):
        """Clear all pools (useful for testing or memory management)"""
        with self.lock:
            self.image_pools.clear()
            self.tensor_pools.clear()
            self.logger.info("Memory pools cleared")

# Global memory pool instance
memory_pool = MemoryPool() 
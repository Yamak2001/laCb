# utils/device_utils.py
import logging
import platform
import torch
from typing import Dict, Any, Tuple, Optional

# Set up logging
logger = logging.getLogger(__name__)

def get_available_device(use_gpu: bool = True) -> Tuple[torch.device, Dict[str, Any]]:
    """
    Determine the best available device for PyTorch models.
    Checks for CUDA, MPS (Metal Performance Shaders for Mac), or falls back to CPU.
    
    Args:
        use_gpu (bool): Whether to attempt using GPU acceleration
        
    Returns:
        Tuple[torch.device, Dict[str, Any]]: 
            - The selected device for PyTorch
            - Dict with device info including type, name, and capabilities
    """
    device_info = {
        "type": "cpu",
        "name": "CPU",
        "capabilities": [],
        "memory_available": None,
        "platform": platform.system()
    }
    
    # If user doesn't want GPU, return CPU immediately
    if not use_gpu:
        logger.info("GPU acceleration disabled by user preference, using CPU")
        return torch.device("cpu"), device_info
    
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_device_count = torch.cuda.device_count()
        cuda_device_props = {
            "compute_capability": torch.cuda.get_device_capability(0),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
        }
        
        logger.info(f"CUDA device available: {cuda_device_name}")
        device_info.update({
            "type": "cuda",
            "name": cuda_device_name,
            "count": cuda_device_count,
            "capabilities": cuda_device_props,
            "memory_available": cuda_device_props["total_memory"]
        })
        return cuda_device, device_info
    
    # Check for MPS (Apple Silicon / Metal)
    try:
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            logger.info("MPS (Metal Performance Shaders) available on Mac")
            device_info.update({
                "type": "mps",
                "name": "Apple Metal",
                "capabilities": ["Metal Performance Shaders"]
            })
            return mps_device, device_info
    except AttributeError:
        # Older PyTorch versions may not have MPS support
        logger.warning("PyTorch version does not support MPS. Consider upgrading for Mac GPU support.")
    except Exception as e:
        logger.warning(f"Error checking MPS availability: {e}")
    
    # Fallback to CPU
    logger.info("No GPU acceleration available, using CPU")
    return torch.device("cpu"), device_info

def optimize_for_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Optimize a PyTorch model for the given device.
    Applies device-specific optimizations when possible.
    
    Args:
        model (torch.nn.Module): The PyTorch model to optimize
        device (torch.device): The device to optimize for
        
    Returns:
        torch.nn.Module: The optimized model
    """
    # Move model to the target device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Apply device-specific optimizations
    if device.type == "cuda":
        # CUDA-specific optimizations (if available in your torch version)
        try:
            # Try to use TensorRT optimization if available
            if hasattr(torch, 'jit') and hasattr(torch.jit, 'trace'):
                logger.info("Applying JIT optimization for CUDA")
                # This is a simple JIT optimization, you might want more complex ones
                # depending on your model needs
                example_input = torch.rand(1, 1, 16000, device=device)  # Example audio input
                model = torch.jit.trace(model, example_input)
        except Exception as e:
            logger.warning(f"CUDA optimization failed, using standard model: {e}")
    
    elif device.type == "mps":
        # MPS-specific optimizations
        # Currently, just moving to device is sufficient for MPS
        logger.info("Using MPS acceleration with standard model")
    
    return model

def get_device_string(device_info: Dict[str, Any]) -> str:
    """
    Get a human-readable string describing the device.
    
    Args:
        device_info (Dict[str, Any]): Device information dictionary
        
    Returns:
        str: Human-readable device description
    """
    if device_info["type"] == "cuda":
        return f"NVIDIA GPU: {device_info['name']}"
    elif device_info["type"] == "mps":
        return f"Apple Metal: {device_info['name']} on {device_info['platform']}"
    else:
        return f"CPU on {device_info['platform']}"
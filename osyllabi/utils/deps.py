"""
Dependency management utilities for Osyllabi.
"""
import sys
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Optional, List

from osyllabi.utils.log import log
from osyllabi.utils.decorators.singleton import singleton

@singleton
class DependencyManager:
    """Manages project dependencies and optimizations."""
    
    @staticmethod
    def check_cuda_capability() -> Tuple[bool, Optional[str]]:
        """Check if system has CUDA capability."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                cuda_version = torch.version.cuda
                return True, f"CUDA {cuda_version} capable device found: {device_name}"
        except ImportError:
            pass

        # Fallback to nvidia-smi check
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, f"NVIDIA GPU detected: {result.stdout.strip()}"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        return False, None

    @staticmethod
    def install_cuda_toolkit() -> Tuple[bool, str]:
        """Install CUDA toolkit if system is capable."""
        cuda_capable, info = DependencyManager.check_cuda_capability()
        if not cuda_capable:
            return False, "No CUDA capable device detected"

        try:
            # Try to install CUDA toolkit via pip
            log.info("Installing CUDA toolkit dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch", "--upgrade"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Verify installation
            import torch
            if torch.cuda.is_available():
                return True, f"CUDA toolkit installed successfully. CUDA version: {torch.version.cuda}"
            else:
                return False, "CUDA installation failed verification"
        except Exception as e:
            return False, f"Failed to install CUDA toolkit: {e}"

    @staticmethod
    def upgrade_faiss() -> Tuple[bool, str]:
        """Upgrade faiss-cpu to faiss-gpu if possible."""
        cuda_capable, _ = DependencyManager.check_cuda_capability()
        if not cuda_capable:
            return False, "No CUDA capable device detected"

        try:
            # Uninstall faiss-cpu if present
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "faiss-cpu"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Install faiss-gpu
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "faiss-gpu"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                # Verify installation
                import faiss
                if hasattr(faiss, 'StandardGpuResources'):
                    return True, "Successfully upgraded to faiss-gpu"
                    
            return False, "Failed to verify faiss-gpu installation"
        except Exception as e:
            return False, f"Failed to upgrade faiss: {e}"

    @classmethod
    def check_deps(cls) -> Dict[str, Dict[str, any]]:
        """Check and optimize all dependencies."""
        results = {
            "cuda": {"status": False, "message": "", "action_taken": False},
            "faiss": {"status": False, "message": "", "action_taken": False}
        }

        # Check CUDA
        cuda_capable, cuda_info = cls.check_cuda_capability()
        if cuda_capable:
            if not cls._is_cuda_installed():
                success, message = cls.install_cuda_toolkit()
                results["cuda"] = {
                    "status": success,
                    "message": message,
                    "action_taken": True
                }
            else:
                results["cuda"] = {
                    "status": True,
                    "message": cuda_info,
                    "action_taken": False
                }

        # Check Faiss
        if cuda_capable and not cls._is_faiss_gpu_installed():
            success, message = cls.upgrade_faiss()
            results["faiss"] = {
                "status": success,
                "message": message,
                "action_taken": True
            }
        else:
            results["faiss"] = {
                "status": cls._is_faiss_gpu_installed(),
                "message": "faiss-gpu already installed" if cls._is_faiss_gpu_installed() else "faiss-cpu in use",
                "action_taken": False
            }

        return results

    @staticmethod
    def _is_cuda_installed() -> bool:
        """Check if CUDA toolkit is installed."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _is_faiss_gpu_installed() -> bool:
        """Check if faiss-gpu is installed and working."""
        try:
            import faiss
            return hasattr(faiss, 'StandardGpuResources')
        except ImportError:
            return False

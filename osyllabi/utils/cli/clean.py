"""
Clean utility functionality for removing temporary files and generated content.
"""
import shutil
from pathlib import Path

from osyllabi.utils.log import log
from osyllabi.utils.const import SUCCESS, FAILURE
from osyllabi.config import OUTPUT_DIR


def clean_directory(directory: Path, force: bool = False) -> bool:
    """
    Clean a specific directory by removing all its contents.
    
    Args:
        directory: Path to the directory to clean
        force: Whether to skip confirmation
            
    Returns:
        bool: True if cleaned successfully, False if cancelled or error
        
    Raises:
        PermissionError: If there are permission issues
    """
    if not directory.exists():
        log.info(f"Directory {directory} does not exist.")
        return True
            
    # Ask for confirmation unless force is True
    if not force:
        confirm = input(f"Are you sure you want to clean {directory}? [y/N] ")
        if confirm.lower() != 'y':
            log.info("Operation cancelled by user.")
            return False
    
    try:
        # Count items before cleaning for better feedback
        item_count = sum(1 for _ in directory.glob("*"))
        
        # Clean the directory
        deleted_files = 0
        deleted_dirs = 0
        
        for item in directory.glob("*"):
            if item.is_file():
                item.unlink()
                deleted_files += 1
            elif item.is_dir():
                shutil.rmtree(item)
                deleted_dirs += 1
                
        # Show detailed cleanup report
        if item_count > 0:
            log.info(f"Cleaned directory: {directory} (removed {deleted_files} files, {deleted_dirs} directories)")
        else:
            log.info(f"Directory {directory} was already empty.")
        
        return True
            
    except PermissionError:
        log.error(f"Permission denied when cleaning {directory}. Try running with elevated privileges.")
        raise


def clean_cache(force: bool = False) -> bool:
    """
    Clean cached files.
    
    Args:
        force: Whether to skip confirmation
        
    Returns:
        bool: True if cleaned successfully, False otherwise
    """
    cache_dir = Path.home() / ".cache" / "osyllabi"
    log.info(f"Cleaning cache directory: {cache_dir}")
    if cache_dir.exists():
        return clean_directory(cache_dir, force)
    else:
        log.info("No cache directory found.")
        return True


def clean_all(force: bool = False) -> bool:
    """
    Clean all generated files.
    
    Args:
        force: Whether to skip confirmation
        
    Returns:
        bool: True if cleaned successfully, False otherwise
    """
    log.info("Performing full cleanup")
    
    success = True
    
    # Clean output directory if it exists
    if OUTPUT_DIR.exists():
        log.info(f"Cleaning output directory: {OUTPUT_DIR}")
        if not clean_directory(OUTPUT_DIR, force):
            success = False
    else:
        log.info(f"Output directory not found: {OUTPUT_DIR}")
        
    # Clean cache
    log.info("Cleaning cache")
    if not clean_cache(force):
        success = False
        
    return success


def clean_from_args(args) -> int:
    """
    Clean files based on command line arguments.
    
    Args:
        args: The parsed command line arguments
        
    Returns:
        int: Exit code (SUCCESS or FAILURE)
    """
    log.info("Cleaning up...")
    
    try:
        if args.output_dir:
            if clean_directory(Path(args.output_dir), args.force):
                log.info("Clean completed successfully!")
                return SUCCESS
        elif args.cache:
            if clean_cache(args.force):
                log.info("Clean completed successfully!")
                return SUCCESS
        elif args.all:
            if clean_all(args.force):
                log.info("Clean completed successfully!")
                return SUCCESS
        else:
            log.warning("No cleaning action specified. Use --output-dir, --cache, or --all.")
            return FAILURE
    except Exception as e:
        log.error(f"Error during cleanup: {e}")
        return FAILURE

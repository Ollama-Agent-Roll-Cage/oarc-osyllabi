"""
Clean command for maintenance tasks.
"""
import argparse
import shutil
import sys
from pathlib import Path

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.parser import setup_clean_arguments
from osyllabi.utils.log import log
from osyllabi.utils.const import SUCCESS, FAILURE


class CleanCommand(Command):
    """Command for cleaning up generated files and temporary data."""
    
    @classmethod
    def register(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        setup_clean_arguments(parser)
    
    def execute(self) -> int:
        """Execute the command and return the exit code."""
        log.info("Cleaning up...")
        
        try:
            if self.args.output_dir:
                self._clean_directory(Path(self.args.output_dir))
            elif self.args.cache:
                self._clean_cache()
            elif self.args.all:
                self._clean_all()
            else:
                log.warning("No cleaning action specified. Use --output-dir, --cache, or --all.")
                print("No cleaning action specified. Use --output-dir, --cache, or --all.")
                print("Run 'osyllabi help clean' for more information.")
                return FAILURE
                
            log.info("Clean completed successfully!")
            print("Clean completed successfully!")
            return SUCCESS
        except Exception as e:
            log.error(f"Error during cleanup: {e}")
            print(f"Error during cleanup: {e}", file=sys.stderr)
            return FAILURE
    
    def _clean_directory(self, directory: Path) -> None:
        """
        Clean a specific directory.
        
        Args:
            directory: Path to the directory to clean
            
        Raises:
            PermissionError: If there are permission issues
        """
        if not directory.exists():
            log.info(f"Directory {directory} does not exist.")
            print(f"Directory {directory} does not exist.")
            return
            
        if not self.args.force:
            confirm = input(f"Are you sure you want to clean {directory}? [y/N] ")
            if confirm.lower() != 'y':
                log.info("Operation cancelled by user.")
                print("Operation cancelled.")
                return
        
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
                print(f"Cleaned directory: {directory}")
                print(f"  Removed {deleted_files} files and {deleted_dirs} directories")
            else:
                log.info(f"Directory {directory} was already empty.")
                print(f"Directory {directory} was already empty.")
                
        except PermissionError:
            error_msg = f"Permission denied when cleaning {directory}. Try running with elevated privileges."
            log.error(error_msg)
            print(error_msg, file=sys.stderr)
            raise
    
    def _clean_cache(self) -> None:
        """Clean cached files."""
        cache_dir = Path.home() / ".cache" / "osyllabi"
        log.info(f"Cleaning cache directory: {cache_dir}")
        if cache_dir.exists():
            self._clean_directory(cache_dir)
        else:
            log.info("No cache directory found.")
            print("No cache directory found.")
    
    def _clean_all(self) -> None:
        """Clean all generated files."""
        from osyllabi.config import OUTPUT_DIR
        
        log.info("Performing full cleanup")
        print("Performing full cleanup:")
        
        # Clean output directory if it exists
        if OUTPUT_DIR.exists():
            log.info(f"Cleaning output directory: {OUTPUT_DIR}")
            print(f"- Cleaning output directory: {OUTPUT_DIR}")
            self._clean_directory(OUTPUT_DIR)
        else:
            log.info(f"Output directory not found: {OUTPUT_DIR}")
            print(f"- Output directory not found: {OUTPUT_DIR}")
            
        # Clean cache
        log.info("Cleaning cache")
        print("- Cleaning cache")
        self._clean_cache()

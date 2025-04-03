"""
Clean command for maintenance tasks.
"""
import argparse
import shutil
import sys
from pathlib import Path

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.parser import setup_clean_arguments
from osyllabi.utils.const import SUCCESS, FAILURE


class CleanCommand(Command):
    """Command for cleaning up generated files and temporary data."""
    
    @classmethod
    def register(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        # Use the centralized parser function to setup arguments
        setup_clean_arguments(parser)
    
    def execute(self) -> int:
        """Execute the command and return the exit code."""
        print("Cleaning up...")
        
        try:
            if self.args.output_dir:
                self._clean_directory(Path(self.args.output_dir))
            elif self.args.cache:
                self._clean_cache()
            elif self.args.all:
                self._clean_all()
            else:
                print("No cleaning action specified. Use --output-dir, --cache, or --all.")
                print("Run 'osyllabi help clean' for more information.")
                return FAILURE
                
            print("Clean completed successfully!")
            return SUCCESS
        except Exception as e:
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
            print(f"Directory {directory} does not exist.")
            return
            
        if not self.args.force:
            confirm = input(f"Are you sure you want to clean {directory}? [y/N] ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return
        
        try:
            # Count items before cleaning for better feedback
            item_count = sum(1 for _ in directory.glob("*"))
            
            # Clean items
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
                print(f"Cleaned directory: {directory}")
                print(f"  Removed {deleted_files} files and {deleted_dirs} directories")
            else:
                print(f"Directory {directory} was already empty.")
                
        except PermissionError:
            print(f"Permission denied when cleaning {directory}. Try running with elevated privileges.", file=sys.stderr)
            raise
    
    def _clean_cache(self) -> None:
        """Clean cached files."""
        cache_dir = Path.home() / ".cache" / "osyllabi"
        if cache_dir.exists():
            self._clean_directory(cache_dir)
        else:
            print("No cache directory found.")
    
    def _clean_all(self) -> None:
        """Clean all generated files."""
        from osyllabi.config import OUTPUT_DIR
        
        print("Performing full cleanup:")
        
        # Clean output directory if it exists
        if OUTPUT_DIR.exists():
            print(f"- Cleaning output directory: {OUTPUT_DIR}")
            self._clean_directory(OUTPUT_DIR)
        else:
            print(f"- Output directory not found: {OUTPUT_DIR}")
            
        # Clean cache
        print("- Cleaning cache")
        self._clean_cache()

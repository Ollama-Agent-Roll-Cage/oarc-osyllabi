"""
Clean command for maintenance tasks.
"""
import argparse
import shutil
from pathlib import Path

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.parser import setup_clean_arguments


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
        
        if self.args.output_dir:
            self._clean_directory(Path(self.args.output_dir))
        elif self.args.cache:
            self._clean_cache()
        elif self.args.all:
            self._clean_all()
        else:
            print("No cleaning action specified. Use --output-dir, --cache, or --all.")
            return 1
            
        print("Clean completed successfully!")
        return 0
    
    def _clean_directory(self, directory: Path) -> None:
        """Clean a specific directory."""
        if not directory.exists():
            print(f"Directory {directory} does not exist.")
            return
            
        if not self.args.force:
            confirm = input(f"Are you sure you want to clean {directory}? [y/N] ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return
                
        for item in directory.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
                
        print(f"Cleaned directory: {directory}")
    
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
        if OUTPUT_DIR.exists():
            self._clean_directory(OUTPUT_DIR)
        self._clean_cache()

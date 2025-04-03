"""
Generation command for creating curriculums.
"""
import argparse

from osyllabi import CurriculumGenerator
from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.parser import setup_create_arguments


class CreateCommand(Command):
    """Command for generating and exporting curriculums."""
    
    @classmethod
    def register_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        # Use the centralized parser function to setup arguments
        setup_create_arguments(parser)
    
    @classmethod
    def help_text(cls) -> str:
        """Return help text for this command."""
        return "Create a new curriculum"
    
    def execute(self) -> int:
        """Execute the command and return the exit code."""
        return self._create_curriculum()
    
    def _create_curriculum(self) -> int:
        """Generate a new curriculum based on CLI arguments."""
        print(f"Creating curriculum on topic: {self.args.topic}")
        
        generator = CurriculumGenerator(
            topic=self.args.topic,
            title=self.args.title,
            skill_level=self.args.skill_level,
            links=self.args.links,
            source=self.args.source
        )
        
        curriculum = generator.create()
        
        if self.args.export_path:
            curriculum.export(
                self.args.export_path,
                fmt=self.args.format
            )
            print(f"Curriculum exported to {self.args.export_path}")
        
        return 0
    
    # Removed _export_curriculum as it's now handled differently

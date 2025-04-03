"""
Generation command for creating curriculums.
"""
import argparse

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.parser import setup_create_arguments


class CreateCommand(Command):
    """Command for generating and exporting curriculums."""
    
    @classmethod
    def register(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        # Use the centralized parser function to setup arguments
        setup_create_arguments(parser)
    
    def execute(self) -> int:
        """Execute the command and return the exit code."""
        # Only import CurriculumGenerator when actually needed for execution
        from osyllabi import CurriculumGenerator
        return self._create_curriculum(CurriculumGenerator)
    
    def _create_curriculum(self, generator_class) -> int:
        """Generate a new curriculum based on CLI arguments."""
        print(f"Creating curriculum on topic: {self.args.topic}")
        
        generator = generator_class(
            topic=self.args.topic,
            title=self.args.title,
            skill_level=self.args.level,  # Changed from skill_level to level
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

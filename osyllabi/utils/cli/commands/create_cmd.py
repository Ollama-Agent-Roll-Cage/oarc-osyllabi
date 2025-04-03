"""
Generation command for creating curriculums.
"""
import os
import argparse
import sys

from osyllabi.utils.cli.cmd import Command
from osyllabi.utils.cli.parser import setup_create_arguments
from osyllabi.utils.log import log
from osyllabi.utils.const import SUCCESS, FAILURE
from osyllabi.utils.paths import get_output_directory, create_unique_file_path


class CreateCommand(Command):
    """Command for generating and exporting curriculums."""
    
    @classmethod
    def register(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        setup_create_arguments(parser)
    
    def execute(self) -> int:
        """Execute the command and return the exit code."""
        try:
            from osyllabi import CurriculumGenerator
            return self._create_curriculum(CurriculumGenerator)
        except Exception as e:
            log.error(f"Error creating curriculum: {e}")
            print(f"Error creating curriculum: {e}", file=sys.stderr)
            if 'OSYLLABI_DEBUG' in os.environ:
                import traceback
                traceback.print_exc()
            return FAILURE
    
    def _create_curriculum(self, generator_class) -> int:
        """Generate a new curriculum based on CLI arguments."""
        log.info(f"Creating curriculum on topic: {self.args.topic}")
        print(f"Creating curriculum on topic: {self.args.topic}")
        
        generator = generator_class(
            topic=self.args.topic,
            title=self.args.title,
            skill_level=self.args.level,
            links=self.args.links,
            source=self.args.source
        )
        
        curriculum = generator.create()
        
        if self.args.export_path:
            export_path = self.args.export_path
        else:
            # Use default output directory if not specified
            output_dir = get_output_directory()
            filename = curriculum.title if curriculum.title else curriculum.topic
            export_path = create_unique_file_path(output_dir, filename, self.args.format)
        
        # Export the curriculum
        try:
            result_path = curriculum.export(export_path, fmt=self.args.format)
            log.info(f"Curriculum exported to {result_path}")
            print(f"Curriculum exported to {result_path}")
            return SUCCESS
        except NotImplementedError as e:
            log.error(f"Export error: {e}")
            print(f"Export error: {e}", file=sys.stderr)
            print(f"Supported formats are: {', '.join(self.args.format.choices)}")
            return FAILURE
        except Exception as e:
            log.error(f"Error exporting curriculum: {e}")
            print(f"Error exporting curriculum: {e}", file=sys.stderr)
            return FAILURE

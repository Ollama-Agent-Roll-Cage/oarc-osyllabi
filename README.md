# Osyllabi

A streamlined Python app for designing personalized curriculums using AI, web crawling, and data integration.

## Features

- AI-driven curriculum generation customized to individual learning goals and preferences.
- Advanced web crawling to collect the latest and most relevant educational resources.
- Seamless integration with multiple educational platforms for a unified experience.
- Fully customizable learning paths with built-in progress tracking and analytics.
- Flexible export options for easy sharing and implementation of curriculums.

## Installation

```bash
# Clone the repository
git clone https://github.com/p3nGu1nZz/osyllabi.git
cd osyllabi

# Create and use virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate # On Linux: source venv/bin/activate 

# Install dependencies
pip install -e . # For development: 'pip install -e .[dev]'
```

## Requirements

- Python 3.12 or higher
- [Ollama](https://ollama.ai/download) running locally (required for AI operations)

## Usage

### Command Line

```bash
# Create a curriculum with basic options
osyllabi create "Machine Learning" --format md

# Create with detailed options
osyllabi create "Machine Learning" \
    --title "ML Fundamentals" \
    --level "Beginner" \
    --links "https://www.wikipedia.org/" \
    --source ./docs/ \
    --format "json"

# Show help
osyllabi help create

# Clean up generated files
osyllabi clean --all
```

### Python API

```python
from osyllabi import Curriculum

# Create a curriculum
curriculum = Curriculum(
    topic="Machine Learning",
    title="ML Fundamentals", 
    skill_level="Beginner", 
    links=["https://www.wikipedia.org/"],
    source=["./docs/"]
)

# Generate content
curriculum.generate_content()

# Export the curriculum
curriculum.export("./output/ml_curriculum.md")
```

## Customizing Templates

Osyllabi uses a template system for generating different parts of the curriculum. The default templates are provided for:

- **Overview**: Introduction and learning objectives
- **Learning Path**: Structured progression of modules
- **Resources**: Curated learning materials
- **Projects**: Practical exercises and projects

You can customize these templates by modifying the `PromptTemplate` class constants in `osyllabi/ai/prompts.py` before installation.

## License

This project is licensed under the [Apache 2.0 License](LICENSE)

## Citations

Please use the following BibTeX entry to cite this project:

```bibtex
@software{osyllabi,
  author = {Kara Rawson},
  title = {Osyllabi: A streamlined Python app for designing personalized curriculums using AI, web crawling, and data integration.},
  year = {2025},
  howpublished = {\url{https://github.com/p3nGu1nZz/Osyllabi}},
  note = {Accessed: 2026-01-26}
}
```

## Contact

For questions or support, please contact us at:

- **Email**: <backrqqms@gmail.com>
- **Discord**: [Join our Discord](https://discord.gg/2xpqjDUkHD)
- **Issues**: [GitHub Issues](https://github.com/p3nGu1nZz/osyllabi/issues)

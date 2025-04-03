# Osyllabi

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

Osyllabi is a streamlined Python app for designing personalized curriculums using AI, web crawling, and data integration.

## Features

- AI-driven curriculum generation customized to individual learning goals and preferences.
- Advanced web crawling to collect the latest and most relevant educational resources.
- Seamless integration with multiple educational platforms for a unified experience.
- Fully customizable learning paths with built-in progress tracking and analytics.
- Flexible export options for easy sharing and implementation of curriculums.

## Installation

```bash
# Clone the repository
git clone https://github.com/p3nGu1nZz/Osyllabi.git
cd Osyllabi

# Create and use virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate # On Linux: source venv/bin/activate 

# Install dependencies
pip install -e .
```

## Usage

### Command Line

```ps
osyllabus create \
    --topic "Machine Learning" \
    --title "Learn how to use Osyllabi" \
    --level "Beginner" \
    --links "https://www.wikipedia.org/" \
    --source "."
```

### Python

```python
# Example of how to use Osyllabus for basic curriculum generation
from osyllabi import CurriculumGenerator

# Initialize the generator with topic and skill level
generator = CurriculumGenerator(
    topic="Machine Learning",
    title="Learn how to use Osyllabi", 
    level="Beginner", 
    links=["https://www.wikipedia.org/"],
    source=["."]) # project root    

# Generate a personalized curriculum
curriculum = generator.create()

# Export the curriculum
curriculum.export(".\docs")
```

For more detailed examples, see the [examples](examples/) directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

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
- **Issues**: [GitHub Issues](https://github.com/p3nGu1nZz/Osyllabus/issues)

# Contributing to RLM

Thank you for your interest in contributing to the Recursive Language Model (RLM) project!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/ondrasek/spike-claude-code-rlm.git
cd spike-claude-code-rlm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install ruff mypy  # Development tools
```

## Code Style

This project uses:
- **Python 3.11+** features and syntax (compatible with 3.13+)
- **Type hints** for all functions and methods
- **ruff** for linting and formatting
- **mypy** for static type checking

Run linters before committing:
```bash
ruff check .
mypy rlm/
```

## Testing

Run the test suite:
```bash
uv run pytest -x --tb=short -m "not slow"
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run linters and tests
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Code Guidelines

- Use modern Python 3.11+ features (type hints, dataclasses, etc.)
- Follow existing code structure and patterns
- Add docstrings to all public functions and classes
- Keep security in mind - validate any code execution carefully
- Write clear commit messages

## Areas for Contribution

- Additional LLM backends
- Performance optimizations
- Better error handling and recovery
- Enhanced security features
- Documentation improvements
- Example use cases

## Questions?

Open an issue for questions or discussions about the project.

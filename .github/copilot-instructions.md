# GitHub Copilot Instructions for spike-claude-code-rlm

## Project Overview

This repository implements the **RLM (Reinforcement Learning Model) pattern** as described in the research paper: https://arxiv.org/pdf/2512.24601

The goal is to create a modern, high-quality Python implementation following current best practices and leveraging Python 3.13+ features.

## Tech Stack

- **Python Version**: 3.13 or higher
- **Language Features**: Use modern Python 3.13+ features where appropriate
- **Package Management**: Standard Python tooling (pip, venv)
- **Code Quality**: Follow PEP 8 and modern Python best practices

## Coding Standards

### Python Style Guidelines

- Follow **PEP 8** for code formatting and style
- Use **type hints** for all function signatures and class attributes
- Prefer **f-strings** for string formatting
- Prefer **dataclasses** from the standard library for data structures. Only use **Pydantic models** if Pydantic is already an approved dependency or there is a clear need for its validation/parsing features.
- Write **docstrings** for all public modules, classes, and functions (Google or NumPy style)
- Keep functions focused and concise (ideally under 50 lines)
- Use meaningful variable and function names that clearly express intent

### Code Organization

- Organize code into logical modules and packages
- Keep related functionality together
- Separate concerns (data models, business logic, utilities)
- Use `__init__.py` files to define package interfaces

### Error Handling

- Use specific exception types rather than generic `Exception`
- Provide informative error messages
- Handle errors at appropriate levels (don't catch and ignore silently)
- Use context managers (`with` statements) for resource management

### Testing (when implemented)

- Write unit tests for all new functionality
- Use descriptive test names that explain what is being tested
- Follow AAA pattern (Arrange, Act, Assert)
- Aim for high test coverage, especially for core functionality
- Use `pytest` as the testing framework

### Dependencies

- Minimize external dependencies when possible
- Use well-maintained, popular libraries when needed
- Pin dependency versions for reproducibility
- Document why each dependency is needed

## Development Workflow

### Before Making Changes

1. Understand the RLM pattern from the referenced paper
2. Review existing code structure and patterns
3. Check for any existing tests or documentation
4. Ensure changes align with the project's goals

### Making Changes

1. Write clean, readable code that follows the coding standards
2. Add or update docstrings and comments where necessary
3. Consider edge cases and error handling
4. Keep changes focused and minimal
5. Ensure backward compatibility unless breaking changes are explicitly required

### Before Committing

1. Verify code follows PEP 8 (use `ruff` or `black` for formatting if available)
2. Check type hints with `mypy` if type checking is set up
3. Run tests if they exist
4. Review your changes to ensure they're minimal and focused
5. Write clear, descriptive commit messages

## Key Principles

- **Simplicity over complexity**: Choose the simplest solution that works
- **Readability counts**: Code is read more often than written
- **Explicit is better than implicit**: Be clear about intent and behavior
- **Modern Python**: Leverage Python 3.13+ features for cleaner, more efficient code
- **Scientific rigor**: When implementing the RLM pattern, stay faithful to the research paper
- **Documentation matters**: Good docs help both humans and AI understand the code

## Project-Specific Guidelines

### RLM Pattern Implementation

- Stay true to the concepts and algorithms described in the referenced paper
- Use clear mathematical notation in comments when explaining algorithms
- Include references to specific sections of the paper where relevant
- Implement the pattern in a modular, extensible way
- Consider performance implications, especially for learning/training loops

### Python 3.13+ Features to Consider

- Pattern matching (`match`/`case`) for complex conditional logic
- New type system improvements
- Performance enhancements in the standard library
- Improved error messages and debugging capabilities

## Common Patterns to Avoid

- Don't use mutable default arguments (`def func(arg=[]):`)
- Avoid bare `except:` clauses
- Don't use `import *` (always import specific names)
- Avoid deeply nested code (max 3-4 levels of indentation)
- Don't shadow built-in names
- Avoid premature optimization

## Resources

- **Paper**: https://arxiv.org/pdf/2512.24601
- **Python 3.13 Docs**: https://docs.python.org/3.13/
- **PEP 8**: https://peps.python.org/pep-0008/
- **Type Hints**: https://docs.python.org/3/library/typing.html

## Notes for GitHub Copilot

When working on this repository:

1. Always prioritize code quality and readability over clever tricks
2. Use type hints comprehensively to improve code clarity and catch errors early
3. When in doubt about the RLM pattern implementation, refer to the paper
4. Keep the Python 3.13+ requirement in mind when suggesting solutions
5. Focus on creating a well-structured, maintainable implementation
6. Document complex algorithms and design decisions
7. Consider both correctness and performance in machine learning contexts

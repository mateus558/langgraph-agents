# Contributing to AI Server

Thank you for your interest in contributing to AI Server! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a positive environment

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/ai-server.git
   cd ai-server
   ```

2. **Set up development environment**
   ```bash
   make install-dev
   make pre-commit
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### 1. Code Style

We use automated tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

Format your code:
```bash
make format
```

Check for issues:
```bash
make lint
make type-check
```

### 2. Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Include edge cases

Run tests:
```bash
make test
make test-cov
```

### 3. Documentation

- Add docstrings to all functions and classes
- Update README.md for user-facing changes
- Include code examples
- Document configuration options

### 4. Commit Guidelines

Follow conventional commit format:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(chat): add streaming support for responses

fix(websearch): handle timeout errors gracefully

docs(readme): update installation instructions
```

### 5. Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run all checks**:
   ```bash
   make format
   make lint
   make type-check
   make test
   ```
4. **Create pull request** with:
   - Clear title and description
   - Reference related issues
   - List of changes
   - Testing performed

5. **Address review feedback** promptly

## Code Review Checklist

Before submitting, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No hard-coded values (use config/env vars)
- [ ] Type hints are added
- [ ] Logging instead of print statements
- [ ] Error handling is implemented
- [ ] No security vulnerabilities introduced

## Architecture Guidelines

### Agent Design

- Implement `AgentProtocol` for consistency
- Use dataclasses for configuration
- Keep nodes pure (return deltas, don't mutate state)
- Add proper logging

### Configuration

- Use environment variables for deployment settings
- Provide sensible defaults
- Document all configuration options
- Validate configuration at startup

### Error Handling

- Use custom exceptions from `core.exceptions`
- Log errors with context
- Provide user-friendly error messages
- Handle edge cases gracefully

### Testing

- Unit tests for utilities
- Integration tests for agents
- Mock external dependencies
- Test error conditions

## Project Structure Guidelines

```
src/
  ├── core/           # Core protocols, exceptions, validation
  ├── chatagent/      # Chat agent implementation
  ├── websearch/      # Web search agent
  └── utils/          # Shared utilities

tests/
  ├── unit/           # Unit tests
  ├── integration/    # Integration tests
  └── fixtures/       # Test fixtures
```

## Common Tasks

### Adding a New Agent

1. Create package in `src/`
2. Implement `AgentProtocol`
3. Add configuration with validation
4. Write comprehensive tests
5. Update documentation
6. Add examples

### Adding Configuration Options

1. Add to `.env.example` with description
2. Update config classes
3. Add validation if needed
4. Document in README.md
5. Provide sensible default

### Fixing a Bug

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Ensure test passes
4. Add regression test if needed
5. Document the fix

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🎉

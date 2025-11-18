# Contributing to Multi-Modal Stock Intelligence Platform

Thank you for your interest in contributing to the Multi-Modal Stock Intelligence Platform! This document provides guidelines and instructions for contributing to the project.

## Welcome

We appreciate your interest in improving this AI-driven stock intelligence system. Whether you're fixing bugs, adding features, improving documentation, or suggesting enhancements, your contributions are valued.

## Code of Conduct

By participating in this project, you agree to:
- Be respectful and inclusive towards all contributors
- Provide constructive feedback
- Accept constructive criticism gracefully
- Focus on what is best for the community and project

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, etc.)
- Screenshots or logs if applicable

### Suggesting Features

To suggest a new feature:
- Create a GitHub Issue with the `enhancement` label
- Clearly describe the feature and its benefits
- Explain your use case
- Discuss potential implementation approaches

### Submitting Pull Requests

We welcome pull requests for bug fixes, features, and improvements!

## Development Setup

1. **Fork and Clone the Repository**

```bash
git clone https://github.com/your-username/stock-intelligence.git
cd stock-intelligence
```

2. **Set Up Development Environment**

Follow the setup instructions in [README.md](README.md):
- Install dependencies: `pip install -r requirements.txt`
- Configure environment variables: `cp env.template .env`
- Start services: `docker-compose up -d`

3. **Create a Feature Branch**

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-technical-indicators`
- `bugfix/fix-websocket-connection`
- `docs/update-api-documentation`

## Coding Standards

### Python

- **Style Guide**: Follow PEP 8
- **Formatter**: Use Black with line length 100
  ```bash
  black --line-length 100 backend/
  ```
- **Linting**: Run flake8 before committing
  ```bash
  flake8 backend/
  ```
- **Type Hints**: Required for all function signatures
  ```python
  def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
      """Calculate Relative Strength Index."""
      pass
  ```
- **Docstrings**: Use Google style
  ```python
  def fetch_stock_data(ticker: str, start_date: str) -> pd.DataFrame:
      """Fetch historical stock data.
      
      Args:
          ticker: Stock symbol (e.g., 'RELIANCE')
          start_date: Start date in YYYY-MM-DD format
          
      Returns:
          DataFrame with OHLCV data
          
      Raises:
          ValueError: If ticker is invalid
      """
      pass
  ```

### JavaScript/TypeScript (Frontend)

- Follow ESLint configuration
- Use Prettier for formatting
- Use TypeScript for type safety

## Testing Requirements

### Writing Tests

- **Coverage**: All new features must include unit tests
- **Target**: Maintain test coverage above 80%
- **Framework**: pytest for Python tests

### Test Structure

```python
# tests/unit/test_feature_engineering.py
import pytest
from backend.services.feature_engineering import calculate_technical_indicators

@pytest.mark.unit
def test_calculate_rsi():
    """Test RSI calculation."""
    # Arrange
    prices = pd.Series([100, 102, 101, 103, 105])
    
    # Act
    rsi = calculate_rsi(prices)
    
    # Assert
    assert rsi is not None
    assert 0 <= rsi.iloc[-1] <= 100
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_feature_engineering.py

# Run with coverage
pytest --cov=backend --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

## Commit Message Guidelines

Use conventional commits format:

```
<type>: <description>

[optional body]

[optional footer]
```

### Types

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `style:` Code style changes (formatting, no logic change)
- `chore:` Maintenance tasks

### Examples

```bash
feat: add LSTM forecasting model

fix: resolve WebSocket connection timeout issue

docs: update API endpoint documentation

test: add unit tests for sentiment analysis

refactor: optimize data preprocessing pipeline

chore: update dependencies
```

### Guidelines

- Keep the first line under 72 characters
- Use present tense ("add feature" not "added feature")
- Be descriptive but concise

## Pull Request Process

1. **Update Documentation**
   - Update README.md if adding features
   - Update API documentation
   - Add docstrings to new functions

2. **Ensure Tests Pass**
   ```bash
   pytest tests/ -v
   ```

3. **Check Code Quality**
   ```bash
   black --check backend/
   flake8 backend/
   mypy backend/
   ```

4. **Update CHANGELOG** (if applicable)

5. **Submit Pull Request**
   - Provide a clear description of changes
   - Reference related issues
   - Add screenshots for UI changes
   - Request review from maintainers

6. **Address Review Comments**
   - Respond to feedback promptly
   - Make requested changes
   - Re-request review after updates

7. **Squash Commits**
   - Before merge, squash commits into logical units
   - Keep history clean and readable

## Branch Naming Conventions

- `feature/description` - New features
  - Example: `feature/add-sentiment-analysis`
- `bugfix/description` - Bug fixes
  - Example: `bugfix/fix-data-loading`
- `hotfix/description` - Urgent fixes for production
  - Example: `hotfix/patch-security-vulnerability`
- `docs/description` - Documentation updates
  - Example: `docs/improve-setup-guide`
- `refactor/description` - Code refactoring
  - Example: `refactor/optimize-model-inference`

## Review Process

- Pull requests require at least one approval from maintainers
- All CI/CD checks must pass
- Code coverage must not decrease
- Address all review comments before merge
- Maintainers will merge approved PRs

## Development Workflow

1. Create an issue or claim an existing one
2. Fork the repository
3. Create a feature branch
4. Implement changes with tests
5. Commit with conventional commit messages
6. Push to your fork
7. Submit a pull request
8. Participate in code review
9. Make requested changes
10. PR gets merged!

## Need Help?

- Check existing [issues](https://github.com/your-repo/issues) and [pull requests](https://github.com/your-repo/pulls)
- Review [documentation](docs/)
- Ask questions in GitHub Discussions
- Contact project maintainers

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

---

Thank you for contributing to the Multi-Modal Stock Intelligence Platform! Your efforts help make this project better for everyone.


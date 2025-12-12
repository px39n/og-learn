# Contributing

We welcome contributions to OG-Learn!

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/og-learn.git
cd og-learn
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install in development mode:

```bash
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8
- Use type hints where practical
- Write docstrings for public functions

## Running Tests

```bash
pytest tests/
```

## Documentation

Build docs locally:

```bash
mkdocs serve
```

View at http://localhost:8000

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Issues

Report bugs and request features via [GitHub Issues](https://github.com/your-username/og-learn/issues).


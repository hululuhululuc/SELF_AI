# Contributing to Self AI

Thanks for your interest in contributing.

## Development Setup

1. Fork and clone the repository.
2. Create a virtual environment.
3. Install development dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
```

4. Copy env template and configure required values.

```powershell
copy .env.example .env
```

## Recommended Workflow

1. Create a branch from `main`:

```powershell
git checkout -b feat/short-description
```

2. Make focused changes with clear commit messages.
3. Run tests locally before opening a PR:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

4. Push your branch and open a Pull Request.

## Pull Request Guidelines

- Keep PRs small and focused.
- Explain what changed and why.
- Link related issues if available.
- Include test updates when behavior changes.
- Do not commit secrets or local environment files.

## Code Style

- Follow existing style in the repository.
- Keep names and comments clear and practical.
- Prefer explicit error handling and observable failure reasons.

## Reporting Issues

When filing an issue, include:

- Environment (OS, Python version)
- Reproduction steps
- Expected behavior
- Actual behavior
- Relevant logs or trace snippets

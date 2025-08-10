# Main Project Context

## Project Overview

This project is focused on modeling and mapping the electrostatic potential on the lunar surface. It involves analyzing data from lunar missions, fitting physical models (like the kappa distribution), and generating potential maps.

## General Instructions

### Development Environment

The project uses `uv` for package and environment management. To set up the development environment, run:

```bash
uv sync
```

This will install all the dependencies listed in `pyproject.toml` and `uv.lock`.

To install the tools used for the project: `pytest`, `ruff`, `black`
```bash
uv sync --locked --all-extras --dev
```

### Running Tests

To run the entire test suite, use the following command from the project's root directory:

```bash
uv run pytest
```

To run a specific test file, you can pass the path to the file:

```bash
uv run pytest tests/physics/test_charging.py
```

### Commit Message Guidelines

To ensure a clear and consistent commit history, this project adheres to the **Conventional Commits** specification. Each commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**

*   **feat:** A new feature
*   **fix:** A bug fix
*   **docs:** Documentation only changes
*   **style:** Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
*   **refactor:** A code change that neither fixes a bug nor adds a feature
*   **perf:** A code change that improves performance
*   **test:** Adding missing tests or correcting existing tests
*   **build:** Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)
*   **ci:** Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)
*   **chore:** Other changes that don't modify `src` or `test` files
*   **revert:** Reverts a previous commit

**Example:**

```
feat(physics): add new charging model

Implement a new photo-electron charging model based on the work of...

Fixes #123
```

## Directory Structure

*   `src/`: Main source code for the project.
*   `tests/`: Unit and integration tests.
*   `notebooks/`: Jupyter notebooks for analysis and exploration.
*   `data/`: Data files, including SPICE kernels.
*   `scripts/`: Helper scripts for data processing and analysis.
*   `docs/`: Documentation and notes regarding specific design decisions.
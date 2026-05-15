# Project Conventions

## Dependency Management

This project uses `uv` exclusively for dependency management and command execution.

- **Always** execute Python commands via `uv run`, e.g. `uv run python <script>` or `uv run pytest`.
- For tests or temporary analysis, create a one-shot isolated environment with `uv run --with`, e.g. `uv run --with pandas numpy python my_script.py`.
- Never use `python`, `python3`, `pip`, `pip3`, `source venv`, `activate`, or direct virtualenv paths in shell commands.
- Use `uv add` and `uv remove` instead of `pip` for package management.

## General Principles

- Prefer workspace facts over assumptions.
- Start from the smallest relevant context.
- Use the fewest tools necessary to complete the task.
- Do not make unrelated refactors.

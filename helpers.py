"""Shared helpers: colored logging and project root."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RED, YELLOW, GREEN, RESET = "\033[31m", "\033[33m", "\033[32m", "\033[0m"
log = lambda message: print(message)
log_error = lambda message: print(f"{RED}{message}{RESET}")
log_warning = lambda message: print(f"{YELLOW}{message}{RESET}")
log_success = lambda message: print(f"{GREEN}{message}{RESET}")

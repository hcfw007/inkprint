#!/usr/bin/env bash
# Convenience wrapper around setup.py. Windows users: run `python setup.py` directly.
exec python3 "$(dirname "$0")/setup.py" "$@"

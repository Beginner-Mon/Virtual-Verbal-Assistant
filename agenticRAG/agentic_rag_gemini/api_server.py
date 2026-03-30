"""Compatibility entrypoint for AgenticRAG API server.

The implementation now lives in api_server_pkg to support incremental modularization.
"""

from api_server_pkg.app import *  # noqa: F401,F403
from api_server_pkg.app import main


if __name__ == "__main__":
    main()

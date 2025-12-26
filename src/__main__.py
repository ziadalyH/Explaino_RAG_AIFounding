"""Entry point for running the API server as a module."""

import os
from .api import run_api_server

if __name__ == '__main__':
    # Get configuration from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    debug = os.getenv('API_DEBUG', 'false').lower() == 'true'
    
    run_api_server(host=host, port=port, debug=debug)

"""Entry point for the RAG Chatbot Backend.

This module provides the main entry point for the CLI application.
It handles argument parsing, configuration loading, and command execution.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.cli import CLI, create_parser


def main() -> int:
    """
    Main entry point for the RAG Chatbot Backend CLI.
    
    This function:
    1. Parses command-line arguments
    2. Loads configuration from files and environment variables
    3. Initializes the CLI with configuration
    4. Executes the requested command
    5. Returns appropriate exit code
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config.from_env()
        
        # Validate configuration
        config.validate()
        
        # Initialize CLI
        cli = CLI(config)
        
        # Execute command
        exit_code = cli.run(args)
        return exit_code
        
    except ValueError as e:
        # Configuration validation error
        print(f"Configuration Error: {str(e)}", file=sys.stderr)
        print("\nPlease check your configuration file and environment variables.", file=sys.stderr)
        print("Required: OPENAI_API_KEY must be set", file=sys.stderr)
        return 1
        
    except FileNotFoundError as e:
        # Missing configuration file or data directory
        print(f"File Not Found Error: {str(e)}", file=sys.stderr)
        print("\nPlease ensure all required files and directories exist.", file=sys.stderr)
        return 1
        
    except KeyboardInterrupt:
        # User interrupted execution
        print("\n\nOperation cancelled by user", file=sys.stderr)
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        # Unexpected error
        print(f"Unexpected Error: {str(e)}", file=sys.stderr)
        
        if args.debug if hasattr(args, 'debug') else False:
            import traceback
            traceback.print_exc()
        else:
            print("\nRun with --debug flag for detailed error information", file=sys.stderr)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("huggingface-text-data-analyzer")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Default development version

# Create common paths
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"

# Import main components for easier access
from .src.base_analyzer import BaseAnalyzer
from .src.advanced_analyzer import AdvancedAnalyzer
from .src.report_generator import ReportGenerator
from .src.utils import setup_logging, parse_args

# Export main components
__all__ = [
    "BaseAnalyzer",
    "AdvancedAnalyzer",
    "ReportGenerator",
    "setup_logging",
    "parse_args",
]
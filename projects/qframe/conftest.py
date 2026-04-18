"""
pytest configuration — adds src/ to sys.path so `import qframe` works
without needing to pip-install the package first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

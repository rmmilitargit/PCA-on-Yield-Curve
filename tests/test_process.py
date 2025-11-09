import os
import sys
import importlib

# Ensure local src/ is on sys.path for imports during tests
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

process = importlib.import_module('process')

def test_run_full_pipeline_exists():
    assert hasattr(process, 'run_full_pipeline')

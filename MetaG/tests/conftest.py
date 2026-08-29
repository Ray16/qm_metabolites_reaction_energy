"""Make `import metag` work when running the tests from a source checkout (no install needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

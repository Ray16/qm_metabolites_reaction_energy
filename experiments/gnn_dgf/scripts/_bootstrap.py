"""Put the package root on sys.path so `import gnn` works from scripts/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

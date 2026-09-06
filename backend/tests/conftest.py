"""
SatQuery AI — pytest configuration
"""
import sys
import os

# Ensure backend/ is on sys.path so that `from core.xxx import` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

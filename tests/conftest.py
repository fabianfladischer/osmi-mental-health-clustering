import sys
from pathlib import Path

# projektordner in den suchpfad, damit "src" importiert werden kann
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

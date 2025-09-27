from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hello import hello_world


def test_hello_world_returns_greeting():
    assert hello_world() == "Hello, world!"

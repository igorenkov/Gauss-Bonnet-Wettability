from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR

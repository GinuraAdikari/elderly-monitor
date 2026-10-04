import pytest
from monitor import config


@pytest.fixture(scope="session")
def cfg():
    return config.load()

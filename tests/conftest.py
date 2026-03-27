import pytest
from rentals_agents.benchmark import Benchmark

@pytest.fixture(autouse=True)
def reset_benchmark():
    Benchmark.reset()
    yield
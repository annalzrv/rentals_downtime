import time
from typing import Optional

class Benchmark:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.final_mse: Optional[float] = None

    @classmethod
    def reset(cls):
        cls._instance = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def add_tokens(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def set_final_mse(self, mse: float):
        self.final_mse = mse

    def report(self) -> str:
        total_tokens = self.total_input_tokens + self.total_output_tokens
        return f"""=== Benchmark Report ===
Total execution time: {self.duration:.2f} seconds
Total input tokens: {self.total_input_tokens}
Total output tokens: {self.total_output_tokens}
Total tokens: {total_tokens}
Final MSE: {self.final_mse if self.final_mse is not None else "N/A"}
"""
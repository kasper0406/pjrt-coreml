import subprocess
import threading
import time
from dataclasses import dataclass, replace
from contextlib import contextmanager
from typing import Callable
import numpy as np


@dataclass
class Stats:
    time: float # unit s
    cpu: int  # unit J
    gpu: int  # unit J
    ane: int  # unit J

# Notice, this assumes the computation period is >>sample_period in duration
class Benchmarking:
    sample_period: int
    is_capturing: bool
    
    collected_stats: Stats

    def __init__(self):
        self.sample_period = 25  # ms
        self.is_capturing = False

        self.thread = threading.Thread(target=self._collect_powermetrics_data)
        self.thread.start()

    def _collect_powermetrics_data(self):
        self.process = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "gpu_power,ane_power,cpu_power", "--sample-rate", str(self.sample_period)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        def parse_mw(str):
            parts = str.strip().split(" ")
            return int(parts[-2])

        for line in self.process.stdout:
            if self.is_capturing:
                if line.startswith("CPU Power: "):
                    self.collected_stats.cpu += float(parse_mw(line) * self.sample_period) / 1000000.0
                elif line.startswith("GPU Power: "):
                    self.collected_stats.gpu += float(parse_mw(line) * self.sample_period) / 1000000.0
                elif line.startswith("ANE Power: "):
                    self.collected_stats.ane += float(parse_mw(line) * self.sample_period) / 1000000.0

    @contextmanager
    def collect(self):
        self.collected_stats = Stats(time=time.time(), cpu=0, gpu=0, ane=0)
        self.is_capturing = True
        try:
            yield
        finally:
            self.is_capturing = False
            end_time = time.time()
            # Update to the duration
            self.collected_stats = replace(self.collected_stats, time=end_time - self.collected_stats.time)

    def results(self) -> Stats:
        return self.collected_stats
    
    def done_benchmarking(self):
        self.process.terminate()
        self.thread.join()

@dataclass
class MeasurementStats:
    mean: float
    std: float

@dataclass
class MeasurementResult:
    name: str

    time: MeasurementStats
    cpu: MeasurementStats
    ane: MeasurementStats
    gpu: MeasurementStats

def benchmark(name: str, benchmarking: Benchmarking, function: Callable, num_trials: int = 5, baseline_collection_time: int = 0.5) -> MeasurementResult:
    baselines = []
    results = []

    # Pre-warm the function in case it needs to compile
    function()

    for trial in range(num_trials):
        print(f"Running trial {trial}")
        # Collect baseline stats
        with benchmarking.collect():
            time.sleep(baseline_collection_time)
        baselines.append(benchmarking.results())

        with benchmarking.collect():
            function()
        results.append(benchmarking.results())

    # Sleep a bit to make sure measurements are cleared
    time.sleep(baseline_collection_time)

    baseline_cpu = np.mean([baseline.cpu for baseline in baselines])
    baseline_gpu = np.mean([baseline.gpu for baseline in baselines])
    baseline_ane = np.mean([baseline.ane for baseline in baselines])

    clock = [result.time for result in results]
    adjusted_cpu = [result.cpu - baseline_cpu for result in results]
    adjusted_gpu = [result.gpu - baseline_gpu for result in results]
    adjusted_ane = [result.ane - baseline_ane for result in results]

    return MeasurementResult(
        name=name,
        time=MeasurementStats(mean=np.mean(clock), std=np.std(clock)),
        cpu=MeasurementStats(mean=np.mean(adjusted_cpu), std=np.std(adjusted_cpu)),
        gpu=MeasurementStats(mean=np.mean(adjusted_gpu), std=np.std(adjusted_gpu)),
        ane=MeasurementStats(mean=np.mean(adjusted_ane), std=np.std(adjusted_ane)),
    )

if __name__ == "__main__":
    benchmarking = Benchmarking()
    with benchmarking.collect():
        time.sleep(5)

    print(benchmarking.results())

    benchmarking.done_benchmarking()

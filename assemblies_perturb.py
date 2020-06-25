from simulator.params import ParamDict, SimResults
from simulator.runner import run_handler
from pathlib import Path


@run_handler
def run_simulation(sim_params: ParamDict, _taskdir: Path) -> SimResults:
    return dict()

from typing import Callable, List, Dict, Any, AnyStr
from pathlib import Path
import sys
from .params import ParamDict, SimResults

SimulationRunner = Callable[[ParamDict, Path, List[Any], Dict[AnyStr, Any]], SimResults]

def run_handler(run_task: SimulationRunner):
    def runner(sim_params: ParamDict, task_dir: Path, dir_suffix: str, *args, **kwargs):
        import traceback
        import logging
        import time
        from .data import open_data_file, DataHandler

        logger = logging.getLogger(__name__)

        task_dir /= f"{time.strftime('%Y-%m-%d-%Hh%Mm%Ss')}{dir_suffix}"
        if not task_dir.exists():
            task_dir.mkdir(parents=True, exist_ok=True)

        result = dict()
        with TeeNoFile("stdout", result), TeeNoFile("stderr", result):
            try:
                result.update(run_task(sim_params, task_dir, *args, **kwargs))
            except Exception as e:
                logger.exception("An exception was thrown during the simulation.")
                error_msg = repr(e) + "\n"
                extracted_list = traceback.extract_tb(e.__traceback__)
                for item in traceback.StackSummary.from_list(extracted_list).format():
                    error_msg += str(item)
                result.update(error_msg=error_msg)

        f: DataHandler
        with open_data_file(task_dir / "results.h5", "w") as f:
            f.store_data_root(result)

        return result

    return runner

class TeeNoFile(object):
    """
    To be used in place of redirect_stdout/stderr from contextlib, because
    it isn't possible to capture stdout/stderr and still have it print to console.

    Modified from https://stackoverflow.com/a/616686
    """
    def __init__(self, std: str, results: SimResults):
        self.lines = []
        self.name = std
        self.std = None
        self.results = results

    def __enter__(self):
        self.std = getattr(sys, self.name)
        setattr(sys, self.name, self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.results[self.name] = self.process()

    def process(self) -> List[str]:
        if not len(self.lines):
            self.lines.append(" \n")
        return ''.join(self.lines).splitlines(keepends=False)

    def close(self):
        if self.std is not None:
            setattr(sys, self.name, self.std)
            self.std = None

    def write(self, data):
        self.lines.append(data)
        self.std.write(data)

    def flush(self):
        self.std.flush()

    def __del__(self):
        self.close()
from typing import Callable, List, Dict, Any, AnyStr
from pathlib import Path
import sys
from .params import ParamDict, SimResults

SimulationRunner = Callable[[ParamDict, Path, List[Any], Dict[AnyStr, Any]], SimResults]


def run_handler(run_task: SimulationRunner):
    def runner(sim_params: ParamDict, current_results_dir: Path, file_name_suffix: str, *args, **kwargs):
        import traceback
        import logging
        from typing import IO
        from .data import open_data_file, DataHandler
        from .params import SimParams

        logger = logging.getLogger(__name__)

        if not current_results_dir.exists():
            current_results_dir.mkdir(parents=True, exist_ok=True)

        results = dict()
        with TeeNoFile("stdout", results), TeeNoFile("stderr", results):
            try:
                results.update(run_task(sim_params, current_results_dir, *args, **kwargs))
            except Exception as e:
                logger.exception("An exception was thrown during the simulation.")
                error_msg = repr(e) + "\n"
                extracted_list = traceback.extract_tb(e.__traceback__)
                for item in traceback.StackSummary.from_list(extracted_list).format():
                    error_msg += str(item)
                results.update(error_msg=error_msg)

        df: DataHandler
        with open_data_file(current_results_dir / f"results~{file_name_suffix}.h5", "w") as df:
            df.store_data_root(results)

        pf: IO[str]
        with open(f"{current_results_dir / f'params~{file_name_suffix}.py'!s}", "w") as pf:
            pf.write(f"params = {{\n{SimParams.to_string(sim_params)}}}")

        return results

    return runner


def get_curr_results_dir(results_dir: Path, dir_suffix: str) -> Path:
    import time
    start_time = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
    return results_dir / (start_time+dir_suffix)


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
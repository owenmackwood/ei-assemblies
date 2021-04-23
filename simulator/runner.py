from typing import Callable, List, Dict, Any, AnyStr
from pathlib import Path
import sys
from .params import ParamDict, SimResults

SimulationRunner = Callable[[ParamDict, Path, List[Any], Dict[AnyStr, Any]], SimResults]


def run_handler(run_task: SimulationRunner):
    def runner(
            sim_params: ParamDict,
            current_results_path: Path,
            file_name_suffix: str,
            resume_path: Path = None,
            *args, **kwargs
    ):
        import traceback
        import logging
        from typing import IO
        from .data import open_data_file, DataHandler
        from .params import SimParams

        logger = logging.getLogger(__name__)

        if not current_results_path.exists():
            current_results_path.mkdir(parents=True, exist_ok=True)

        results_file_name = f"results~{file_name_suffix}.h5"

        if resume_path is not None:
            handler: DataHandler
            with open_data_file(resume_path / results_file_name, mode="r") as handler:
                resume_data = handler.read_data_root()
                kwargs["resume_data"] = resume_data

        results = dict()
        with TeeNoFile("stdout", results), TeeNoFile("stderr", results):
            try:
                results.update(run_task(sim_params, current_results_path, *args, **kwargs))
            except Exception as e:
                logger.exception("An exception was thrown during the simulation.")
                error_msg = repr(e) + "\n"
                extracted_list = traceback.extract_tb(e.__traceback__)
                for item in traceback.StackSummary.from_list(extracted_list).format():
                    error_msg += str(item)
                results.update(error_msg=error_msg)

        df: DataHandler
        with open_data_file(current_results_path / results_file_name, "w") as df:
            df.store_data_root(results)

        pf: IO[str]
        with open(f"{current_results_path / f'params~{file_name_suffix}.py'!s}", "w") as pf:
            pf.write(f"params = {{\n{SimParams.to_string(sim_params)}}}")

        return results

    return runner


def get_curr_results_path(results_dir: Path, dir_suffix: str) -> Path:
    import time
    start_time = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
    return results_dir / (start_time+dir_suffix)


def user_select_from_list(items, prompt):
    if len(items) > 1:
        for i, item in enumerate(items, 1):
            print(f'{i}: {item}')
        inp = 0
        while inp not in range(1, len(items)+1):
            try:
                inp = int(input(f'{prompt} (1-{len(items)}): '))
            except ValueError:
                print('Invalid selection')
    else:
        inp = 1
    return inp - 1


def user_select_resume_path(get_from_dir) -> Path:
    from operator import itemgetter
    all_exps = []

    for subdir in Path(get_from_dir).iterdir():
        if subdir.is_dir():
            if any(sdp.is_file() and sdp.suffix == ".h5"  for sdp in subdir.iterdir()):
                all_exps.append((subdir, subdir.stem))

    all_exps = sorted(all_exps, key=itemgetter(1))
    items = [subdir for ffp, subdir in all_exps]
    inp = user_select_from_list(items, 'Select an experiment')
    path, subdir = all_exps[inp]
    return path


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
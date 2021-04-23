# Learning excitatory-inhibitory neuronal assemblies in recurrent networks

Here you will find Python source code for a simulator that can generate the results used for our eLife
paper _Learning excitatory-inhibitory neuronal assemblies in recurrent networks_ (Owen Mackwood, Laura B. Naumann, 
Henning Sprekeler). The preprint is available [here](https://doi.org/10.1101/2020.03.30.016352), and a link
to the final eLife version will be added as soon as it's available.

Questions about this simulator should be directed towards Owen Mackwood. More general questions about the paper should
be sent to the corresponding author, Henning Sprekeler.

## How to use the simulator

To use this simulator, download or clone this repository. On the command line, go to the root directory of this 
repository and run `python main.py`. This will generate the data used for Figures 1-3 of the paper (and many of
the figure supplements). To generate the data for Figure 4, open `main.py` in your preferred editor, and change
the value of `RUN_EXP` to `ExperimentType.PERTURB`. It will prompt you to select the network you trained in the
previous step (which by default is stored in your home directory at `experiments/ei-assemblies`). Likewise, to
simulate networks with the full gradient rules, change `RUN_EXP` to `ExperimentType.GRADIENT`. 

## Data storage format

All results are stored using a standard HDF file format. There are several applications that can be used to open
these files, if you would like to inspect their contents. Otherwise, you can use the `DataHandler` class in 
the `simulator.data` module to extract data from the HDF file. A context manager is included in the same module
for your convenience and is used as follows:

```python
from simulator.data import open_data_file, DataHandler
from pathlib import Path

file_path = Path.home() / "experiments" / "ei-assemblies" / "<subdir>" / "<filename>.h5"
handler: DataHandler
with open_data_file(file_path) as handler:
    data = handler.read_data_root()
```

The `data` variable now refers to a dictionary that contains the final state of the network (`data["sim_state"]`), 
various computed quantities (`data["computed"]`), and some raw diagnostic values (`data["raw_data"]`) including firing 
rates of neurons and (when appropriate) the angle between the approximate-rule updates and the full gradient update.

## Requirements

We recommend that you use the Anaconda distribution of Python, which includes most of the packages necessary
to run this simulator. If you prefer to use another distribution, see the `requirements.txt` file for required packages
that are not part of the standard Python library.

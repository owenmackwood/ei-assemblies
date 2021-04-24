DISABLE_NUMBA = False
# Disabling Numba is useful for debugging purposes
if DISABLE_NUMBA:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'

from simulator.params import ExperimentType, PlasticityTypeItoE, PlasticityTypeEtoI, IsPlastic
from simulator.params import AllParameters, experiment_type_from_str
from pathlib import Path
from typing import Tuple


def prepare_params(
    n_trials: int,
    e2i_plasticity: PlasticityTypeEtoI,
    i2e_plasticity: PlasticityTypeItoE,
    eta_e: float,
    eta_i: float,
    compute_gradient_angles: bool,
) -> AllParameters:
    from simulator.params import NumericalIntegration, NeuronGroups
    from simulator.params import Synapses, x2y, Plasticity
    from simulator.params import Population, PopulationInput, Inputs

    numint = NumericalIntegration(
        dt=1e-3,
        max_dr=0.5,
        max_steps=1500,
        do_abort=False,
        n_trials=n_trials,
        every_n=n_trials // 1,  # Estimate the network response every_n trials. Must divide evenly into n_trials
    )

    ng = NeuronGroups(
        n_d=3,
        r_max=1e3,
        exc=Population(tau_m=50e-3, n_per_axis=8),
        inh=Population(tau_m=25e-3, n_per_axis=-1),  # Inh pop size is always n_i == n_e // 8
    )

    sy = Synapses(
        w_dist="lognorm",
        lognorm_sigma=0.65,
        e2e=x2y(w_total=2.0, p=0.6),
        e2i=x2y(w_total=5.0, p=0.6),
        i2i=x2y(w_total=1.0, p=0.6),
        i2e=x2y(w_total=1.0, p=0.6),
    )

    pl = Plasticity(
        rho0=1.0,
        is_plastic=IsPlastic.BOTH,
        bp_weights=True,
        eta_e=eta_e,
        eta_i=eta_i,
        w_max=10**5,
        soft_max=False,
        wei_decay=0.1,
        wie_decay=0.1,
        convergence_max=0,
        convergence_mean=0,
        plasticity_type_ei=i2e_plasticity, plasticity_type_ie=e2i_plasticity,
        compute_gradient_angles=compute_gradient_angles,
    )

    inp = Inputs(
        n_stimuli=12,
        sharp_input=False,
        regular_pref=True,
        exc=PopulationInput(bg_input=5.0, peak_stimulus=50.0),
        inh=PopulationInput(bg_input=5.0),
        vonmises_kappa=1.0,
        sharp_wf=0.76,
    )
    return AllParameters(numint=numint, ng=ng, sy=sy, pl=pl, inp=inp)


def run(run_exp: ExperimentType, result_path: Path, n_trials: int) -> None:
    from itertools import product
    from simulator.runner import get_curr_results_path, user_select_resume_path
    from assemblies_learn import run_simulation as learn
    from assemblies_perturb import run_simulation as perturb
    run_task = perturb if run_exp == ExperimentType.PERTURB else learn

    trials_str = "" if run_exp == ExperimentType.PERTURB else f" Training for {n_trials} trials."
    print(f"Running {run_exp}.{trials_str} Results will be stored in {result_path}")

    vary_pl_type = True
    vary_n_neurons = False
    resume_path = None
    dir_suffix = f"_{n_trials}trials"
    if run_exp == ExperimentType.GRADIENT:
        dir_suffix += "_gradients"
        e2i_plasticity = PlasticityTypeEtoI.GRADIENT
        i2e_plasticity = PlasticityTypeItoE.GRADIENT
        compute_gradient_angles = False
        eta_e = 1e-3
        eta_i = 1e-3
    elif run_exp == ExperimentType.APPROXIMATE:
        dir_suffix += "_approx"
        e2i_plasticity = PlasticityTypeEtoI.APPROXIMATE
        i2e_plasticity = PlasticityTypeItoE.APPROXIMATE
        compute_gradient_angles = True
        eta_e = 1e-5
        eta_i = 1e-5
    else:
        dir_suffix += "_perturb"
        assert run_exp == ExperimentType.PERTURB
        compute_gradient_angles = False
        e2i_plasticity = PlasticityTypeEtoI.APPROXIMATE
        i2e_plasticity = PlasticityTypeItoE.APPROXIMATE
        eta_e = 0e0
        eta_i = 0e0
        print(
            "To run a perturbation experiment, you must select a pre-trained network:"
        )
        resume_path = user_select_resume_path(result_path)

    curr_results_path = get_curr_results_path(result_path, dir_suffix)

    with prepare_params(
        n_trials,
        e2i_plasticity,
        i2e_plasticity,
        eta_e,
        eta_i,
        compute_gradient_angles,
    ) as params:
        all_ranges = []

        if vary_pl_type:
            all_ranges.append(((params.pl.is_plastic, is_plastic) for is_plastic in IsPlastic))

        if vary_n_neurons:
            all_ranges.append(((params.ng.exc.n_per_axis, n) for n in [4, 6, 8]))

        if all_ranges:
            for param_ranges in product(*all_ranges):
                param_str = " ".join([f"{params.path_string_repr(param)}={value}" for param, value in param_ranges])
                print(f"Starting simulation for {param_str}")
                temp_params = params.dict()
                string_reprs = [params.modify_value(temp_params, param, value) for param, value in param_ranges]
                file_name_suffix = "~".join(string_reprs)
                run_task(temp_params, curr_results_path, file_name_suffix, resume_path)
        else:
            run_task(params.dict(), curr_results_path, "", resume_path)


def parse_arguments() -> Tuple[ExperimentType, Path, int]:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--experiment",
        help="The experiment to run. APPROXIMATE must be run before PERTURB.",
        choices=[e.name for e in ExperimentType],
        type=str,
        default=ExperimentType.APPROXIMATE.name
    )
    parser.add_argument(
        "-p", "--path",
        help="Path to the results directory.",
        type=Path,
        default=Path.home() / "experiments" / "ei-assemblies"
    )
    parser.add_argument(
        "-n", "--numtrials",
        help="Number to trials for training.",
        type=int,
        default=500
    )
    args = parser.parse_args()

    run_exp = experiment_type_from_str(args.experiment)
    result_path = args.path
    n_trials = int(args.numtrials)
    return run_exp, result_path, n_trials


if __name__ == "__main__":
    run(*parse_arguments())

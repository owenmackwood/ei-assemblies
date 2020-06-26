DISABLE_NUMBA = False
if DISABLE_NUMBA:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'

from simulator.params import ExperimentType, PlasticityTypeItoE, PlasticityTypeEtoI, IsPlastic
from simulator.params import AllParameters
from pathlib import Path

RUN_EXP = ExperimentType.APPROXIMATE

if RUN_EXP == ExperimentType.GRADIENT:
    E2I_PLASTICITY = PlasticityTypeEtoI.GRADIENT
    I2E_PLASTICITY = PlasticityTypeItoE.GRADIENT
    COMPUTE_GRADIENT_ANGLES = False
    ETA_E = 1e-3
    ETA_I = 1e-3
elif RUN_EXP == ExperimentType.APPROXIMATE:
    E2I_PLASTICITY = PlasticityTypeEtoI.APPROXIMATE
    I2E_PLASTICITY = PlasticityTypeItoE.APPROXIMATE
    COMPUTE_GRADIENT_ANGLES = True
    ETA_E = 1e-5
    ETA_I = 1e-5
else:
    assert RUN_EXP == ExperimentType.PERTURB
    COMPUTE_GRADIENT_ANGLES = False
    ETA_E = 0e0
    ETA_I = 0e0
    raise NotImplementedError("No implementation of this experiment yet.")


N_TRIALS = 500
EVERY_N = N_TRIALS


def prepare_params() -> AllParameters:
    from simulator.params import NumericalIntegration, NeuronGroups
    from simulator.params import Synapses, x2y, Plasticity
    from simulator.params import Population, PopulationInput, Inputs

    numint = NumericalIntegration(
        dt=1e-3,
        max_dr=0.5,
        max_steps=1500,
        do_abort=False,
        n_trials=N_TRIALS,
        every_n=EVERY_N,
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
        eta_e=ETA_E,
        eta_i=ETA_I,
        w_max=10**5,
        soft_max=False,
        wei_decay=0.1,
        wie_decay=0.1,
        convergence_max=0,
        convergence_mean=0,
        plasticity_type_ei=I2E_PLASTICITY, plasticity_type_ie=E2I_PLASTICITY,
        compute_gradient_angles=COMPUTE_GRADIENT_ANGLES,
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


def run(result_path: Path, dir_suffix: str) -> None:
    from itertools import product
    from simulator.runner import get_curr_results_dir
    from assemblies_learn import run_simulation as learn
    from assemblies_perturb import run_simulation as perturb
    run_task = perturb if RUN_EXP == ExperimentType.PERTURB else learn

    vary_pl_type = True
    vary_n_neurons = False

    curr_results_dir = get_curr_results_dir(result_path, dir_suffix)

    with prepare_params() as params:
        all_ranges = []

        if vary_pl_type:
            all_ranges.append(((params.pl.is_plastic, is_plastic) for is_plastic in IsPlastic))

        if vary_n_neurons:
            all_ranges.append(((params.ng.exc.n_per_axis, n) for n in [4, ]))  # 6, 8

        if all_ranges:
            for param_ranges in product(*all_ranges):
                temp_params = params.dict()
                string_reprs = [params.modify_value(temp_params, param, value) for param, value in param_ranges]
                file_name_suffix = "~".join(string_reprs)
                run_task(temp_params, curr_results_dir, file_name_suffix)
        else:
            run_task(params.dict(), curr_results_dir, "")


if __name__ == "__main__":

    dir_suffix = f"_{N_TRIALS}trials"
    if RUN_EXP == ExperimentType.GRADIENT:
        dir_suffix += "_gradients"
    elif RUN_EXP == ExperimentType.APPROXIMATE:
        dir_suffix += "_approx"
    else:
        dir_suffix += "_perturb"

    run(
        result_path=Path.home() / "experiments" / "ei-test",
        dir_suffix=dir_suffix,
    )

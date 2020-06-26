DISABLE_NUMBA = False
if DISABLE_NUMBA:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'

from simulator.params import ExperimentType, PlasticityTypeItoE, PlasticityTypeEtoI, IsPlastic
from simulator.params import AllParameters
from pathlib import Path

N_TRIALS = 2
EVERY_N = N_TRIALS

DEFAULT_PLASTIC = IsPlastic.NEITHER
VARY_PL_TYPE = True
VARY_N_NEURONS = False

RUN_EXP = ExperimentType.APPROXIMATE

if RUN_EXP == ExperimentType.GRADIENT:
    E2I_PLASTICITY = PlasticityTypeEtoI.GRADIENT
    I2E_PLASTICITY = PlasticityTypeItoE.GRADIENT
    COMPUTE_GRADIENT_ANGLES = False
    ETA_E = 1e-3
    ETA_I = 1e-3
else:
    E2I_PLASTICITY = PlasticityTypeEtoI.APPROXIMATE
    I2E_PLASTICITY = PlasticityTypeItoE.APPROXIMATE
    COMPUTE_GRADIENT_ANGLES = True
    """
    Looks like 1e-4 is too fast for eta_i when e-to-e connections are present
    needs to be 1e-5 for the inh synapses to learn in Pyr-to-PV plasticity
    is knocked out.
    """
    ETA_E = 1e-5
    ETA_I = 1e-5


class ExcInhAssemblies:

    @staticmethod
    def _prepare_params() -> AllParameters:
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


    @staticmethod
    def run(result_path: Path, dir_suffix: str):
        from itertools import product
        from assemblies_learn import run_simulation as learn
        from assemblies_perturb import run_simulation as perturb
        run_task = perturb if RUN_EXP == ExperimentType.PERTURB else learn

        with ExcInhAssemblies._prepare_params() as params:
            all_ranges = []

            if VARY_PL_TYPE:
                all_ranges.append(((params.pl.is_plastic, is_plastic) for is_plastic in IsPlastic))

            if VARY_N_NEURONS:
                all_ranges.append(((params.ng.exc.n_per_axis, n) for n in [2, 4, 8]))

            if all_ranges:
                for param_ranges in product(*all_ranges):
                    temp_params = params.dict()
                    for param, value in param_ranges:
                        dir_suffix += params.modify_value(
                            temp_params, param, value
                        )
                    run_task(temp_params, result_path, dir_suffix)
            else:
                run_task(params.dict(), result_path, dir_suffix)


if __name__ == "__main__":
    import os

    dir_suffix = f"_{N_TRIALS}trials"
    if RUN_EXP == ExperimentType.GRADIENT:
        dir_suffix += "_gradients"
    elif RUN_EXP == ExperimentType.APPROXIMATE:
        dir_suffix += "_approx"
    else:
        dir_suffix += "_perturb"

    job_info = ExcInhAssemblies.run(
        Path.home() / "experiments" / "ei-test",
        dir_suffix,
    )

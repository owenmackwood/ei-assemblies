import numpy as np
from scipy.stats import sem
from simulator.params import ParamDict, SimResults
from simulator.runner import run_handler
from pathlib import Path

N_PERTURBED = 90
P_MIN = 10.
P_MAX = 10.


@run_handler
def run_simulation(sim_params: ParamDict, _current_results_dir: Path, resume_data: SimResults) -> SimResults:
    from itertools import product
    from simulator.params import VectorE, VectorI, ArrayIE, ArrayEE, ArrayEI, ArrayII
    from simulator.params import AllParameters
    from simulator.utils import allocate_aligned
    from simulator.setup import make_afferents
    from simulator.rates import compute_steady_state

    dtype = AllParameters.float_type
    perturbations = np.linspace(P_MIN, P_MAX, 1, dtype=dtype, endpoint=True)

    params = AllParameters(**sim_params)
    inp = params.inp
    ni = params.numint
    ng = params.ng
    pl = params.pl

    computed = resume_data["computed"]
    resume_state = resume_data["sim_state"]

    responses_exc = computed["responses_exc"]
    response_sim_ee = computed["response_sim_ee"]
    correlations_ee = computed["correlations_ee"][..., -1]

    max_dr = dtype(.05)
    max_steps = ni.max_steps
    n_e = ng.exc.n_per_axis ** ng.n_d
    n_i = n_e // 8
    dt_tau_e = ni.dt / ng.exc.tau_m
    dt_tau_i = ni.dt / ng.inh.tau_m
    max_dr_dt_exc = dt_tau_e * max_dr
    max_dr_dt_inh = dt_tau_i * max_dr

    wee = ArrayEE(resume_state["wee"])
    wei = ArrayEI(resume_state["wei"])
    wii = ArrayII(resume_state["wii"])
    wie = ArrayIE(resume_state["wie"])

    recording_re = allocate_aligned((n_e, ni.max_steps), dtype=dtype)
    recording_ri = allocate_aligned((n_i, ni.max_steps), dtype=dtype)

    aff_arrays = make_afferents(ng.n_d, ng.exc.n_per_axis, inp.n_stimuli, inp.exc.bg_input,
                                inp.exc.peak_stimulus, inp.vonmises_kappa)

    r_e = VectorE(allocate_aligned(n_e, dtype=dtype))
    r_p = VectorE(allocate_aligned(n_e, dtype=dtype))
    r_i = VectorI(allocate_aligned(n_i, dtype=dtype))

    all_perturbed = np.random.choice(n_e, N_PERTURBED, replace=False)
    n_perturbed = all_perturbed.size

    shape = (n_perturbed, perturbations.size) + aff_arrays.afferents.shape
    rate_change = np.zeros(shape, dtype=dtype)
    all_corr = np.zeros_like(rate_change)
    responses_exc.fill(np.NaN)
    for n in range(n_perturbed):
        print(f"Perturbation {n+1} of {n_perturbed}.")
        for i, j, k in product(*aff_arrays.locations_idx):
            r_e.fill(pl.rho0)
            r_i.fill(inp.inh.bg_input)
            converged, t, _, _ = compute_steady_state(
                r_e, r_i, wee, wei, wie, wii, aff_arrays.afferents[i, j, k, :], inp.inh.bg_input,
                dt_tau_e, dt_tau_i, ng.r_max, max_dr_dt_exc, max_dr_dt_inh,
                max_steps, recording_re, recording_ri
            )
            if not converged:
                print(f"Failed to converge at {(i, j, k)}")
            responses_exc[i, j, k, :] = r_e
            perturb_n = all_perturbed[n]

            for p in range(perturbations.size):
                r_p.fill(pl.rho0)
                r_i.fill(inp.inh.bg_input)

                perturbed = aff_arrays.afferents[i, j, k, :].copy()
                perturbed[perturb_n] += perturbations[p]
                recording_re.fill(np.NaN)
                recording_ri.fill(np.NaN)
                converged, t, _, _ = compute_steady_state(
                    r_p, r_i, wee, wei, wie, wii, perturbed, inp.inh.bg_input,
                    dt_tau_e, dt_tau_i, ng.r_max, max_dr_dt_exc, max_dr_dt_inh,
                    max_steps, recording_re, recording_ri
                )
                if not converged:
                    print(f"Failed to converge at {(i, j, k)} for perturbation {perturbations[p]}")

                r = (r_p - r_e) / perturbations[p]
                c = correlations_ee[perturb_n, :]
                rate_change[n, p, i, j, k, :] = r
                all_corr[n, p, i, j, k, :] = c

    binned_corr_mu, binned_corr_sem, corr_bins, orig_bins = bin_perturbations(all_corr, rate_change)

    raw_data = dict(
        rate_change=rate_change,
        perturbed=all_perturbed,
        all_correlations=all_corr,
    )
    computed = dict(
        responses_exc=responses_exc,
        response_sim_ee=response_sim_ee,
        correlations_ee=correlations_ee,
        correlation_bins=corr_bins,
        correlation_mean=binned_corr_mu,
        correlation_sem=binned_corr_sem,
        original_bins=orig_bins,
    )
    sim_state = dict(wie=wie, wei=wei, wii=wii, wee=wee)
    results = dict(raw_data=raw_data, computed=computed, sim_state=sim_state)

    return results


def bin_perturbations(all_corr: np.ndarray, rate_change: np.ndarray, n_bins=10):
    all_corr = all_corr.flatten()
    rate_change = rate_change.flatten()

    if isinstance(n_bins, np.ndarray):
        orig_bins = n_bins
    else:
        orig_bins = np.linspace(np.min(all_corr) - 1e-3, np.max(all_corr) + 1e-3, n_bins)
    corr_hist, bins = np.histogram(all_corr, bins=orig_bins)
    assert (bins == orig_bins).all()
    print(corr_hist)
    binned_corr_mu = np.empty(corr_hist.size)
    binned_corr_sem = np.empty(corr_hist.size)
    idx = np.digitize(all_corr, bins=bins)
    for i in range(1, bins.size):
        in_bin = idx == i
        size_bin = np.sum(in_bin)
        dr = rate_change[in_bin]
        binned_corr_mu[i-1] = np.mean(dr)
        binned_corr_sem[i-1] = sem(dr)
        print(f"{size_bin} values in bin {i}")

    corr_bins = np.diff(bins)/2 + bins[:-1]
    print(f"Bins: {corr_bins}")
    print(f"Mean: {binned_corr_mu}")
    print(f"SEM: {binned_corr_sem}")
    return binned_corr_mu, binned_corr_sem, corr_bins, orig_bins

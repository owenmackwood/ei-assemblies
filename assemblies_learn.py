DISABLE_NUMBA = False
if DISABLE_NUMBA:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'

from matplotlib.gridspec import GridSpec
from simulator.params import ExperimentType, PlasticityTypeItoE, PlasticityTypeEtoI, IsPlastic
from simulator.params import AllParameters

N_TRIALS = 500
EVERY_N = N_TRIALS

DEFAULT_PLASTIC = IsPlastic.BOTH
VARY_PL_TYPE = 1

RUN_EXP = ExperimentType.APPROXIMATE

if RUN_EXP == ExperimentType.GRADIENT:
    E2I_PLASTICITY = PlasticityTypeEtoI.GRADIENT
    I2E_PLASTICITY = PlasticityTypeItoE.GRADIENT
    COMPUTE_GRADIENT_ANGLES = False
    ETA_E = 1e-3
    ETA_I = 1e-3
else:
    E2I_PLASTICITY = PlasticityTypeEtoI.ANTEROGRADE
    I2E_PLASTICITY = PlasticityTypeItoE.APPROXIMATE
    COMPUTE_GRADIENT_ANGLES = True
    """
    Looks like 1e-4 is too fast for eta_i when e-to-e connections are present
    needs to be 1e-5 for the inh synapses to learn in Pyr-to-PV plasticity
    is knocked out.
    """
    ETA_E = 1e-5
    ETA_I = 1e-5

PLOT_AFFERENTS = False
PLOT_WEIGHT_HIST = False
ADD_BACKGROUND_NOISE = False
LOCAL_AFFERENTS = True


def run_task(task_info: dict, _taskdir: str, _tempdir: str):
    import numpy as np
    from time import time
    from scipy.stats import spearmanr
    from numpy.linalg import norm
    from simulator.utils import allocate_aligned
    from simulator.rates import train_network, estimate_responses
    from simulator.setup import make_afferents, afferents_plot, make_synapses
    from simulator.params import AllParameters
    from simulator.params import VectorE, VectorI, ArrayIE, ArrayEE, ArrayEI
    from simulator.plasticity import MomentEstimate
    # import mkl
    # if on_cluster:
    #     mkl.set_num_threads(1)
    # else:
    #     mkl.set_num_threads(4)

    t0 = time()
    dtype = AllParameters.float_type

    params = AllParameters(**task_info)
    inp = params.inp
    ni = params.numint
    ng = params.ng
    pl = params.pl

    n_e = ng.exc.n_per_axis ** ng.n_d
    n_i = n_e // 8
    dt_tau_e = ni.dt / ng.exc.tau_m
    dt_tau_i = ni.dt / ng.inh.tau_m
    max_dr_dt_exc = dt_tau_e * ni.max_dr
    max_dr_dt_inh = dt_tau_i * ni.max_dr

    if pl.is_plastic == IsPlastic.INH:
        pl.eta_e = dtype(0)
    elif pl.is_plastic == IsPlastic.EXC:
        pl.eta_i = dtype(0)
    elif pl.is_plastic == IsPlastic.NEITHER:
        pl.eta_e = pl.eta_i = dtype(0)
        ni.n_trials = 0

    aff_arrays = make_afferents(ng.n_d, ng.exc.n_per_axis, inp.n_stimuli, inp.exc.bg_input,
                                inp.exc.peak_stimulus, inp.vonmises_kappa, PLOT_AFFERENTS)
    if params.sy.e2e.w_total > 0:
        if not LOCAL_AFFERENTS:
            for i in range(n_e):
                flat_afferents = aff_arrays.afferents[..., i].flatten()
                np.random.shuffle(flat_afferents)
                aff_arrays.afferents[..., i] = flat_afferents.reshape(aff_arrays.afferents.shape[:-1])
            if PLOT_AFFERENTS:
                afferents_plot(aff_arrays.stimulus_pref, aff_arrays.afferents)

        # if ADD_BACKGROUND_NOISE:
        #     aff_arrays.afferents += np.random.uniform(0, inp.exc.bg_input, size=aff_arrays.afferents.shape)

        # corr_kappa = inp.vonmises_kappa
        corr_kappa = inp.vonmises_kappa / 4
        tmp_arrays = make_afferents(ng.n_d, ng.exc.n_per_axis, inp.n_stimuli, inp.exc.bg_input,
                                    inp.exc.peak_stimulus, corr_kappa, PLOT_AFFERENTS)
        flattened = np.zeros((n_e, inp.n_stimuli ** ng.n_d), dtype=dtype)
        for n in range(n_e):
            flattened[n, :] = tmp_arrays.afferents[..., n].flatten()
        flattened = flattened.astype(dtype)
        target_correlations = np.corrcoef(flattened)
        target_correlations = ArrayEE(target_correlations.astype(dtype))
    else:
        target_correlations = None

    sya = make_synapses(params, n_e, n_i, target_correlations, plot_weight_hist=PLOT_WEIGHT_HIST)
    wei_init = sya.wei.copy()
    wie_init = sya.wie.copy()

    pop_in = aff_arrays.afferents.sum(axis=-1)
    print(f"Sharp: {inp.sharp_input} Avg in: {pop_in.mean():.1f}, std: {pop_in.std():.1f}")
    per_exc = aff_arrays.afferents.sum(axis=(0, 1, 2))
    print(f"Per neuron. Avg in: {per_exc.mean()}, std: {per_exc.std():.1f}")

    recording_re = allocate_aligned((n_e, ni.max_steps), dtype=dtype)
    recording_ri = allocate_aligned((n_i, ni.max_steps), dtype=dtype)

    m_batches = ni.n_trials // EVERY_N
    assert ni.n_trials % EVERY_N == 0

    if ni.n_trials:
        correlations_ee = np.empty((n_e, n_e, m_batches+1))
    else:
        correlations_ee = np.empty((n_e, n_e, 1))

    responses_exc, responses_inh, exc_in, inh_in = estimate_responses(
        inp.n_stimuli, aff_arrays.locations_idx, aff_arrays.afferents, inp.inh.bg_input,
        n_e, n_i, pl.rho0, sya,
        dt_tau_e, dt_tau_i,
        params.ng.r_max,
        max_dr_dt_exc, max_dr_dt_inh,
        ni.max_steps,
        recording_re, recording_ri
    )
    print(f"Maximum exc. rate {np.max(responses_exc):.2f}")
    flattened = np.zeros((n_e, inp.n_stimuli ** ng.n_d), dtype=dtype)
    for n in range(n_e):
        flattened[n, :] = responses_exc[..., n].flatten()
    corr = np.corrcoef(flattened)
    correlations_ee[..., 0] = corr

    if COMPUTE_GRADIENT_ANGLES and pl.eta_e > 0:
        angles_ie = allocate_aligned((inp.n_stimuli ** params.ng.n_d * ni.n_trials), np.NaN, dtype=dtype)
    else:
        angles_ie = None

    if COMPUTE_GRADIENT_ANGLES and pl.eta_i > 0:
        angles_ei = allocate_aligned((inp.n_stimuli**params.ng.n_d * ni.n_trials), np.NaN, dtype=dtype)
    else:
        angles_ei = None

    if pl.plasticity_type_ei == PlasticityTypeItoE.GRADIENT and ni.n_trials:
        adam_ei = MomentEstimate(
            ArrayEI(allocate_aligned(sya.wei.shape, dtype=sya.wei.dtype)),
            ArrayEI(allocate_aligned(sya.wei.shape, dtype=sya.wei.dtype)),
        )
    else:
        adam_ei = None
    if pl.plasticity_type_ie == PlasticityTypeEtoI.GRADIENT and ni.n_trials:
        adam_ie = MomentEstimate(
            ArrayIE(allocate_aligned(sya.wie.shape, dtype=sya.wie.dtype)),
            ArrayIE(allocate_aligned(sya.wie.shape, dtype=sya.wie.dtype)),
        )
    else:
        adam_ie = None

    converged = True
    if ni.n_trials:
        r_e = VectorE(allocate_aligned(n_e, dtype=dtype))
        r_i = VectorI(allocate_aligned(n_i, dtype=dtype))

        recording_mu = allocate_aligned((ni.n_trials * inp.n_stimuli**ng.n_d, 4, 2), np.NaN, dtype=dtype)
        all_t = allocate_aligned((ni.n_trials, inp.n_stimuli ** ng.n_d), -1, dtype=np.int32)
        all_di = allocate_aligned((1, 2), dtype=dtype)
        last_n = 10
        inh_in_buffer = allocate_aligned((inp.n_stimuli, inp.n_stimuli, inp.n_stimuli, n_e, last_n), dtype=dtype)

        for m in range(m_batches):
            print(f"Batch {m+1} of {m_batches}")
            mu_idx = EVERY_N * (inp.n_stimuli ** ng.n_d)

            if False:  # and not converged:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                gs = GridSpec(2, 2)
                ax = fig.add_subplot(gs[0, 0])
                ax.plot(recording_re.T)
                ax = fig.add_subplot(gs[0, 1])
                last_idx = np.argwhere(np.isfinite(recording_re[0, :]))[-1]
                ax.hist(np.squeeze(recording_re[:, last_idx]), bins=100)
                ax = fig.add_subplot(gs[1, 0])
                ax.plot(recording_ri.T)
                ax = fig.add_subplot(gs[1, 1])
                ax.hist(np.squeeze(recording_ri[:, last_idx]), bins=100)
                plt.show()

            converged, n_run, ni.max_steps, all_di = train_network(
                n_trials=EVERY_N, rho0=pl.rho0,
                re=r_e, ri=r_i, sya=sya,
                eta_e=pl.eta_e, eta_i=pl.eta_i,
                wie_decay=pl.wie_decay, wei_decay=pl.wei_decay,
                plasticity_type_ie=pl.plasticity_type_ie, plasticity_type_ei=pl.plasticity_type_ei,
                bp_weights=pl.bp_weights,
                afferents=aff_arrays.afferents,  bg_input_inh=inp.inh.bg_input,
                inh_in=inh_in_buffer, trial_t=all_t[m * EVERY_N: (m + 1) * EVERY_N, :],
                dt_tau_e=dt_tau_e, dt_tau_i=dt_tau_i,
                dt_bcm_tau_inv=ni.dt * pl.bcm.tau_inv,
                r_max=params.ng.r_max,
                max_dr_dt_exc=max_dr_dt_exc, max_dr_dt_inh=max_dr_dt_inh,
                convergence_max=pl.convergence_max, convergence_mean=pl.convergence_mean,
                x_locations=aff_arrays.locations_idx[0],
                y_locations=aff_arrays.locations_idx[1],
                z_locations=aff_arrays.locations_idx[2],
                rec_mu=recording_mu[m * mu_idx:(m + 1) * mu_idx, ...],
                rec_re=recording_re, rec_ri=recording_ri,
                max_steps=ni.max_steps,
                do_abort=ni.do_abort,
                increment_steps_on_non_convergence=0,
                bcm_theta=pl.bcm.theta,
                adam_ie=adam_ie, adam_ei=adam_ei,
                angles_ie=angles_ie[m * mu_idx:(m + 1) * mu_idx] if angles_ie is not None else None,
                angles_ei=angles_ei[m * mu_idx:(m + 1) * mu_idx] if angles_ei is not None else None,
                compute_angles=COMPUTE_GRADIENT_ANGLES,
            )

            responses_exc, responses_inh, exc_in, inh_in = estimate_responses(
                    inp.n_stimuli, aff_arrays.locations_idx,
                    aff_arrays.afferents, inp.inh.bg_input,
                    n_e, n_i, pl.rho0, sya,
                    dt_tau_e, dt_tau_i,
                    params.ng.r_max,
                    max_dr_dt_exc, max_dr_dt_inh,
                    ni.max_steps,
                    recording_re, recording_ri
            )
            flattened = np.zeros((n_e, inp.n_stimuli ** ng.n_d), dtype=dtype)
            for n in range(n_e):
                flattened[n, :] = responses_exc[..., n].flatten()
            corr = np.corrcoef(flattened)
            correlations_ee[..., m+1] = corr

    else:
        recording_mu = np.zeros((1, 4, 2), dtype=dtype)
        all_t = np.zeros((1, inp.n_stimuli ** ng.n_d), dtype=np.int32)
        all_di = np.zeros((1, 2), dtype=dtype)

    if not np.isfinite(sya.wie).all():
        print("wie had NaN or inf values")
    if not np.isfinite(sya.wei).all():
        print("wei had NaN or inf values")

    if False:  # and not converged:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        gs = GridSpec(2, 2)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(recording_re.T)
        ax = fig.add_subplot(gs[0, 1])
        m = np.argwhere(np.isfinite(recording_re[0, :]))[-1]
        ax.hist(np.squeeze(recording_re[:, m]), bins=100)
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(recording_ri.T)
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(np.squeeze(recording_ri[:, m]), bins=100)
        plt.show()

    osi_e = np.empty((n_e,), dtype=dtype)
    osi_i = np.empty((n_i,), dtype=dtype)
    compute_osi = 1
    if compute_osi:
        sin_d = np.sin(aff_arrays.locations)
        cos_d = np.cos(aff_arrays.locations)
        for response, osi in zip((responses_exc, responses_inh), (osi_e, osi_i)):
            for n in range(osi.size):
                max_ij = np.argmax(response[..., n].mean(axis=-1))
                max_ij = np.unravel_index(max_ij, (inp.n_stimuli, inp.n_stimuli))
                tcs = response[max_ij[0], max_ij[1], :, n]
                osi_l = np.sum(tcs * sin_d, axis=-1) ** 2
                osi_r = np.sum(tcs * cos_d, axis=-1) ** 2
                osi[n] = np.sqrt(osi_l + osi_r) / np.sum(tcs, axis=-1)

    cc = np.ones((n_e,))
    cp = np.ones((n_e,))
    for k in range(n_e):
        rho, p = spearmanr(exc_in[..., k].flatten(), inh_in[..., k].flatten())
        cc[k] = rho
        cp[k] = p if np.isfinite(p) else 0
    f_cells = 100 * np.nanmean(cp > 1e-3, axis=0)
    mean_cc = np.nanmean(cc)
    print(f"Avg correlation between synaptic currents: {mean_cc:.1f}")
    print(f"Percentage of cells without strong correlation: {f_cells:.1f}")

    response_sim_ee = np.zeros((n_e, n_e))
    response_sim = np.zeros((n_e, n_i))
    compute_similarity = 1
    if compute_similarity:
        n2_exc = np.empty(n_e)
        for i in range(n_e):
            eci = responses_exc[..., i].flatten()
            n2_exc[i] = norm(eci, 2)
            if np.isclose(n2_exc[i], 0.0):
                print(f"Exc. {i} had no response")

        n2_inh = np.empty(n_i)
        for j in range(n_i):
            icj = responses_inh[..., j].flatten()
            n2_inh[j] = norm(icj, 2)
            if np.isclose(n2_inh[j], 0.0):
                print(f"Inh. {j} had no response")

        for i in range(n_e):
            eci = responses_exc[..., i].flatten()
            for j in range(n_i):
                icj = responses_inh[..., j].flatten()
                rs = np.dot(eci, icj) / (n2_exc[i] * n2_inh[j])
                response_sim[i, j] = rs
            for j in range(n_e):
                ecj = responses_exc[..., j].flatten()
                response_sim_ee[i, j] = np.dot(eci, ecj) / (n2_exc[i] * n2_exc[j])

    plot_rs = True
    if plot_rs:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        gs = GridSpec(2, 2)
        response_sim_ee[np.diag_indices_from(response_sim_ee)] = 0

        ax = fig.add_subplot(gs[0, 0])
        ax.set_title("corr_ee")
        ax.imshow(correlations_ee[..., 0])
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[0, 1])
        ax.set_title("wee")
        ax.imshow(sya.wee)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[1, 0])
        ax.set_title("response_sim")
        ax.imshow(response_sim_ee)
        ax.set_xticks([])
        ax.set_yticks([])

        if params.sy.e2e.w_total > dtype(0):
            ax = fig.add_subplot(gs[1, 1])
            ax.set_title("wee - corr_ee")
            _wee = sya.wee - np.min(sya.wee)
            _wee /= _wee.max()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(_wee - correlations_ee[..., 0])
        plt.show()

    print("Finished computing responses")
    t1 = time()
    t_compute = (t1 - t0) / 60.0
    print(f"Compute time {t_compute:.1f} min.")

    raw_data = dict(
        converged=np.array([converged]),
        steps_to_converge=all_t,
        max_inh_syn_change=all_di,
        recording_re=recording_re,
        recording_ri=recording_ri,
        recording_mu=recording_mu,
        angles_ie=angles_ie if angles_ie is not None else np.zeros(1),
        angles_ei=angles_ei if angles_ei is not None else np.zeros(1),
    )
    computed = dict(
        responses_exc=responses_exc,
        responses_inh=responses_inh,
        response_sim=response_sim,
        response_sim_ee=response_sim_ee,
        cc=cc,
        cp=cp,
        exc_in=exc_in,
        inh_in=inh_in,
        osi_e=osi_e,
        osi_i=osi_i,
        correlations_ee=correlations_ee,
    )
    sim_state = dict(
        wee=sya.wee,
        wei=sya.wei,
        wie=sya.wie,
        wii=sya.wii,
        wei_init=wei_init,
        wie_init=wie_init,
        zei=sya.zei,
        zie=sya.zie,
        stimulus_pref=aff_arrays.stimulus_pref,
        afferents=aff_arrays.afferents,
        ei_min=np.array([sya.ei_min]),
        ie_min=np.array([sya.ie_min]),
    )
    results = dict(raw_data=raw_data, computed=computed, sim_state=sim_state)

    return results


class AssembliesLearn:
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
    def run(result_path, dir_suffix):
        from itertools import product

        with AssembliesLearn._prepare_params() as params:
            all_ranges = []
            if VARY_PL_TYPE:
                all_ranges.append(((params.pl.is_plastic, is_plastic) for is_plastic in IsPlastic))
                all_ranges.append(((params.ng.exc.n_per_axis, n) for n in [4, 8]))

            for param_ranges in product(*all_ranges):
                temp_params = params.dict()
                for param, value in param_ranges:
                    params.modify_value(
                        temp_params, param, value
                    )
                run_task(temp_params, '', '')


if __name__ == "__main__":
    import os

    dir_suffix = f"_{N_TRIALS}trials"
    if RUN_EXP == ExperimentType.GRADIENT:
        dir_suffix += "_gradients"
    else:
        dir_suffix += "_approx"

    job_info = AssembliesLearn.run(
        os.path.expanduser("~/experiments"),
        dir_suffix=dir_suffix,
    )

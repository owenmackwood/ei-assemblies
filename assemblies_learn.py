from simulator.params import ParamDict, SimResults
from simulator.runner import run_handler
from pathlib import Path

PLOT_AFFERENTS = False
PLOT_WEIGHT_HIST = False
PLOT_RESP_SIM = False
ADD_BACKGROUND_NOISE = False
LOCAL_AFFERENTS = True


@run_handler
def run_simulation(sim_params: ParamDict, _current_results_dir: Path) -> SimResults:
    from matplotlib.gridspec import GridSpec
    from simulator.params import PlasticityTypeItoE, PlasticityTypeEtoI, IsPlastic
    import numpy as np
    from time import time
    from simulator.utils import allocate_aligned
    from simulator.rates import train_network, estimate_responses
    from simulator.setup import make_afferents, afferents_plot, make_synapses
    from simulator.params import AllParameters
    from simulator.params import VectorE, VectorI, ArrayIE, ArrayEE, ArrayEI
    from simulator.plasticity import MomentEstimate
    from simulator.measure import orientation_selectivity_index, compute_syn_current_spearmanr, compute_response_similarity
    # import mkl
    # if on_cluster:
    #     mkl.set_num_threads(1)
    # else:
    #     mkl.set_num_threads(4)

    t0 = time()
    dtype = AllParameters.float_type

    params = AllParameters(**sim_params)
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

    m_batches = ni.n_trials // ni.every_n
    assert ni.n_trials % ni.every_n == 0

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

    if pl.compute_gradient_angles and pl.eta_e > 0:
        angles_ie = allocate_aligned((inp.n_stimuli ** params.ng.n_d * ni.n_trials), np.NaN, dtype=dtype)
    else:
        angles_ie = None

    if pl.compute_gradient_angles and pl.eta_i > 0:
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
            mu_idx = ni.every_n * (inp.n_stimuli ** ng.n_d)

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
                n_trials=ni.every_n, rho0=pl.rho0,
                re=r_e, ri=r_i, sya=sya,
                eta_e=pl.eta_e, eta_i=pl.eta_i,
                wie_decay=pl.wie_decay, wei_decay=pl.wei_decay,
                plasticity_type_ie=pl.plasticity_type_ie, plasticity_type_ei=pl.plasticity_type_ei,
                bp_weights=pl.bp_weights,
                afferents=aff_arrays.afferents,  bg_input_inh=inp.inh.bg_input,
                inh_in=inh_in_buffer, trial_t=all_t[m * ni.every_n: (m + 1) * ni.every_n, :],
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
                compute_angles=pl.compute_gradient_angles,
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

    osi_e, osi_i = orientation_selectivity_index(inp, responses_exc, responses_inh, aff_arrays)
    response_sim_ee = compute_response_similarity(responses_exc, responses_exc)
    response_sim = compute_response_similarity(responses_exc, responses_inh)
    cc, cp = compute_syn_current_spearmanr(exc_in, inh_in)
    print(f"Avg correlation between synaptic currents: {np.nanmean(cc):.1f}")
    print(f"Percentage of cells without strong correlation: {100 * np.nanmean(cp > 1e-3, axis=0):.1f}")

    if PLOT_RESP_SIM:
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
        converged=converged,
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
        ei_min=sya.ei_min,
        ie_min=sya.ie_min,
    )
    results: SimResults = dict(raw_data=raw_data, computed=computed, sim_state=sim_state)

    return results

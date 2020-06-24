import numpy as np
from numba import njit
from itertools import product
from typing import Union, List
from .utils import allocate_aligned
from typing import Tuple, Optional
from .params import dtype, numba_cache
from .params import VectorE, VectorI, ArrayIE, ArrayEE, ArrayEI, ArrayII
from .params import Afferents, PlasticityTypeItoE, PlasticityTypeEtoI
from .setup import SynapseArrays
from .plasticity import MomentEstimate, update_weights


@njit(cache=numba_cache)
def compute_steady_state(
    re: VectorE, ri: VectorI, wee: ArrayEE, wei: ArrayEI, wie: ArrayIE, wii: ArrayII, total_input,
    bg_input_inh: Union[dtype, VectorI],
    dt_tau_e, dt_tau_i, r_max, max_dr_dt_exc, max_dr_dt_inh,
    max_steps, rec_re, rec_ri
) -> Tuple[bool, int, VectorE, VectorI]:
    converged = False
    t = 0
    ge, gi = dtype(0), dtype(0)
    de, di = np.zeros_like(re), np.zeros_like(ri)
    he, hi = re.copy(), ri.copy()
    while not converged and t < max_steps:
        de[:] = dt_tau_e * (total_input - wei @ ri + wee @ re - re)
        di[:] = dt_tau_i * (wie @ re - wii @ ri - ri + bg_input_inh)

        he += de
        re[:] = he
        neg_e = re < 0
        pos_e = re > r_max
        re[neg_e] = 0
        re[pos_e] = r_max
        escape_e = np.logical_or(neg_e, pos_e)
        ge = (
            np.abs(de[np.logical_not(escape_e)]).max()
            if not np.all(escape_e)
            else dtype(0)
        )

        hi += di
        ri[:] = hi
        neg_i = ri < 0
        pos_i = ri > r_max
        ri[neg_i] = 0
        ri[pos_i] = r_max
        escape_i = np.logical_or(neg_i, pos_i)
        gi = (
            np.abs(di[np.logical_not(escape_i)]).max()
            if not np.all(escape_i)
            else dtype(0)
        )

        if t < rec_re.shape[1]:
            rec_re[:, t] = re
            rec_ri[:, t] = ri
        t += 1
        converged = ge < max_dr_dt_exc and gi < max_dr_dt_inh
    if not converged:
        print(
            "Max-> Steps:", max_steps, "Grad. Exc:", ge, "Inh:", gi,
        )

    return converged, t, he, hi



@njit(cache=numba_cache)
def train_network(
        n_trials: int,
        rho0: dtype,
        re: VectorE,
        ri: VectorI,
        sya: SynapseArrays,
        eta_e: dtype,
        eta_i: dtype,
        wie_decay: dtype,
        wei_decay: dtype,
        bp_weights: bool,
        plasticity_type_ie: PlasticityTypeEtoI,
        plasticity_type_ei: PlasticityTypeItoE,
        afferents: Afferents,
        bg_input_inh: Union[dtype, VectorI],
        inh_in: np.ndarray,
        trial_t: np.ndarray,
        dt_tau_e: dtype,
        dt_tau_i: dtype,
        dt_bcm_tau_inv: dtype,
        r_max: dtype,
        max_dr_dt_exc: dtype,
        max_dr_dt_inh: dtype,
        convergence_max: dtype,
        convergence_mean: dtype,
        x_locations: np.ndarray,
        y_locations: np.ndarray,
        z_locations: np.ndarray,
        rec_mu: np.ndarray,
        rec_re: np.ndarray,
        rec_ri: np.ndarray,
        max_steps: int,
        do_abort: bool,
        increment_steps_on_non_convergence: int,
        bcm_theta: dtype,
        adam_ie: Optional[MomentEstimate] = None,
        adam_ei: Optional[MomentEstimate] = None,
        angles_ie: Optional[np.ndarray] = None,
        angles_ei: Optional[np.ndarray] = None,
        compute_angles: bool = False,
):
    n_e = re.size
    max_steps = max_steps
    mu_i = n = 0
    min_max = min_mu = np.infty
    converged = False
    last_n = inh_in.shape[-1]
    # inh_in = np.zeros((n_stimuli, n_stimuli, n_stimuli, n_e, last_n), dtype=dtype)
    # trial_t = np.empty((n_trials, n_stimuli ** n_d), dtype=np.int32)
    # trial_t.fill(-1)
    all_di = allocate_aligned((n_trials - 1, 2), np.NaN, dtype=dtype)
    buff_idx = np.arange(last_n)
    r_i_slow = allocate_aligned(ri.shape, bcm_theta, ri.dtype)

    # ris_rec = np.empty((n_stimuli ** n_d, ri.size, 3))
    # re_rec = np.empty((n_stimuli ** n_d, re.size))

    for n in range(n_trials):
        print("Trial", n+1, "of", n_trials)
        t_i = 0
        not_converged = 0
        np.random.shuffle(x_locations)
        for i in x_locations:
            np.random.shuffle(y_locations)
            for j in y_locations:
                np.random.shuffle(z_locations)
                for k in z_locations:
                    re[:] = rho0  # np.random.uniform(0, 100., n_e).astype(dtype)
                    ri[:] = bg_input_inh  # np.random.uniform(0, 100., n_i).astype(dtype)
                    rec_re.fill(np.NaN)
                    rec_ri.fill(np.NaN)

                    converged, t, he, hi = compute_steady_state(
                        re, ri,
                        sya.wee, sya.wei, sya.wie, sya.wii,
                        afferents[i, j, k, :], bg_input_inh,
                        dt_tau_e, dt_tau_i, r_max,
                        max_dr_dt_exc, max_dr_dt_inh,
                        max_steps, rec_re, rec_ri
                    )
                    trial_t[n, t_i] = t
                    inh_in[i, j, k, :, buff_idx[0]] = sya.wei @ ri
                    r_i_slow += dt_bcm_tau_inv * (ri - r_i_slow)
                    if converged:
                        angle_ei, angle_ie = update_weights(
                            rho0, re, ri, he, hi,
                            sya.vie, sya.vei, sya.zei, sya.zie, sya.wie, sya.wei, sya.wee, sya.wii,
                            eta_e, eta_i, wie_decay, wei_decay, sya.alpha_ie, sya.alpha_ei,
                            sya.ie_max, sya.ei_max, bp_weights, sya.ei_min, sya.ie_total,
                            plasticity_type_ie, plasticity_type_ei, adam_ie, adam_ei, compute_angles)
                        if angles_ei is not None:
                            angles_ei[mu_i] = angle_ei
                        if angles_ie is not None:
                            angles_ie[mu_i] = angle_ie
                    elif do_abort:
                        print("Abort on no convergence", mu_i)
                        return False, n, max_steps, all_di
                    else:
                        max_steps += increment_steps_on_non_convergence
                        not_converged += 1
                    rec_mu[mu_i, ...] = np.array(
                        [
                            [re.mean(), re.var()],
                            [ri.mean(), ri.var()],
                            [sya.wei.mean(), sya.wei.var()],
                            [sya.wie.mean(), sya.wie.var()],
                        ]
                    )
                    t_i += 1
                    mu_i += 1
        print(
            "Steps:",
            np.min(trial_t[n, :]),
            "-",
            np.max(trial_t[n, :]),
            "( mu:",
            np.int(np.round(np.mean(trial_t[n, :]))),
            ")",
        )
        if not_converged > 0:
            print(not_converged, "of", trial_t.size, "failed to converge!")
        # if DO_DEBUG_PLOT:
        #     fig = plt.figure(figsize=(12, 9))
        #     ar = 4
        #     gs = GridSpec(5, 1)
        #     ax = fig.add_subplot(gs[0]); ax.set_xticks([]); ax.set_yticks([])
        #     ax.imshow(re_rec.T, aspect=re_rec.shape[1] / re_rec.shape[0])
        #     for p_i in range(3):
        #         sps = GridSpecFromSubplotSpec(1, 2, gs[p_i+1], width_ratios=(4, 1))
        #         ax = fig.add_subplot(sps[0]); ax.set_xticks([]); ax.set_yticks([])
        #         ax.imshow(ris_rec[:,:,p_i].T, aspect=ris_rec.shape[0] / ris_rec.shape[1] / ar)
        #         ax = fig.add_subplot(sps[1])
        #         bins = np.linspace(0, ris_rec[:, :, p_i].max()+.1, 100)
        #         for h_i in (0, -1):
        #             ax.hist(ris_rec[h_i,:,p_i], bins=bins, histtype='step')
        #     sps = GridSpecFromSubplotSpec(1, 2, gs[-1], width_ratios=(4, 1))
        #     ax = fig.add_subplot(sps[0]); ax.set_xticks([]); ax.set_yticks([])
        #     ris_diff = np.diff(ris_rec[:,:,2].T, axis=1)
        #     ax.imshow(ris_diff, aspect=ris_rec.shape[0] / ris_rec.shape[1] / ar)
        #     ax = fig.add_subplot(sps[1])
        #     bins = np.linspace(ris_diff.min() - (1e-3), ris_diff.max()+(1e-3), 100)
        #     ax.hist(ris_diff.flatten(), bins=bins, histtype='step')
        #     plt.show()

        if n > 0:
            di_max, di_mu = plasticity_converged(
                buff_idx, n, inh_in, x_locations, y_locations, z_locations, n_e,
            )
            # di_max, di_mu = plasticity_converged(
            #     buff_idx, n, inh_in, x_locations, y_locations, z_locations
            # )
            min_max, min_mu = min(di_max, min_max), min(di_mu, min_mu)
            print(
                "Input change, max:", di_max, "(", min_max, ")", "mean:", di_mu, "(", min_mu, ")",
            )
            all_di[n - 1, :] = di_max, di_mu
            if di_max <= convergence_max and di_mu <= convergence_mean:
                break
        buff_idx[:] = np.roll(buff_idx, 1)

    print("Finished training")
    return converged, n, max_steps, all_di


@njit(cache=numba_cache)
def plasticity_converged(
    buff_idx: np.ndarray, n: int, inh_in: np.ndarray,
    x_locations: np.ndarray, y_locations: np.ndarray, z_locations: np.ndarray, n_e: int,
) -> Tuple[dtype, dtype]:
    inh_diff = np.zeros((x_locations.size, y_locations.size, z_locations.size, n_e), dtype=dtype)
    inh_dup = np.zeros_like(inh_in)
    inh_dup[..., :] = inh_in[..., buff_idx]
    valid_idx = slice(
        None, n + 1 if (n + 1 < inh_in.shape[-1]) else inh_in.shape[-1]
    )
    for i in x_locations:
        for j in y_locations:
            for k in z_locations:
                for n in range(n_e):
                    inh_diff[i, j, k, n] = np.abs(
                        np.mean(np.diff(inh_dup[i, j, k, n, valid_idx]))
                    )
    # inh_mean = inh_in.mean(axis=-1)
    # di = inh_mean[..., None] - inh_in
    # inh_diff[...] = di.sum(axis=-1)
    di_max, di_mu = np.max(inh_diff), np.mean(inh_diff)
    return dtype(di_max), dtype(di_mu)


def estimate_responses(
        n_stimuli: int,
        locations_idx: List[np.ndarray],
        afferents: Afferents,
        bg_input_inh: Union[dtype, VectorI],
        n_e: int, n_i: int,
        rho0: dtype,
        sya: SynapseArrays,
        dt_tau_e: dtype, dt_tau_i: dtype,
        r_max: dtype,
        max_dr_dt_exc: dtype, max_dr_dt_inh: dtype,
        max_steps: int,
        recording_re: np.ndarray, recording_ri: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Estimating responses.")
    wee = sya.wee
    wei = sya.wei
    wie = sya.wie
    wii = sya.wii
    responses_exc = np.empty((n_stimuli, n_stimuli, n_stimuli, n_e), dtype=dtype)
    responses_inh = np.empty((n_stimuli, n_stimuli, n_stimuli, n_i), dtype=dtype)
    inh_in = np.zeros((n_stimuli, n_stimuli, n_stimuli, n_e), dtype=dtype)
    exc_in = np.zeros((n_stimuli, n_stimuli, n_stimuli, n_e), dtype=dtype)
    r_e = VectorE(allocate_aligned(n_e, dtype=dtype))
    r_i = VectorI(allocate_aligned(n_i, dtype=dtype))
    for i, j, k in product(*locations_idx):
        recording_re.fill(np.NaN)
        recording_ri.fill(np.NaN)
        r_e.fill(rho0)
        r_i[:] = bg_input_inh
        total_input = afferents[i, j, k, :]
        converged, t, _, _ = compute_steady_state(
                                            r_e, r_i,
                                            wee, wei,
                                            wie, wii,
                                            total_input,
                                            bg_input_inh,
                                            dt_tau_e, dt_tau_i,
                                            r_max,
                                            max_dr_dt_exc, max_dr_dt_inh,
                                            max_steps,
                                            recording_re, recording_ri
                                        )

        if not converged:
            print(f"Response failed to converge at {i}, {j}, {k}")
            # if not converged and not on_cluster:
            #     fig = plt.figure()
            #     gs = GridSpec(2, 1)
            #     ax = fig.add_subplot(gs[0])
            #     ax.plot(recording_re.T)
            #     ax = fig.add_subplot(gs[1])
            #     ax.plot(recording_ri.T)
            #     plt.show()

        responses_exc[i, j, k, :] = r_e
        responses_inh[i, j, k, :] = r_i
        inh_in[i, j, k, :] = wei @ r_i
        exc_in[i, j, k, :] = wee @ r_e + total_input

    print(f"Exc rates -> Max {np.max(responses_exc):.1f} "
          f"Mean {np.mean(responses_exc):.1f} "
          f"Median {np.median(responses_exc):.1f}")
    print(f"Inh rates -> Max {np.max(responses_inh):.1f} "
          f"Mean {np.mean(responses_inh):.1f} "
          f"Median {np.median(responses_inh):.1f}")

    return responses_exc, responses_inh, exc_in, inh_in

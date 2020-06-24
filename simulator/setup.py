import numpy as np
from typing import Union, NamedTuple, List, Optional
from .params import SimParams, Float, x2y
from .params import ArrayIE, ArrayEE, ArrayEI, ArrayII, ArrayXY
from .params import StimulusPreference, Locations, Afferents
from .params import AllParameters
from .utils import nonlinearity, nonlinearity_inverse
from .utils import allocate_aligned, reallocate_aligned


class SynapseArrays(NamedTuple):
    vie: ArrayIE
    vei: ArrayEI
    zie: ArrayIE
    zei: ArrayEI
    wie: ArrayIE
    wei: ArrayEI
    ie_max: Float
    ei_max: Float
    ie_min: Float
    ei_min: Float
    ie_total: Float
    alpha_ie: Float
    alpha_ei: Float
    wii: ArrayII
    wee: ArrayEE


class AfferentArrays(NamedTuple):
    stimulus_pref: StimulusPreference
    locations: Locations
    locations_idx: List[np.ndarray]
    afferents: Afferents


def make_afferents(
        n_d: int, a_e: int, n_stimuli: int, bg_input: Float,
        peak_stimulus: Float, vonmises_kappa: Float,
        plot_afferents: bool = False
) -> AfferentArrays:
    from itertools import product
    dtype = SimParams.float_type
    n_e = a_e ** n_d
    locations = Locations(np.linspace(-np.pi, np.pi, n_stimuli, endpoint=False, dtype=dtype))
    s_q: Union[str, List[np.ndarray]]  # str is to stop PyCharm from complaining about the type of s_q
    s_q = np.meshgrid(*(np.linspace(-np.pi, np.pi, a_e, endpoint=False, dtype=dtype)
                      for _ in range(n_d)))
    s_q = [reallocate_aligned(s) for s in s_q]
    stimulus_pref = StimulusPreference(allocate_aligned((n_e, n_d), dtype=dtype))
    inputs_q = []
    for i, s in enumerate(s_q):
        stimulus_pref[:, i] = s.flatten()
        loc, pref = np.meshgrid(locations, stimulus_pref[:, i])
        input_q = np.exp(vonmises_kappa * (np.cos(loc - pref) - dtype(1)))
        input_q = reallocate_aligned(input_q)
        input_q -= np.min(input_q)
        input_q /= np.max(input_q)
        inputs_q.append(input_q)

    afferents = Afferents(allocate_aligned(n_d * (n_stimuli,) + (n_e,), dtype=dtype))
    locations_idx = [reallocate_aligned(np.arange(n_stimuli)) for _ in range(n_d)]
    for ijk in product(*locations_idx):
        afferent_ijk = np.prod([inp[:, q] for inp, q in zip(inputs_q, ijk)], axis=0)
        afferents[ijk + (slice(None),)] = peak_stimulus * afferent_ijk

    afferents += bg_input

    if plot_afferents:
        afferents_plot(stimulus_pref, afferents)

    return AfferentArrays(stimulus_pref, locations, locations_idx, afferents)


def afferents_plot(
        stimulus_pref: StimulusPreference, afferents: Afferents, show=True
) -> None:
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt

    n_e = afferents.shape[-1]
    n_d = len(afferents.shape[:-1])
    n_stimuli = afferents.shape[0]
    sqr_e = np.int(np.ceil(np.sqrt(n_e)))
    for n in range(n_d):
        fig = plt.figure(figsize=(15, 15))
        fig.suptitle("Afferents")
        gs = GridSpec(
            sqr_e,
            sqr_e,
            hspace=0.02,
            wspace=0.02,
            top=0.95,
            bottom=0.01,
            left=0.01,
            right=0.99,
        )
        for m in range(n_e):
            ij = np.unravel_index(m, (sqr_e, sqr_e))
            ax = fig.add_subplot(gs[ij])
            ax.imshow(afferents[..., m].sum(axis=n))
            a = [0, 1, 2]
            a.remove(n)
            x = n_stimuli * (stimulus_pref[m, a[1]] + np.pi) / (2 * np.pi)
            y = n_stimuli * (stimulus_pref[m, a[0]] + np.pi) / (2 * np.pi)
            ax.plot(x, y, 'x')
            ax.set_xticks([])
            ax.set_yticks([])
    if show:
        plt.show()


def afferents_to_wee(
        e2e: x2y,
        target_correlations: ArrayEE,
        adjust_ee_conn_prob: bool,
        rank_order: bool = True,
        exp_ee: bool = False,
        plot: bool = False,
) -> ArrayEE:
    import matplotlib.pyplot as plt

    jee = e2e.w_total

    wee = target_correlations.copy()
    if rank_order:
        wee[np.diag_indices_from(wee)] = 0
        fee = wee.flatten()
        sorted_idx = np.argsort(fee)
        to_zero = int((1 - e2e.p) * fee.size)
        fee -= fee.min()
        fee[sorted_idx[:to_zero]] = 0
        w_min = fee.min()
        if w_min < 0:
            fee[sorted_idx[to_zero:]] -= w_min
        wee[...] = fee.reshape(wee.shape)
        wee[np.diag_indices_from(wee)] = 0
        p_ee = np.sum(wee > 0) / target_correlations.size
        print(f"WEE connection probability {p_ee}")
    else:
        wee -= np.min(wee)
        wee[np.diag_indices_from(wee)] = 0
        if adjust_ee_conn_prob:
            wee[wee < 0] = 0
            p_ee = np.sum(wee > 0) / target_correlations.size
            while p_ee > e2e.p + 0.05:
                wee -= 1e-2
                wee[wee < 0] = 0
                wee[np.diag_indices_from(wee)] = 0
                p_ee = np.sum(wee > 0) / target_correlations.size
                print(f"Decrement to {p_ee}")
        wee[wee < 0] = 0

    if exp_ee:
        eta = SimParams.float_type(3)
        wee = (wee > 0) * np.exp(eta*wee)

    w_tot = wee.sum(axis=1)
    wee *= jee / w_tot[:, None]
    print(f"Positive correlation: {np.sum(target_correlations > 0) / target_correlations.size}")
    print(f"Connection probability: {np.sum(wee > 0) / target_correlations.size}")
    wee = ArrayEE(reallocate_aligned(wee))

    if plot:
        plt.hist(wee[wee > 0].flatten(), bins=40)
        plt.figure()
        plt.imshow(wee)
        plt.show()
    return wee


def make_synapses(
        params: AllParameters,
        n_e: int, n_i: int,
        target_correlations: Optional[ArrayEE] = None,
        adjust_ee_conn_prob: bool = True,
        plot_weight_hist: bool = False,
) -> SynapseArrays:
    import matplotlib.pyplot as plt
    dtype = SimParams.float_type

    sy = params.sy
    pl = params.pl
    w_max = pl.w_max

    p_ee = sy.e2e.p
    p_ei = sy.i2e.p
    p_ii = sy.i2i.p
    p_ie = sy.e2i.p
    kee = np.int(np.round(p_ee * n_e))
    kei = np.int(np.round(p_ei * n_i))
    kii = np.int(np.round(p_ii * n_i))
    kie = np.int(np.round(p_ie * n_e))

    jee = sy.e2e.w_total / max(kee, 1)
    jei = sy.i2e.w_total / kei
    jii = sy.i2i.w_total / kii
    jie = sy.e2i.w_total / kie
    ei_min = sy.i2e.w_min / kei
    ie_min = sy.e2i.w_min / kie

    if jee > 0:
        wee = afferents_to_wee(sy.e2e, target_correlations, adjust_ee_conn_prob)
    else:
        wee = ArrayEE(allocate_aligned((n_e, n_e), dtype=dtype))
    zee = wee[wee > 0]

    wei = ArrayEI(allocate_aligned((n_e, n_i), dtype=dtype))
    wii = ArrayII(allocate_aligned((n_i, n_i), dtype=dtype))
    wie = ArrayIE(allocate_aligned((n_i, n_e), dtype=dtype))
    zei = ArrayEI(allocate_aligned(wei.shape, dtype=wei.dtype))
    zie = ArrayIE(allocate_aligned(wie.shape, dtype=wie.dtype))
    zii = ArrayII(allocate_aligned(wii.shape, dtype=wii.dtype))

    if sy.w_dist == "normal":
        for w, z, n_post, n_pre, jxy, k in zip(
            (wei, wii, wie),
            (zei, zii, zie),
            (n_e, n_i, n_i),
            (n_i, n_i, n_e),
            (jei, jii, jie),
            (kei, kii, kie),
        ):
            var = jxy / 4
            for i in range(n_post):
                wj = np.random.normal(jxy, var, size=n_pre)
                not_pos_w = wj <= 0
                pos_w = np.logical_not(not_pos_w)
                to_be_zeroed = pos_w.sum() - k
                if to_be_zeroed > 0:
                    tbz = np.squeeze(np.argwhere(pos_w))
                    np.random.shuffle(tbz)
                    tbz = tbz[:to_be_zeroed]
                    wj[tbz] = 0
                elif to_be_zeroed < 0:
                    to_be_pos = -to_be_zeroed
                    tbp = np.squeeze(np.argwhere(not_pos_w))
                    if tbp.size > 0:
                        np.random.shuffle(tbp)
                        tbp = tbp[:to_be_pos]
                        wj[tbp] = np.random.normal(jxy, var, size=to_be_pos)
                    not_pos_w = wj <= 0
                wj[not_pos_w] = 0
                w[i, :] = wj
                z[i, wj > 0] = 1
            # rel_err = np.mean([np.abs(w[i,:].mean() - jxy) / jxy for i in range(n_post)])
    else:
        lognorm_sigma = sy.lognorm_sigma
        j = np.arange(n_i)
        for i in range(n_e):
            np.random.shuffle(j)
            if sy.w_dist == "lognorm":
                wj = np.random.lognormal(
                    np.log(jei * np.exp(-(lognorm_sigma ** 2) / 2)),
                    lognorm_sigma,
                    size=kei,
                )
            else:
                wj = jei * np.random.uniform(low=1e-12, high=2, size=kei)
            wei[i, j[:kei]] = wj
            zei[i, j[:kei]] = 1
        for i in range(n_i):
            np.random.shuffle(j)
            if sy.w_dist == "lognorm":
                wj = np.random.lognormal(
                    np.log(jii * np.exp(-(lognorm_sigma ** 2) / 2)),
                    lognorm_sigma,
                    size=kii,
                )
            else:
                wj = jii * np.random.uniform(low=1e-12, high=2, size=kii)
            wii[i, j[:kii]] = wj
            zii[i, j[:kii]] = 1
        j = np.arange(n_e)
        for i in range(n_i):
            np.random.shuffle(j)
            if sy.w_dist == "lognorm":
                wj = np.random.lognormal(
                    np.log(jie * np.exp(-(lognorm_sigma ** 2) / 2)),
                    lognorm_sigma,
                    size=kie,
                )
            else:
                wj = jie * np.random.uniform(low=1e-12, high=2, size=kie)
            wie[i, j[:kie]] = wj
            zie[i, j[:kie]] = 1

    soft_max = params.pl.soft_max
    if soft_max:
        ei_max = dtype(np.max(wei))
        ie_max = dtype(np.max(wie))
        alpha_ie = dtype(4.0 / ie_max)
        alpha_ei = dtype(4.0 / ei_max)
        # wei_decay = dtype(0)
        # wie_decay = dtype(0)
        print(f"{type(ei_max)}, {type(ie_max)}")
    else:
        alpha_ie = dtype(0.05 / jie)
        alpha_ei = dtype(0.05 / jei)
        ei_max = w_max * jei
        ie_max = w_max * jie
        print(f"alpha_ie {alpha_ie}, alpha_ei {alpha_ei}")

    vei = ArrayEI(np.zeros_like(wei))
    for i in range(n_e):
        rei = wei[i, :]
        vei[i, rei > 0] = nonlinearity_inverse(rei[rei > 0], alpha_ei)
    vie = ArrayIE(np.zeros_like(wie))
    for i in range(n_i):
        rie = wie[i, :]
        vie[i, rie > 0] = nonlinearity_inverse(rie[rie > 0], alpha_ie)
    vei *= zei
    vie *= zie
    vei = ArrayEI(reallocate_aligned(vei))
    vie = ArrayIE(reallocate_aligned(vie))

    if jee > 0:
        print(f"Wee {np.max(wee)/np.min(wee[wee > 0]): .2e}")
    print(f"Wei {np.max(wei)/np.min(wei[wei > 0]): .2e}")
    print(f"Wie {np.max(wie)/np.min(wie[wie > 0]): .2e}")
    print(f"Wii {np.max(wii)/np.min(wii[wii > 0]): .2e}")

    fig_hist = plt.figure() if plot_weight_hist else None
    fig_im = plt.figure() if plot_weight_hist else None
    for i, (w, z, n_post, n_pre, jxy, k, v, a, wm) in enumerate(
        zip(
            (wee, wei, wii, wie),
            (zee, zei, zii, zie),
            (n_e, n_e, n_i, n_i),
            (n_e, n_i, n_i, n_e),
            (jee, jei, jii, jie),
            (kee, kei, kii, kie),
            (None, vei, None, vie),
            (0, alpha_ei, 0, alpha_ie),
            (0, ei_max, 0, ie_max),
        )
    ):
        w1 = w.copy()
        w1[w1 <= 0] = np.NaN
        if jxy > dtype(0):
            rel_err = np.mean(np.abs(np.nanmean(w1, axis=1) - jxy) / jxy)
            print(f"Relative error in mean weight: {rel_err:.2f}")

            print(f"Std dev of non-zero weights: {np.std(np.log(w[w > 0]))}")
            tot_w = np.sum(w, axis=1)
            exp_w = k*jxy
            avg_err = np.mean(np.abs(tot_w - exp_w) / exp_w)
            print(f"Average error in total weight: {avg_err:.2f}")
            print(f"w_max: {w.max()} non-neg: {(w > 0).sum()/np.product(w.shape)}")
            assert (w >= 0).all()
            if plot_weight_hist:
                ax = fig_im.add_subplot(2, 2, i+1)
                ax.imshow(w)
                ax = fig_hist.add_subplot(4, 3, (3 * i) + 1)
                ax.set_title(f"{w[w>0].size / w.size}")
                ax.hist(
                    w[w > 0].flatten(),
                    bins=np.logspace(np.log10(w[w > 0].min()), np.log10(w.max()), 100),
                )
                ax.vlines(jxy, ymin=0, ymax=plt.ylim()[1])
                ax.set_xscale("log")
                ax = fig_hist.add_subplot(4, 3, (3 * i + 1) + 1)
                ax.hist(w[w > 0].flatten(), bins=np.linspace(w[w > 0].min(), w.max(), 100))
                ax.vlines(jxy, ymin=0, ymax=plt.ylim()[1])
                if v is not None:
                    ax = fig_hist.add_subplot(4, 3, (3 * i + 2) + 1)
                    wt = ArrayXY(np.empty_like(v))
                    nonlinearity(v, wt, a, wm)
                    wt *= z
                    ax.hist(
                        wt[wt > 0].flatten(),
                        bins=np.linspace(wt[wt > 0].min(), np.max(wt), 100),
                    )
    if plot_weight_hist:
        plt.show()

    sya = SynapseArrays(
        vie=vie, vei=vei,
        zie=zie, zei=zei,
        wie=wie, wei=wei,
        ie_max=ie_max, ei_max=ei_max,
        ie_min=ie_min, ei_min=ei_min,
        ie_total=sy.e2i.w_total,
        alpha_ie=alpha_ie, alpha_ei=alpha_ei,
        wii=wii, wee=wee
    )
    return sya

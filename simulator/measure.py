import numpy as np
from typing import Tuple
from .params import SimParams, Inputs
from .setup import AfferentArrays


def orientation_selectivity_index(
        inp: Inputs, responses_exc: np.ndarray, responses_inh: np.ndarray, aff_arrays: AfferentArrays
) -> Tuple[np.ndarray, np.ndarray]:

    n_e = responses_exc.shape[-1]
    n_i = responses_inh.shape[-1]
    osi_e = np.empty((n_e,), dtype=SimParams.float_type)
    osi_i = np.empty((n_i,), dtype=SimParams.float_type)
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
    return osi_e, osi_i


def compute_syn_current_spearmanr(exc_in: np.ndarray, inh_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.stats import spearmanr
    n_e = exc_in.shape[-1]
    cc = np.ones((n_e,), dtype=SimParams.float_type)
    cp = np.ones_like(cc)
    for k in range(n_e):
        rho, p = spearmanr(exc_in[..., k].flatten(), inh_in[..., k].flatten())
        cc[k] = rho
        cp[k] = p if np.isfinite(p) else 0

    return cc, cp


def compute_response_similarity(responses_a: np.ndarray, responses_b: np.ndarray) -> np.ndarray:
    from numpy.linalg import norm

    n_a, n_b = responses_a.shape[-1], responses_b.shape[-1]
    response_sim = np.zeros((n_a, n_b))

    n2_a = np.empty(n_a)
    for i in range(n_a):
        n2_a[i] = norm(responses_a[..., i].flatten(), 2)
        if np.isclose(n2_a[i], 0.0):
            print(f"Neuron {i} in population a had no response")

    n2_b = np.empty(n_b)
    for j in range(n_b):
        n2_b[j] = norm(responses_b[..., j].flatten(), 2)
        if np.isclose(n2_b[j], 0.0):
            print(f"Neuron {j} in population b had no response")

    for i in range(n_a):
        eci = responses_a[..., i].flatten()
        for j in range(n_b):
            icj = responses_b[..., j].flatten()
            rs = np.dot(eci, icj) / (n2_a[i] * n2_b[j])
            response_sim[i, j] = rs

    return response_sim

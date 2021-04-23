import numpy as np
from numba import njit
from typing import Union, NamedTuple, Tuple, Optional
from .utils import nonlinearity, nonlinearity_derivative, angle
from .params import numba_cache, Float, SimParams
from .params import VectorE, VectorI, ArrayIE, ArrayEE, ArrayEI, ArrayII, ArrayXY
from .params import PlasticityTypeEtoI, PlasticityTypeItoE


class MomentEstimate(NamedTuple):
    m: ArrayXY
    v: ArrayXY
    b1: Float = SimParams.float_type(0.9)
    b2: Float = SimParams.float_type(0.999)
    epsilon: Float = SimParams.float_type(1e-8)


@njit(cache=numba_cache)
def approximate_update(
        delta_vxy: ArrayXY,
        vxy: ArrayXY, zxy: ArrayXY, wxy: ArrayXY,
        eta: Float, wxy_decay: Float, alpha_xy: Float,
        xy_max: Float
) -> None:
    if wxy_decay > 0:
        delta_vxy -= wxy_decay * wxy
    # delta_vxy *= zxy
    delta_vxy *= eta
    vxy += delta_vxy

    nonlinearity(vxy, wxy, alpha_xy, xy_max)
    wxy *= zxy


@njit(cache=numba_cache)
def weight_update_adam(
        grad: ArrayXY,
        vxy: ArrayXY, zxy: ArrayXY, wxy: ArrayXY,
        eta: Float, wxy_decay: Float, alpha_xy: Float,
        wxy_max: Float,
        adam: MomentEstimate,
) -> None:
    if wxy_decay > 0:
        grad -= wxy_decay * wxy
    adam.m[...] = adam.b1 * adam.m + (1 - adam.b1) * grad
    adam.v[...] = adam.b2 * adam.v + (1 - adam.b2) * np.power(grad, 2)
    m_hat = adam.m / (1 - adam.b1)
    v_hat = adam.v / (1 - adam.b2)
    vxy += eta / (np.sqrt(v_hat) + adam.epsilon) * m_hat

    nonlinearity(vxy, wxy, alpha_xy, wxy_max)
    wxy *= zxy


@njit(cache=numba_cache)
def wie_approximate_gradient(
        error, re: VectorE, hi: VectorI,
        vie: ArrayIE, alpha_ie: Float,
        propagation_matrix: Union[ArrayIE, ArrayEI],
        ie_max: Float,
) -> ArrayIE:
    # cie = wie.copy()
    #     cie[...] = np.where(cie > 0, 1, 0)
    # sum_ie = cie.sum(axis=1)
    # sum_ie[sum_ie <= 0] = 1
    # cie /= sum_ie
    fp_error = propagation_matrix @ error

    fp_error *= np.reciprocal(1 + np.exp(-hi))
    delta_vie = np.outer(fp_error, re)
    delta_vie *= nonlinearity_derivative(vie, alpha_ie, ie_max)
    return delta_vie


@njit(cache=numba_cache)
def wei_approximate_gradient(
        error, ri: VectorI,
        vei: ArrayEI, alpha_ei: Float,
        ei_max: Float
) -> ArrayEI:
    delta_vei = np.outer(error, ri)
    delta_vei *= nonlinearity_derivative(vei, alpha_ei, ei_max)
    return delta_vei


@njit(cache=numba_cache)
def gradient_compute(
        error,
        re: VectorE, ri: VectorI,
        he: VectorE, hi: VectorI,
        vie: ArrayIE, wie: ArrayIE,
        vei: ArrayEI, wei: ArrayEI,
        wee: ArrayEE, wii: ArrayII,
        eta_e: Float, eta_i: Float,
        alpha_ie: Float, alpha_ei: Float,
        ie_max: Float, ei_max: Float,
        grad_ie: ArrayIE, grad_ei: ArrayEI,
) -> None:
    n_e, n_i = wei.shape
    dia_i = np.diag(nonlinearity_derivative(hi, 1))
    dia_e = np.diag(nonlinearity_derivative(he, 1))
    # M = np.linalg.inv(np.eye(n_i, dtype=wii.dtype) + wii @ dia_i)
    # W = np.linalg.inv(np.eye(n_e, dtype=wee.dtype) - wee @ dia_e + wei_i_M @ wie @ dia_e)
    wei_dia_i_m = wei @ dia_i @ np.linalg.inv(np.eye(n_i, dtype=wei.dtype) + wii @ dia_i)
    error_w = error @ np.linalg.inv(np.eye(n_e, dtype=wee.dtype) - wee @ dia_e + wei_dia_i_m @ wie @ dia_e)
    if eta_i > 0:
        delta_ei = np.outer(error_w, ri)
        grad_ei[...] = delta_ei * nonlinearity_derivative(vei, alpha_ei, ei_max)

    if eta_e > 0:
        delta_ie = np.outer(error_w @ wei_dia_i_m, re)
        grad_ie[...] = delta_ie * nonlinearity_derivative(vie, alpha_ie, ie_max)


@njit(cache=numba_cache)
def bcm_update(
        re: VectorE, ri: VectorI,
        zie: ArrayIE, wie: ArrayIE,
        r_i_slow: VectorI, eta_e: Float, ie_max: Float, bcm_theta: Float
) -> None:
    ris2 = np.power(r_i_slow, 2)
    ris2 /= bcm_theta
    wie += eta_e * np.outer((ri - ris2) * ri, re)

    wie[...] = np.where(wie < ie_max, wie, ie_max)
    wie *= zie
    wie[...] = np.where(wie > 0, wie, 0)


@njit(cache=numba_cache)
def update_weights(
        rho0: Float, re: VectorE, ri: VectorI, he: VectorE, hi: VectorI,
        vie: ArrayIE, vei: ArrayEI,
        zei: ArrayEI, zie: ArrayIE,
        wie: ArrayIE, wei: ArrayEI,
        wee: ArrayEE, wii: ArrayII,
        eta_e: Float, eta_i: Float,
        wie_decay: Float, wei_decay: Float,
        alpha_ie: Float, alpha_ei: Float,
        ie_max: Float, ei_max: Float,
        weights_in_error_prop: bool, ei_min: Float, ie_total: Float,
        pl_type_ie: PlasticityTypeEtoI, pl_type_ei: PlasticityTypeItoE,
        adam_ie: Optional[MomentEstimate], adam_ei: Optional[MomentEstimate],
        compute_angles: bool = False,
) -> Tuple[Float, Float]:

    error = he - rho0

    grad_ei = approx_ei = vei
    grad_ie = approx_ie = vie

    gradient_plasticity = pl_type_ie == PlasticityTypeEtoI.GRADIENT or pl_type_ei == PlasticityTypeItoE.GRADIENT
    compute_grad = compute_angles or gradient_plasticity

    compute_approx_ie = compute_angles or (eta_e > 0 and
                                           pl_type_ie in (PlasticityTypeEtoI.APPROXIMATE,
                                                          PlasticityTypeEtoI.BACKPROP))
    compute_approx_ei = compute_angles or (eta_i > 0 and
                                           pl_type_ei in (PlasticityTypeItoE.APPROXIMATE,))

    if compute_grad:
        grad_ei = vei.copy()
        grad_ie = vie.copy()
        gradient_compute(
            error,
            re, ri,
            he, hi,
            vie, wie,
            vei, wei,
            wee, wii,
            eta_e, eta_i,
            alpha_ie, alpha_ei,
            ie_max, ei_max,
            grad_ie, grad_ei,
        )

    if compute_approx_ie:
        if pl_type_ie == PlasticityTypeEtoI.APPROXIMATE:
            propagation_matrix = wie if weights_in_error_prop else vie
        else:
            propagation_matrix = wei.T if weights_in_error_prop else vei.T

        approx_ie = wie_approximate_gradient(
            error, re, hi, vie, alpha_ie, propagation_matrix, ie_max
        )

    if compute_approx_ei:
        approx_ei = wei_approximate_gradient(
            error, ri, vei, alpha_ei, ei_max
        )

    angle_ei = angle(grad_ei, approx_ei) if compute_grad and compute_approx_ei \
        else np.NaN

    angle_ie = angle(grad_ie, approx_ie) if compute_grad and compute_approx_ie \
        else np.NaN

    if gradient_plasticity:
        assert pl_type_ei == PlasticityTypeItoE.GRADIENT and pl_type_ei == PlasticityTypeItoE.GRADIENT
        assert (adam_ei is not None) and (adam_ie is not None)

        if eta_i > 0:
            weight_update_adam(
                grad_ei, vei, zei, wei, eta_i, wei_decay, alpha_ei, ei_max, adam_ei
            )

        if eta_e > 0:
            weight_update_adam(
                grad_ie, vie, zie, wie, eta_e, wie_decay, alpha_ie, ie_max, adam_ie
            )

    else:
        if eta_e > 0:
            approximate_update(
                approx_ie, vie, zie, wie, eta_e, wie_decay, alpha_ie, ie_max
            )
            if pl_type_ie == PlasticityTypeEtoI.APPROXIMATE:
                sum_ie = wie.sum(axis=1)
                sum_ie[sum_ie <= 0] = 1
                # wie *= ie_total / sum_ie[:, None]
                # wie *= ie_total / sum_ie  # This is broken in numba!
                for j in range(wie.shape[0]):
                    wie[j, :] *= ie_total / sum_ie[j]

        if eta_i > 0 and pl_type_ei == PlasticityTypeItoE.APPROXIMATE:
            approximate_update(
                approx_ei, vei, zei, wei, eta_i, wei_decay, alpha_ei, ei_max
            )

    return angle_ei, angle_ie

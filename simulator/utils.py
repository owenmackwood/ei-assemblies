import numpy as np
from numba import njit
from typing import Union, Tuple, Type
from .params import SimParams, Float, numba_cache
from .params import VectorE, VectorI, ArrayXY


@njit(cache=False)
def allocate_aligned(
        shape: Union[int, Tuple[int, ...]],
        fill_value: [np.number, int, float] = 0,
        dtype: Type[np.number] = SimParams.float_type
) -> np.ndarray:
    """

    :param shape:
    :param fill_value:
    :param dtype_:
    :return:
    """
    # NUMBA allocates arrays with 32-byte alignment
    return np.full(shape, fill_value, dtype=dtype)


def reallocate_aligned(
        a: np.ndarray
) -> np.ndarray:
    """

    :param a: Array to be reallocated.
    :return: Returns a newly allocated array
    """
    b = allocate_aligned(a.shape, a.dtype)
    np.copyto(b, a)
    # assert not b.ctypes.data % alignment
    return b


@njit(cache=numba_cache)
def nonlinearity(
        vxy: ArrayXY, wxy: ArrayXY, alpha_xy: Float, wxy_max: Float = SimParams.float_type(np.infty)
) -> None:
    wxy[...] = vxy
    wxy *= alpha_xy
    wxy[...] = np.exp(wxy)
    wxy[...] = np.log1p(wxy)
    wxy /= alpha_xy
    wxy[...] = np.where(np.isfinite(wxy), wxy, vxy)
    wxy[...] = np.where(wxy < wxy_max, wxy, wxy_max)


def nonlinearity_inverse(
        wxy: ArrayXY, alpha_xy: Float
) -> ArrayXY:
    vxy = wxy.copy()
    vxy *= alpha_xy
    np.expm1(vxy, out=vxy)
    np.log(vxy, out=vxy)
    vxy /= alpha_xy
    return vxy


@njit(cache=numba_cache)
def nonlinearity_derivative(
        vxy: Union[ArrayXY, VectorE, VectorI], alpha_xy: Float, _wxy_max: Float = SimParams.float_type(np.NaN)
) -> Union[ArrayXY, VectorE, VectorI]:
    out = vxy.copy()
    out *= -alpha_xy
    out[...] = np.exp(out)
    out += 1
    out[...] = np.reciprocal(out)

    if not np.isfinite(out).all():
        print("nonlinearity_derivative blew up")

    out[...] = np.where(np.isfinite(out), out, 1)
    return out


@njit(cache=numba_cache)
def nonlinearity_max(
        vxy: ArrayXY, wxy: ArrayXY, alpha_xy: Float, wxy_max: Float
) -> None:
    wxy[...] = vxy
    wxy *= alpha_xy
    wxy[...] = np.exp(wxy)

    _m = wxy.copy()
    _m *= np.exp(-alpha_xy * wxy_max)

    wxy[...] = np.log1p(wxy)
    wxy[...] = np.where(np.isfinite(wxy), wxy, vxy)
    _m[...] = np.log1p(_m)
    _m[...] = np.where(np.isfinite(_m), _m, vxy - wxy_max)

    wxy -= _m
    wxy /= alpha_xy


@njit(cache=numba_cache)
def nonlinearity_max_derivative(
        vxy: ArrayXY, alpha_xy: Float, wxy_max: Float
) -> ArrayXY:
    out = vxy.copy()
    out *= -alpha_xy
    out[...] = np.exp(out)
    out[...] = np.reciprocal(1 + out) - np.reciprocal(
        1 + out * np.exp(alpha_xy * wxy_max)
    )
    # e = np.exp(-alpha * v)
    # u = np.exp(alpha * c)
    # return (1 + e)**-1 - (1 + e*u)**-1
    return out


@njit(cache=numba_cache)
def angle(a, b) -> Float:
    a = a.flatten()
    b = b.flatten()
    return np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

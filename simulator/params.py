import numpy as np
from numba import njit
import inspect, warnings
from typing import Tuple, Dict, Union, ClassVar, NewType, List, Type
from enum import Enum

numba_cache = False

Float = Union[float, Type[np.floating]]
VectorE = NewType("VectorE", np.ndarray)
VectorI = NewType("VectorI", np.ndarray)
ArrayIE = NewType("ArrayIE", np.ndarray)
ArrayEI = NewType("ArrayEI", np.ndarray)
ArrayII = NewType("ArrayII", np.ndarray)
ArrayEE = NewType("ArrayEE", np.ndarray)
ArrayXY = Union[ArrayEE, ArrayEI, ArrayII, ArrayIE]

Locations = NewType("Locations", np.ndarray)
StimulusPreference = NewType("StimulusPreference", np.ndarray)
Afferents = NewType("Afferents", np.ndarray)


_leaf_types = (str, bytes, float, int, bool, np.ndarray,
               np.float16, np.int32, np.int64, np.float32, np.float64, np.float128)
LeafTypes = Union[_leaf_types]
ParamDict = Dict[str, Union['ParamDict', LeafTypes]]
ParamPath = Tuple[str, ...]
ParamSpacePt = Dict[ParamPath, LeafTypes]

ParamArray = Union[List[_leaf_types], np.ndarray]
ParamRanges = Dict[str, Union["ParamRanges", ParamArray]]


class Leaf:
    """
    Wrapper class for the types stored in a SimParams structure.
    It is only used when trying to find a user supplied parameter value, because the SimParams class
    uses ``obj_a is obj_b`` to check for identity, which misidentifies bools, ints and floats.
    i.e.
    a = 0
    b = 0
    a is b (evaluates to true)
    """
    def __init__(self, value: LeafTypes):
        self.leaf = value


class MissingContext(Exception):
    pass


class SimParams:
    """
    Base class for all user-defined simulation parameters. It has two modes of operation: as a context manager
    while setting up the set up simulations to be run, and as a normal object which can be constructed
    using a dictionary, mapping strings to parameter values or other dictionaries of parameters.

    When used as a context manager,
    """
    float_type: ClassVar[Float] = np.float32

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            self.__setattr__(name, value)

    def __setattr__(self, name, value):
        if isinstance(value, _leaf_types):
            if np.issubdtype(type(value), np.floating):
                value = SimParams.float_type(value)
            elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                value = value.astype(SimParams.float_type)
        super().__setattr__(name, value)

    def __enter__(self) -> 'SimParams':
        for name in self.__dict__.keys():
            value = self.__getattribute__(name)
            if isinstance(value, _leaf_types):
                self.__dict__[name] = Leaf(value)
            elif isinstance(value, SimParams):
                self.__dict__[name] = value.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name in self.__dict__.keys():
            value = self.__getattribute__(name)
            if isinstance(value, Leaf):
                self.__dict__[name] = value.leaf
            elif isinstance(value, SimParams):
                value.__exit__(exc_type, exc_val, exc_tb)

    def _path_to(self, leaf: Leaf, path_to: ParamPath = ()) -> ParamPath:
        """
        :param leaf: The parameter to find the path to. Is of type _leaf, but PyCharm's type checker cannot handle
                        this due to the context manager trickiness.
        :param path_to: Used internally for recursion.
        :return: A tuple of strings containing the full path to the parameter at `leaf`
        """
        for name in self.__dict__.keys():
            value = self.__getattribute__(name)
            if isinstance(value, SimParams):
                pt = value._path_to(leaf, path_to + (name,))
                if pt != path_to + (name,):
                    path_to = pt
                    break
            elif value is leaf:
                if not isinstance(value, Leaf):
                    raise MissingContext('SimParams._path_to must be called inside of a `with` block.')
                path_to = path_to + (name,)
                break
        return path_to

    def _magic(self, params: Dict) -> None:
        for name in self.__dict__.keys():
            value = self.__getattribute__(name)
            if isinstance(value, SimParams):
                value._magic(params)
            else:
                if name in params:
                    warnings.warn(f'Repeat parameter name {name} will cause collision in namespace!')
                params[name] = value

    def magic(self, level: int = 0) -> None:
        params = {}
        self._magic(params)

        frame = inspect.stack()[level + 1][0]
        both_names = set(params.keys()).intersection(frame.f_globals.keys())
        if len(both_names):
            warnings.warn(f'The following local parameters will be overwritten by `magic` {both_names}')
        frame.f_globals.update(params)

        del frame

    def include_range(self, param_ranges: ParamRanges,
                      leaf: Union[Leaf, LeafTypes], values: ParamArray) -> None:
        """
        Only to be called when this object is being used as a context manager.

        :param param_ranges: Nested dictionaries that should store all parameter ranges.
        :param leaf: Leaf containing the parameter to be varied.
        :param values: The values that the parameter should take.
        :return: None
        """
        path_to = self._path_to(leaf)
        pr = param_ranges
        for name in path_to[:-1]:
            if name not in pr:
                pr[name] = {}
            pr = pr[name]
        pr[path_to[-1]] = values

    def link_ranges(self, *leafs: Union[Leaf, LeafTypes]) -> Tuple[Tuple[str, ...], ...]:
        """
        Only to be called when this object is being used as a context manager.

        Intended to be used in conjunction with `snep.tables.ExperimentTables.link_parameter_ranges`

        :param leafs: An arbitrary number of parameters to be linked i.e. co-varied
        :return: A tuple where each element contain the full path to the corresponding parameter, as required by `SNEP`.
        """
        return tuple(self._path_to(leaf) for leaf in leafs)

    def dict(self) -> ParamDict:
        """
        Only to be called when this object is being used as a context manager.

        Returns the equivalent dictionary representation of all simulation parameters. This can
        be used with `snep.tables.ExperimentTables.add_parameters` or if you want to pickle the
        parameters for later use.

        :return: Dictionary of dictionaries or parameter values.
        """
        dict_representation = {}
        for name in self.__dict__.keys():
            value = self.__getattribute__(name)
            if isinstance(value, SimParams):
                dict_representation[name] = value.dict()
            elif isinstance(value, Leaf):
                dict_representation[name] = value.leaf
            else:
                raise MissingContext('SimParams.dict must be called inside of a `with` block.')
        return dict_representation


class ExperimentType(Enum):
    GRADIENT = 1
    APPROXIMATE = 2


class PlasticityTypeEtoI(Enum):
    GRADIENT = 1
    APPROXIMATE = 2
    ANTEROGRADE = 3
    # Doesn't work with numba
    # @classmethod
    # def from_str(cls, name: str) -> "PlasticityTypeEtoI":
    #     return cls._member_map_[name]


class PlasticityTypeItoE(Enum):
    GRADIENT = 1
    APPROXIMATE = 2
    # Doesn't work with numba
    # @classmethod
    # def from_str(cls, name: str) -> "PlasticityTypeItoE":
    #     return cls._member_map_[name]


@njit(cache=numba_cache)
def plasticity_ie_type_from_str(name: str) -> PlasticityTypeEtoI:
    """
    Necessary due to Numba not supporting _member_map_ access in class methods.
    :param name: The name of the enumeration value.
    :return: The enumeration value.
    """
    if name == "GRADIENT":
        return PlasticityTypeEtoI.GRADIENT
    elif name == "APPROXIMATE":
        return PlasticityTypeEtoI.APPROXIMATE
    elif name == "ANTEROGRADE":
        return PlasticityTypeEtoI.ANTEROGRADE
    else:
        print(name)
        raise ValueError("Unknown PlasticityTypeEtoI")


@njit(cache=numba_cache)
def plasticity_ei_type_from_str(name: str) -> PlasticityTypeItoE:
    """
    Necessary due to Numba not supporting _member_map_ access in class methods.
    :param name: The name of the enumeration value.
    :return: The enumeration value.
    """
    if name == "GRADIENT":
        return PlasticityTypeItoE.GRADIENT
    elif name == "APPROXIMATE":
        return PlasticityTypeItoE.APPROXIMATE
    else:
        print(name)
        raise ValueError("Unknown PlasticityTypeItoE")


class NumericalIntegration(SimParams):
    def __init__(
        self, dt: Float, max_dr: Float, max_steps: int, do_abort: bool, n_trials: int
    ):
        # super().__init__(dt=dt, max_dr=max_dr, max_steps=max_steps, do_abort=do_abort, n_trials=n_trials)
        super().__init__()
        self.dt: Float = dt
        self.max_dr: Float = max_dr
        self.max_steps: int = max_steps
        self.do_abort: bool = do_abort
        self.n_trials: int = n_trials


class Population(SimParams):
    def __init__(self, tau_m: Float, n_per_axis: int):
        # super().__init__(tau_m=tau_m, n_per_axis=n_per_axis)
        super().__init__()
        self.tau_m: Float = tau_m
        self.n_per_axis: int = n_per_axis


class NeuronGroups(SimParams):
    def __init__(
        self,
        n_d: int,
        r_max: Float,
        exc: Union[Population, Dict],
        inh: Union[Population, Dict],
    ):
        # super().__init__(n_d=n_d, r_max=r_max, exc=exc, inh=inh)
        super().__init__()
        self.n_d: int = n_d
        self.r_max: Float = r_max
        self.exc: Population = Population(**exc) if isinstance(exc, dict) else exc
        self.inh: Population = Population(**inh) if isinstance(inh, dict) else inh


class x2y(SimParams):
    def __init__(self, w_total: Float, p: Float, w_min: Float = 0.0):
        # super().__init__(w_total=w_total, p=p)
        super().__init__()
        self.w_total: Float = w_total
        self.p: Float = p
        self.w_min: Float = w_min


class Synapses(SimParams):
    def __init__(
        self,
        w_dist: str,
        lognorm_sigma: Float,
        e2i: Union[x2y, Dict],
        i2e: Union[x2y, Dict],
        i2i: Union[x2y, Dict],
        e2e: Union[x2y, Dict] = None,
    ):
        # super().__init__(w_dist=w_dist, w_max=w_max, lognorm_sigma=lognorm_sigma, e2i=e2i, i2e=i2e, i2i=i2i)
        super().__init__()
        self.w_dist: str = w_dist
        self.lognorm_sigma: Float = lognorm_sigma
        self.e2e: x2y = x2y(**e2e) if isinstance(e2e, dict) else e2e
        self.e2i: x2y = x2y(**e2i) if isinstance(e2i, dict) else e2i
        self.i2e: x2y = x2y(**i2e) if isinstance(i2e, dict) else i2e
        self.i2i: x2y = x2y(**i2i) if isinstance(i2i, dict) else i2i


class BCMParams(SimParams):
    def __init__(
        self,
        active: bool = False,
        theta: Float = np.NaN,
        tau_inv: Float = 0.0,
    ):
        super().__init__()
        self.active: bool = active
        self.theta: Float = theta
        self.tau_inv: Float = tau_inv


class Plasticity(SimParams):
    def __init__(
        self,
        rho0: Float,
        is_plastic: str,
        bp_weights: bool,
        eta_e: Float,
        eta_i: Float,
        w_max: Float,
        wei_decay: Float,
        wie_decay: Float,
        convergence_max: Float,
        convergence_mean: Float,
        bcm: BCMParams = BCMParams(),
        soft_max: bool = False,
        plasticity_type_ei: Union[str, PlasticityTypeItoE] = PlasticityTypeItoE.APPROXIMATE,
        plasticity_type_ie: Union[str, PlasticityTypeEtoI] = PlasticityTypeEtoI.APPROXIMATE,
    ):
        super().__init__()
        self.rho0: Float = rho0
        self.is_plastic: str = is_plastic
        self.bp_weights: bool = bp_weights
        self.bcm: BCMParams = BCMParams(**bcm) if isinstance(bcm, dict) else bcm
        self.eta_e: Float = eta_e
        self.eta_i: Float = eta_i
        self.w_max: Float = w_max
        self.wei_decay: Float = wei_decay
        self.wie_decay: Float = wie_decay
        self.convergence_max: Float = convergence_max
        self.convergence_mean: Float = convergence_mean
        self.soft_max: bool = soft_max
        self.plasticity_type_ei: str = plasticity_type_ei.name \
            if isinstance(plasticity_type_ei, PlasticityTypeItoE) else plasticity_type_ei
        self.plasticity_type_ie: str = plasticity_type_ie.name \
            if isinstance(plasticity_type_ie, PlasticityTypeEtoI) else plasticity_type_ie


class PopulationInput(SimParams):
    def __init__(
            self,
            bg_input: Float = 0.0,
            peak_stimulus: Float = 0.0
    ):
        super().__init__()
        self.bg_input: Float = bg_input
        self.peak_stimulus: Float = peak_stimulus


class Inputs(SimParams):
    def __init__(
        self,
        n_stimuli: int,
        sharp_input: bool,
        regular_pref: bool,
        exc: Union[PopulationInput, Dict],
        vonmises_kappa: Float,
        sharp_wf: Float,
        inh: Union[PopulationInput, Dict] = PopulationInput(),
    ):
        super().__init__()
        self.n_stimuli: int = n_stimuli
        self.sharp_input: bool = sharp_input
        self.regular_pref: bool = regular_pref
        self.exc: PopulationInput = PopulationInput(**exc) if isinstance(
            exc, dict
        ) else exc
        self.vonmises_kappa: Float = vonmises_kappa
        self.sharp_wf: Float = sharp_wf
        self.inh: PopulationInput = PopulationInput(**inh) if isinstance(
            inh, dict
        ) else inh


class AllParameters(SimParams):
    def __init__(
        self,
        numint: Union[NumericalIntegration, Dict],
        ng: Union[NeuronGroups, Dict],
        sy: Union[Synapses, Dict],
        pl: Union[Plasticity, Dict],
        inp: Union[Inputs, Dict],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.numint: NumericalIntegration = NumericalIntegration(
            **numint
        ) if isinstance(numint, dict) else numint
        self.ng: NeuronGroups = NeuronGroups(**ng) if isinstance(ng, dict) else ng
        self.sy: Synapses = Synapses(**sy) if isinstance(sy, dict) else sy
        self.pl: Plasticity = Plasticity(**pl) if isinstance(pl, dict) else pl
        self.inp: Inputs = Inputs(**inp) if isinstance(inp, dict) else inp

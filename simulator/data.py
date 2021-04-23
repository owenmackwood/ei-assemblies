import tables
from contextlib import contextmanager
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from collections import namedtuple
from typing import Dict, Tuple, Optional, Union, List
from collections.abc import Iterable
from .params import SimResults, Scalar

mmap_array = namedtuple('mmap_array', ['identifier', 'dtype', 'shape', 'filename', 'T'])

VLArray = Union[List[np.ndarray], Dict[int, np.ndarray]]


class DataHandler:
    def __init__(self, h5f: tables.File, log_info, log_err):
        self.h5f = h5f
        self.filters = tables.Filters(complevel=5, complib='zlib')
        self.log_info = log_info
        self.log_err = log_err

    def __del__(self):
        if self.h5f and self.h5f.isopen:
            self.h5f.close()

    def read_data_root(self):
        return self._read_node(self.h5f.root, None)

    def store_data_root(self, all_data: SimResults) -> None:
        return self.store_data(self.h5f.root, all_data)

    @staticmethod
    def _maps_int_to_ndarray(data: dict) -> bool:
        """
        Checks whether data should be stored as a VLArray.
        :param data: dict
                If this is a mapping from integers to ndarrays, then it will be stored as a VLArray
        :return:
        """
        if len(data) == 0:
            return False
        return all(isinstance(k, (int, np.integer)) and isinstance(v, Iterable)
                   for k, v in data.items())

    def store_data(self, group: tables.Group, all_data: SimResults, overwrite=False) -> None:

        # If overwrite is enabled, we want to provide a list
        # of keys that should be deleted. This means any key
        # that maps to a non-dictionary (e.g. an array), or a
        # dictionary that stores a VLArray.
        if overwrite:
            to_delete = [k for k, v in all_data.items()
                         if not isinstance(v, dict) or DataHandler._maps_int_to_ndarray(v)]
            node: tables.Node
            for node in group:
                name = getattr(node, "_v_name")
                if name in to_delete:
                    self.log_info(f'!!! OVERWRITING {name}', self.h5f)
                    self.h5f.remove_node(node, recursive=True)

        for name, value in all_data.items():
            if isinstance(value, list) or (isinstance(value, dict)
                                           and DataHandler._maps_int_to_ndarray(value)):
                self._create_vlarray(group, name, value)
            elif isinstance(value, dict):
                subgroup = self._single_get_or_create_group(group, name)
                self.store_data(subgroup, value, overwrite)
            elif sparse.issparse(value):
                self._store_sparse(group, name, value)
            elif isinstance(value, np.ndarray):
                self._create_carray(group, name, value)
            elif isinstance(value, mmap_array):
                try:
                    mmap = np.memmap(value.filename, value.dtype, 'r', shape=value.shape)
                    if value.T:
                        mmap = mmap.T
                    self._create_carray(group, name, mmap, True)
                except FileNotFoundError as e:
                    self.log_err(self.h5f, e)
                except OSError as e:
                    self.log_err(self.h5f, e)
            elif isinstance(value, bytes):
                self.h5f.create_array(group, name, value)
            elif isinstance(value, str):
                self.h5f.create_array(group, name, value.encode())
            elif not isinstance(value, Iterable):
                self.h5f.set_node_attr(group, name, value)
            else:
                self.log_info(f'UNKNOWN TYPE IN DATA {name} {type(value)}', self.h5f)

    def _single_get_or_create_group(self, parent: tables.Group, name: str) -> tables.Group:
        """
        It's necessary to have both this function and the below because if
        we combine them, the todelete list would not work correctly since
        names would have to be unique across all layers of the hierarchy.
        """
        try:
            group = self.h5f.get_node(parent, name)
        except tables.NoSuchNodeError:
            group = self.h5f.create_group(parent, name)
        # else:
        #     if overwrite:
        #         ident = '/'.join((parent._v_name, name))
        #         self.log_info('!!! OVERWRITING group '+ident)
        #         for node in group:
        #             if node._v_name in todelete:
        #                 node._f_remove(recursive=True)
        return group

    def _nested_get_or_create_groups(self, parent: tables.Group, path: str) -> tables.Group:
        for name in path.split('/'):
            try:
                parent = parent[name]
            except tables.NoSuchNodeError:
                parent = self.h5f.create_group(parent, name)
        return parent

    def _store_sparse(self, group: tables.Group, name: str, arr: csr_matrix) -> None:
        if not sparse.isspmatrix_csr(arr):
            arr = arr.tocsr()

        csr_group = self.h5f.create_group(group, name)
        csr_group.was_sparse = True

        if arr is not None and arr.nnz > 0:
            self.h5f.create_array(csr_group, 'data',   arr.data)
            self.h5f.create_array(csr_group, 'indptr', arr.indptr)
            self.h5f.create_array(csr_group, 'indices', arr.indices)
            self.h5f.create_array(csr_group, 'shape',  arr.shape)
        self.h5f.flush()

    def _create_carray(self, group: tables.Group, name: str, data: np.ndarray, mmap: bool = False) -> None:
        try:
            atom = tables.Atom.from_dtype(data.dtype)
            arr = self.h5f.create_carray(group, name, atom, data.shape, filters=self.filters)
            self.h5f.set_node_attr(arr, "was_mmap", mmap)
            arr[...] = data[...]
            self.h5f.flush()
        # except tables.NodeError as e:
        #     self.log_err(self.h5f, 'EXCEPTION: {} {} {}'.format(name, np.ndim(data), e.args))
        except Exception as e:
            self.log_err(self.h5f, f'EXCEPTION: {name} {np.ndim(data)} {e.args}')

    def _create_vlarray(self, group: tables.Group, name: str, data: VLArray) -> None:
        assert len(data), "VLArray must have at least one element"

        was_dict = isinstance(data, dict)
        if was_dict:
            data = [data[i] for i in sorted(data.keys())]

        types = set(type(v) for v in data)
        assert len(types) == 1, f"More than one type found in VLArray {name}: {types}"

        if str in types:
            atom = tables.VLUnicodeAtom()
        elif bytes in types:
            atom = tables.VLStringAtom()
        else:
            data = [v if isinstance(v, np.ndarray) else np.asarray(v) for v in data]
            dtypes = set(v.dtype for v in data)
            assert len(dtypes) == 1, f"More than one dtype found in VLArray {name}: {dtypes}"
            atom = tables.Atom.from_dtype(dtypes.pop())

        _d: tables.VLArray = self.h5f.create_vlarray(group, name, atom, filters=self.filters)
        for v in data:
            _d.append(v)
        _d.set_attr("was_dict", was_dict)
        self.h5f.flush()

    def _read_node(
            self, node: tables.Node, key: Optional[Tuple[Union[int, slice, type(...)], ...]] = None
    ) -> Union[SimResults, np.ndarray, VLArray, tables.CArray, tables.Array]:
        """
        :param node:
        :param key: Tuple. Numpy-style fancy index for an array. e.g. (5, ellipsis, slice(3, -1, 2))
        :return:
        """
        if isinstance(node, tables.Group):
            data = self._read_group(node)
        elif isinstance(node, tables.VLArray):
            data = self._read_VLArray(node, key)
        else:  # for tables.CArray and tables.Array
            assert isinstance(node, (tables.Array, tables.CArray))
            if key is not None:
                data = node[key]
            else:
                try:
                    if self.h5f.get_node_attr(node, "was_mmap"):
                        data = node
                    else:
                        data = node.read()
                except ValueError as e:
                    print(f"ERROR reading node {node}, {e}")
                    data = np.zeros(1)
        return data

    def _read_scalars(self, node: tables.Node) -> Dict[str, Scalar]:
        attr_names = getattr(getattr(node, "_v_attrs"), "_f_list")()
        return {name: self.h5f.get_node_attr(node, name)
                for name in attr_names if name not in ("was_mmap", "was_sparse", "was_dict", )}

    def _read_group(self, group: tables.Group) -> SimResults:
        try:
            self.h5f.get_node_attr(group, "was_sparse")
            data = self._read_sparse(group)
        except AttributeError:
            data = {getattr(node, "_v_name"): self._read_node(node, None)
                    for node in group}
            scalars = self._read_scalars(group)
            data.update(scalars)
        return data

    def _read_VLArray(
            self, vlarray: tables.VLArray, key: Optional[int] = None
    ) -> Union[VLArray, np.ndarray]:
        """
        :param vlarray: hdf5 node that is a VLArray
        :param key: Row index to retrieve. Cannot be a slice / tuple.
        :return: Return type depends on whether key is None, and the value of asdictionary.
        """
        if key is None:
            data = vlarray.read()

            if vlarray.get_attr("was_dict"):
                data = {i: values for i, values in enumerate(data)}
        else:
            data = vlarray.__getitem__(key)
        return data

    @staticmethod
    def _read_sparse(group: tables.Group) -> csr_matrix:
        data = group.data.read()
        indices = group.indices.read()
        indptr = group.indptr.read()
        shape = group.shape.read()
        arr = csr_matrix((data, indices, indptr), shape=shape)
        return arr


@contextmanager
def open_data_file(filename, mode='a') -> DataHandler:
    h5f = tables.open_file(filename, mode=mode)
    handler = DataHandler(h5f, print, print)
    try:
        yield handler
    finally:
        handler.h5f.close()

from numba.core.types.npytypes import Array
from numba.core import types
from numba.core.datamodel.models import StructModel
import numpy as np
from llvmlite import ir

class DPPYArray(Array):
    """
    Type class for DPPY arrays.
    """

    def __init__(self, dtype, ndim, layout, py_type=np.ndarray, readonly=False, name=None,
                 aligned=True, addrspace=None):
        self.addrspace = addrspace
        super(DPPYArray, self).__init__(dtype, ndim, layout, py_type=py_type,
              readonly=readonly, name=name, aligned=aligned)

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None, addrspace=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        if addrspace is None:
            addrspace = self.addrspace
        return DPPYArray(dtype=dtype, ndim=ndim, layout=layout, readonly=readonly,
                         aligned=self.aligned, addrspace=addrspace)

    @property
    def key(self):
        return self.dtype, self.ndim, self.layout, self.mutable, self.aligned, self.addrspace

    def is_precise(self):
        return self.dtype.is_precise()


class DPPYArrayModel(StructModel):
    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            ('meminfo', types.MemInfoPointer(fe_type.dtype)),
            ('parent', types.pyobject),
            ('nitems', types.intp),
            ('itemsize', types.intp),
            ('data', types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace)),
            ('shape', types.UniTuple(types.intp, ndim)),
            ('strides', types.UniTuple(types.intp, ndim)),

        ]
        super(DPPYArrayModel, self).__init__(dmm, fe_type, members)


    def from_argument(self, builder, value):
        print("Calling our from argument")
        methname = "from_argument"
        struct = ir.Constant(self.get_value_type(), ir.Undefined)

        for i, (dm, val) in enumerate(zip(self._models, value)):
            v = getattr(dm, methname)(builder, val)
            struct = self.set(builder, struct, v, i)

        return struct



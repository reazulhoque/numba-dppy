# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module provides utilities to interact with USM memory."""

import numpy as np
import dpctl
import dpctl.memory as dpctl_mem

from numba_dppy import config

supported_numpy_dtype = [
    np.int32,
    np.int64,
    np.uint32,
    np.int64,
    np.float32,
    np.float64,
]


def has_usm_memory(obj):
    """
    Determine and return a SYCL device accesible object.

    as_usm_memory() converts Python object with `__sycl_usm_array_interface__`
    property to one of :class:`.MemoryUSMShared`, :class:`.MemoryUSMDevice`, or
    :class:`.MemoryUSMHost` instances. For more information please refer:
    https://github.com/IntelPython/dpctl/blob/0.8.0/dpctl/memory/_memory.pyx#L673

    Args:
        obj: Object to be tested and data copied from.

    Returns:
        obj: USM backed object if object is USM backed, None otherwise.
    """
    usm_mem = None
    try:
        usm_mem = dpctl_mem.as_usm_memory(obj)
    except Exception as e:
        if hasattr(obj, "base"):
            try:
                usm_mem = dpctl_mem.as_usm_memory(obj.base)
            except Exception as e:
                if config.DEBUG:
                    print(e)
        else:
            if config.DEBUG:
                print(e)

    return usm_mem


def copy_from_numpy_to_usm_obj(usm_backed, obj):
    """
    Copy from supported objects to USM backed data.

    This function copies the data of a supported Python type (only
    numpy.ndarray is supported at this point) into object that
    defines a __sycl_usm_array_interface__ attribute. For more information
    please refer to the specification of __sycl_usm_array_interface__:
    https://github.com/IntelPython/dpctl/wiki/Zero-copy-data-exchange-using-SYCL-USM#sycl-usm-array-interface

    Args:
        usm_backed: An object that should define a
            __sycl_usm_array_interface__ dictionary. A TypeError is thrown
            if the object does not have such an attribute.
        obj (numpy.ndarray): Numpy ndarray, the data will be copied into.

    Return:
        obj: The original numpy.ndarray.
        packed_obj: If the original numpy.ndarray is not C-contiguous
            this represents the new packed numpy.ndarray.
        packed: Flag to indicate packing of strided array into a
            contiguous array.

    Raises:
        TypeError: If any argument is not of permitted type.
        ValueError: If size of data does not match.
    """
    usm_mem = has_usm_memory(usm_backed)
    if usm_mem is None:
        raise TypeError("Source is not USM backed.")

    if not isinstance(obj, np.ndarray):
        raise TypeError(
            "Obj is not USM backed and is not of type "
            "numpy.ndarray. Obj type: %s" % (type(obj))
        )

    if obj.dtype not in [np.dtype(typ) for typ in supported_numpy_dtype]:
        raise ValueError(
            "dtype is not supprted. Supported dtypes "
            "are: %s" % (supported_numpy_dtype)
        )

    packed = False
    if not obj.flags.c_contiguous:
        packed_obj = obj.flatten(order="C")
        packed = True
    else:
        packed_obj = obj

    size = np.prod(packed_obj.shape)
    if usm_mem.size != (packed_obj.dtype.itemsize * size):
        raise ValueError(
            "Size (Bytes) of data does not match. USM backed "
            "memory size %d, supported object size: %d"
            % (usm_mem.size, (packed_obj.dtype.itemsize * size))
        )

    obj_memview = memoryview(packed_obj)
    obj_memview = obj_memview.cast("B")
    usm_mem.copy_from_host(obj_memview)

    return (obj, packed_obj, packed)


def copy_to_numpy_from_usm_obj(usm_backed, obj):
    """
    Copy from USM backed data to supported objects.

    Args:
        usm_backed: An object that should define a
            __sycl_usm_array_interface__ dictionary. A TypeError is thrown
            if the object does not have such an attribute.
        obj (numpy.ndarray): Numpy ndarray, the data will be copied into.


    Raises:
        TypeError: If any argument is not of permitted type.
        ValueError: If size of data does not match.
    """
    usm_mem = has_usm_memory(usm_backed)
    if usm_mem is None:
        raise TypeError("Source is not USM backed.")

    if not isinstance(obj, np.ndarray):
        raise TypeError(
            "Obj is not USM backed and is not of type "
            "numpy.ndarray. Obj type: %s" % (type(obj))
        )

    if obj.dtype not in [np.dtype(typ) for typ in supported_numpy_dtype]:
        raise ValueError(
            "dtype is not supprted. Supported dtypes "
            "are: %s" % (supported_numpy_dtype)
        )

    size = np.prod(obj.shape)
    if usm_mem.size != (obj.dtype.itemsize * size):
        raise ValueError(
            "Size (Bytes) of data does not match. USM backed "
            "memory size %d, supported object size: %d"
            % (usm_mem.size, (obj.dtype.itemsize * size))
        )

    obj_memview = memoryview(obj)
    obj_memview = obj_memview.cast("B")
    usm_mem.copy_to_host(obj_memview)


def as_usm_backed(obj, queue=None, usm_type="shared", copy=True):
    """
    Determine and return a SYCL device accesible object.

    We try to determine if the provided object defines a dictionary called
    sycl_usm_array_interface that conforms to __sycl_usm_array_interface__.
    If not, we create a USM memory of `usm_type` and try to copy the data
    `obj` holds. Only numpy.ndarray is supported currently as `obj` if
    the object is not already backed by USM.

    Args:
        obj: Object to be tested and data copied from.
        usm_type: USM type used in case obj is not already backed by USM.
        queue (dpctl.SyclQueue): SYCL queue to be used to allocate USM
            memory in case obj is not already USM backed.
        copy (bool): Flag to determine if we copy data from obj.

    Returns:
        obj: USM backed memory.

    Raises:
        TypeError:
            1. If obj is not allocated on USM memory or is not of type
               numpy.ndarray, TypeError is raised.
            2. If queue is not of type dpctl.SyclQueue.
        ValueError:
            1. In case obj is not USM backed, users need to pass
               the SYCL queue to be used for creating new memory. ValuieError
               is raised if queue argument is not provided.
            2. If usm_type is not valid.
            3. If dtype of the passed ndarray(obj) is not supported.
    """
    usm_mem = has_usm_memory(obj)

    if queue is None:
        raise ValueError(
            "Queue can not be None. Please provide the SYCL queue to be used."
        )
    if not isinstance(queue, dpctl.SyclQueue):
        raise TypeError(
            "queue has to be of dpctl.SyclQueue type. Got %s" % (type(queue))
        )

    if usm_mem is None:
        if not isinstance(obj, np.ndarray):
            raise TypeError(
                "Obj is not USM backed and is not of type "
                "numpy.ndarray. Obj type: %s" % (type(obj))
            )

        if obj.dtype not in [np.dtype(typ) for typ in supported_numpy_dtype]:
            raise ValueError(
                "dtype is not supprted. Supported dtypes "
                "are: %s" % (supported_numpy_dtype)
            )

        size = np.prod(obj.shape)
        if usm_type == "shared":
            usm_mem = dpctl_mem.MemoryUSMShared(size * obj.dtype.itemsize, queue=queue)
        elif usm_type == "device":
            usm_mem = dpctl_mem.MemoryUSMDevice(size * obj.dtype.itemsize, queue=queue)
        elif usm_type == "host":
            usm_mem = dpctl_mem.MemoryUSMHost(size * obj.dtype.itemsize, queue=queue)
        else:
            raise ValueError(
                "Supported usm_type are: 'shared', "
                "'device' and 'host'. Provided: %s" % (usm_type)
            )

        if copy:
            # Copy data from numpy.ndarray
            copy_from_numpy_to_usm_obj(usm_mem, obj)

    return usm_mem

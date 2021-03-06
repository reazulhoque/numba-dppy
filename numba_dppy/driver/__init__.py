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

"""
The ``driver`` module implements Numba's interface to the dpctl library, a
lightweight Python and C wrapper to a subset of DPC++'s runtime API. The 
``driver`` module includes:

- LLVM IR builders for dpctl C API functions to be called directly from a Numba
  generated LLVM module.
- Functions to lauch kernels on the dpctl "current queue".

"""
from numba_dppy.driver.dpctl_capi_fn_builder import DpctlCAPIFnBuilder
from numba_dppy.driver.kernel_launch_ops import KernelLaunchOps
from numba_dppy.driver.usm_ndarray_type import USMNdArrayType

__all__ = [
    DpctlCAPIFnBuilder,
    KernelLaunchOps,
    USMNdArrayType,
]

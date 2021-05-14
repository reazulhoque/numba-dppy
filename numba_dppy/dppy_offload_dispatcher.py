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

from numba.core import dispatcher, compiler
from numba.core.registry import cpu_target
from numba.core.extending_hardware import dispatcher_registry, hardware_registry
import numba_dppy.config as dppy_config
from numba_dppy.target import SyclDevice


class DppyOffloadDispatcher(dispatcher.Dispatcher):
    targetdescr = cpu_target

    def __init__(
        self,
        py_func,
        locals={},
        targetoptions={},
        impl_kind="direct",
        pipeline_class=compiler.Compiler,
    ):
        if dppy_config.dppy_present:
            from numba_dppy.compiler import DPPYCompiler

            targetoptions["parallel"] = True
            dispatcher.Dispatcher.__init__(
                self,
                py_func,
                locals=locals,
                targetoptions=targetoptions,
                impl_kind=impl_kind,
                pipeline_class=DPPYCompiler,
            )
        else:
            print(
                "---------------------------------------------------------------------"
            )
            print(
                "WARNING : DPPY pipeline ignored. Ensure OpenCL drivers are installed."
            )
            print(
                "---------------------------------------------------------------------"
            )
            dispatcher.Dispatcher.__init__(
                self,
                py_func,
                locals=locals,
                targetoptions=targetoptions,
                impl_kind=impl_kind,
                pipeline_class=pipeline_class,
            )

if '__dppy_offload_gpu__' not in hardware_registry:
    hardware_registry['__dppy_offload_gpu__'] = SyclDevice
if '__dppy_offload_cpu__' not in hardware_registry:
    hardware_registry['__dppy_offload_cpu__'] = SyclDevice

if hardware_registry["__dppy_offload_gpu__"] not in dispatcher_registry:
    dispatcher_registry[hardware_registry["__dppy_offload_gpu__"]] = DppyOffloadDispatcher
if hardware_registry["__dppy_offload_cpu__"] not in dispatcher_registry:
    dispatcher_registry[hardware_registry["__dppy_offload_cpu__"]] = DppyOffloadDispatcher

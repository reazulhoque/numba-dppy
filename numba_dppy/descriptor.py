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

import contextlib
from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions
from numba.core.compiler_lock import global_compiler_lock

from numba.core import dispatcher, utils, typing
from .target import (DPPYTargetContext,
                     DPPYTypingContext,
                     DPPYCpuTargetContext)

from numba.core.cpu import CPUTargetOptions


class _NestedContext(object):
    _typing_context = None
    _target_context = None

    @contextlib.contextmanager
    def nested(self, typing_context, target_context):
        old_nested = self._typing_context, self._target_context
        try:
            self._typing_context = typing_context
            self._target_context = target_context
            yield
        finally:
            self._typing_context, self._target_context = old_nested

class DPPYCpuTarget(TargetDescriptor):
    options = CPUTargetOptions
    _nested = _NestedContext()

    @utils.cached_property
    @global_compiler_lock
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DPPYCpuTargetContext(self.typing_context)

    @utils.cached_property
    @global_compiler_lock
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return typing.Context()

    @property
    @global_compiler_lock
    def target_context(self):
        """
        The target context for CPU targets.
        """
        nested = self._nested._target_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_target_context

    @property
    @global_compiler_lock
    def typing_context(self):
        """
        The typing context for CPU targets.
        """
        nested = self._nested._typing_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_typing_context

    @global_compiler_lock
    def nested_context(self, typing_context, target_context):
        """
        A context manager temporarily replacing the contexts with the
        given ones, for the current thread of execution.
        """
        return self._nested.nested(typing_context, target_context)


class DPPYTarget(TargetDescriptor):
    options = CPUTargetOptions
    # typingctx = DPPYTypingContext()
    # targetctx = DPPYTargetContext(typingctx)

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DPPYTargetContext(self.typing_context)

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return DPPYTypingContext()

    @property
    def target_context(self):
        """
        The target context for DPPY targets.
        """
        return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for DPPY targets.
        """
        return self._toplevel_typing_context


# The global DPPY target
dppy_target = DPPYTarget()
dppy_cpu_target = DPPYCpuTarget()

""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""Simulation using DeepMind Control Suite."""

import copy
import logging
import re
from typing import Any

import myosuite.utils.import_utils as import_utils
from myosuite.utils.prompt_utils import prompt, Prompt
import_utils.dm_control_isavailable()
import_utils.mujoco_isavailable()
import dm_control.mujoco as dm_mujoco
try:
    import numpy as _np
    from dm_control.mujoco import index as _dm_index

    if not getattr(_dm_index, "_myosuite_struct_patch", False):
        _orig_struct_indexer = _dm_index.struct_indexer

        def _patched_struct_indexer(struct, *args, **kwargs):
            try:
                return _orig_struct_indexer(struct, *args, **kwargs)
            except AttributeError as exc:
                if "light_directional" not in str(exc):
                    raise

                class _Proxy:
                    def __init__(self, obj):
                        self._obj = obj
                        self.light_directional = _np.zeros(
                            getattr(obj, "nlight", 0), dtype=_np.int32
                        )

                    def __getattr__(self, name):
                        return getattr(self._obj, name)

                return _orig_struct_indexer(_Proxy(struct), *args, **kwargs)

        _dm_index.struct_indexer = _patched_struct_indexer
        _dm_index._myosuite_struct_patch = True
except Exception:
    pass

from myosuite.renderer.mj_renderer import MJRenderer
from myosuite.physics.sim_scene import SimScene

_DM_STRUCT_INDEXER_PATCHED = False


def _extract_missing_attr(exc: AttributeError):
    match = re.search(r"has no attribute '([^']+)'", str(exc))
    return match.group(1) if match else None


def _patch_dm_control_struct_indexer_compat():
    """Patch dm_control indexer to tolerate mujoco/dm_control field drift.

    Some Mujoco + dm_control version combinations disagree on mjModel/mjData
    field names (for example `light_directional` or `C_colind`). dm_control
    builds a static field map and crashes on missing fields. For robustness,
    filter unavailable fields and retry once.
    """
    global _DM_STRUCT_INDEXER_PATCHED
    if _DM_STRUCT_INDEXER_PATCHED:
        return

    from dm_control.mujoco import index as dm_index

    original = dm_index.struct_indexer
    if getattr(original, "_myosuite_compat_patch", False):
        _DM_STRUCT_INDEXER_PATCHED = True
        return

    warned_missing = set()

    def _patched_struct_indexer(struct, *args, **kwargs):
        try:
            return original(struct, *args, **kwargs)
        except AttributeError as exc:
            missing_attr = _extract_missing_attr(exc)
            axis_indexers = kwargs.get("axis_indexers")
            if axis_indexers is None and len(args) >= 2:
                axis_indexers = args[1]
            if not missing_attr or not isinstance(axis_indexers, dict):
                raise
            if missing_attr not in axis_indexers:
                raise

            filtered = {
                name: indexer
                for name, indexer in axis_indexers.items()
                if hasattr(struct, name)
            }
            if len(filtered) == len(axis_indexers):
                raise

            missing_fields = tuple(sorted(set(axis_indexers) - set(filtered)))
            unseen = [name for name in missing_fields if name not in warned_missing]
            if unseen:
                warned_missing.update(unseen)
                logging.warning(
                    "dm_control index compatibility: ignoring unsupported "
                    "field(s) on %s: %s",
                    type(struct).__name__,
                    ", ".join(unseen),
                )

            if len(args) >= 2:
                retry_args = list(args)
                retry_args[1] = filtered
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("axis_indexers", None)
                return original(struct, *retry_args, **retry_kwargs)

            retry_kwargs = dict(kwargs)
            retry_kwargs["axis_indexers"] = filtered
            return original(struct, *args, **retry_kwargs)

    _patched_struct_indexer._myosuite_compat_patch = True
    dm_index.struct_indexer = _patched_struct_indexer
    _DM_STRUCT_INDEXER_PATCHED = True


class DMSimScene(SimScene):
    """Encapsulates a MuJoCo robotics simulation using dm_control."""

    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle.

        Args:
            model_handle: This can be a path to a Mujoco XML file, or an MJCF
                object.

        Returns:
            A dm_control Physics object.
        """
        _patch_dm_control_struct_indexer_compat()
        if isinstance(model_handle, str):
            if model_handle.endswith('.xml'):
                sim = dm_mujoco.Physics.from_xml_path(model_handle)
            elif isinstance(model_handle, str) and "<mujoco" in model_handle:
                sim = dm_mujoco.Physics.from_xml_string(model_handle)
            else:
                sim = dm_mujoco.Physics.from_binary_path(model_handle)
        else:
            raise NotImplementedError(model_handle)
        self._patch_mjmodel_accessors(sim.model)
        self._patch_mjdata_accessors(sim.data)
        return sim

    def advance(self, substeps: int = 1, render:bool = True):
        """Advances the simulation for one step."""
        # Step the simulation substeps (frame_skip) times.
        try:
            self.sim.step(substeps)
        except:
            prompt("Simulation couldn't be stepped as intended. Issuing a reset", type=Prompt.WARN)
            self.sim.reset()

        if render:
            # self.renderer.refresh_window()
            self.renderer.render_to_window()

    def _create_renderer(self, sim: Any) -> MJRenderer:
        """Creates a renderer for the given simulation."""
        return MJRenderer(sim)

    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""
        # dm_control's MjModel defines __copy__.
        model_copy = copy.copy(self.model)
        self._patch_mjmodel_accessors(model_copy)
        return model_copy

    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """
        if not path.endswith('.mjb'):
            path = path + '.mjb'
        self.model.save_binary(path)
        return path

    def upload_height_field(self, hfield_id: int):
        """Uploads the height field to the rendering context."""
        if not self.sim.contexts:
            logging.warning('No rendering context; not uploading height field.')
            return
        with self.sim.contexts.gl.make_current() as ctx:
            ctx.call(self.get_mjlib().mjr_uploadHField, self.model.ptr,
                     self.sim.contexts.mujoco.ptr, hfield_id)

    def get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""
        return dm_mujoco.wrapper.mjbindings.mjlib

    def get_handle(self, value: Any) -> Any:
        """Returns a handle that can be passed to mjlib methods."""
        return value.ptr

    def _patch_mjmodel_accessors(self, model):
        """Adds accessors to MjModel objects to support mujoco_py API.

        This adds `*_name2id` methods to a Physics object to have API
        consistency with mujoco_py.

        TODO(michaelahn): Deprecate this in favor of dm_control's named methods.
        """
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(model.ptr,
                                      mjlib.mju_str2Type(type_name.encode()),
                                      name.encode())
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(
                    type_name, name))
            return obj_id

        def get_xml():
            from tempfile import TemporaryDirectory
            import os
            with TemporaryDirectory() as td:
                filename = os.path.join(td, 'model.xml')
                ret = mjlib.mj_saveLastXML(filename.encode(), model.ptr)
                if ret == 0:
                    raise Exception('Failed to save XML')
                with open(filename, 'r') as f:
                    return f.read()

        if not hasattr(model, 'body_name2id'):
            model.body_name2id = lambda name: name2id('body', name)

        if not hasattr(model, 'geom_name2id'):
            model.geom_name2id = lambda name: name2id('geom', name)

        if not hasattr(model, 'site_name2id'):
            model.site_name2id = lambda name: name2id('site', name)

        if not hasattr(model, 'joint_name2id'):
            model.joint_name2id = lambda name: name2id('joint', name)

        if not hasattr(model, 'actuator_name2id'):
            model.actuator_name2id = lambda name: name2id('actuator', name)

        if not hasattr(model, 'camera_name2id'):
            model.camera_name2id = lambda name: name2id('camera', name)

        if not hasattr(model, 'sensor_name2id'):
            model.sensor_name2id = lambda name: name2id('sensor', name)

        if not hasattr(model, 'get_xml'):

            model.get_xml = lambda : get_xml()


    def _patch_mjdata_accessors(self, data):
        """Adds accessors to MjData objects to support mujoco_py API."""
        if not hasattr(data, 'body_xpos'):
            data.body_xpos = data.xpos

        if not hasattr(data, 'body_xquat'):
            data.body_xquat = data.xquat

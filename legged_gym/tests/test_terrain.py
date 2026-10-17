# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""Regression tests for Terrain.selected_terrain (issue #37).

Importing any legged_gym module requires isaacgym and rsl_rl, which are only
distributed for Linux. On platforms where they are unavailable, minimal
stand-ins are installed in sys.modules first; the code under test
(legged_gym/utils/terrain.py) is always the real implementation.

Run directly:  python legged_gym/tests/test_terrain.py
Or with pytest (if installed):  pytest legged_gym/tests/test_terrain.py
"""

import os
import sys
import types

import numpy as np

# allow running this file directly from a source checkout
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _install_missing_dep_stubs():
    """Stub isaacgym/rsl_rl only when they cannot be imported.

    gymapi/gymutil/gymtorch/torch_utils are never exercised at import time by
    legged_gym, so a module that accepts any attribute is enough for them.
    terrain_utils needs a functional SubTerrain and pyramid_sloped_terrain
    because Terrain.selected_terrain() really calls them.
    """
    try:
        import isaacgym  # noqa: F401
    except ImportError:
        def _permissive(name, package=False):
            module = types.ModuleType(name)
            # keep standard module attributes real so inspect/pickle keep working
            module.__file__ = name.replace(".", "/") + ".py"
            module.__spec__ = None
            if package:
                module.__path__ = []
            def _getattr(attr, _name=name):
                if attr.startswith("__") and attr.endswith("__"):
                    raise AttributeError(attr)
                return type(_name + "." + attr, (), {"__call__": lambda self, *a, **k: None})()
            module.__dict__["__getattr__"] = _getattr
            return module

        sys.modules.setdefault("isaacgym", _permissive("isaacgym", package=True))
        for sub in ("gymapi", "gymutil", "gymtorch", "torch_utils"):
            sys.modules.setdefault("isaacgym." + sub, _permissive("isaacgym." + sub))
            setattr(sys.modules["isaacgym"], sub, sys.modules["isaacgym." + sub])

        class SubTerrain:
            def __init__(self, profile_name="terrain", width=256, length=256,
                         vertical_scale=1.0, horizontal_scale=1.0):
                self.profile_name = profile_name
                self.width = width
                self.length = length
                self.vertical_scale = vertical_scale
                self.horizontal_scale = horizontal_scale
                self.height_field_raw = np.zeros((self.width, self.length))

        def pyramid_sloped_terrain(terrain, slope=1.0, platform_size=3.0):
            x = np.arange(terrain.length)
            y = np.arange(terrain.width)
            center = terrain.length / 2
            x = np.abs(center - x) * terrain.horizontal_scale * slope
            center = terrain.width / 2
            y = np.abs(center - y) * terrain.horizontal_scale * slope
            xx, yy = np.meshgrid(x, y)
            heights = np.maximum(xx, yy)
            half = int(platform_size / terrain.horizontal_scale / 2)
            x1, x2 = terrain.length // 2 - half, terrain.length // 2 + half
            y1, y2 = terrain.width // 2 - half, terrain.width // 2 + half
            heights[x1:x2, y1:y2] = 0.
            terrain.height_field_raw += (heights / terrain.vertical_scale).astype(np.int16)

        terrain_utils = _permissive("isaacgym.terrain_utils")
        terrain_utils.SubTerrain = SubTerrain
        terrain_utils.pyramid_sloped_terrain = pyramid_sloped_terrain
        sys.modules["isaacgym.terrain_utils"] = terrain_utils
        setattr(sys.modules["isaacgym"], "terrain_utils", terrain_utils)


    try:
        import rsl_rl  # noqa: F401
    except ImportError:
        env_module = types.ModuleType("rsl_rl.env")
        env_module.VecEnv = type("VecEnv", (), {})
        runners_module = types.ModuleType("rsl_rl.runners")
        runners_module.OnPolicyRunner = type("OnPolicyRunner", (), {"__init__": lambda self, *a, **k: None})
        rsl_module = types.ModuleType("rsl_rl")
        rsl_module.env = env_module
        rsl_module.runners = runners_module
        sys.modules.setdefault("rsl_rl", rsl_module)
        sys.modules.setdefault("rsl_rl.env", env_module)
        sys.modules.setdefault("rsl_rl.runners", runners_module)


_install_missing_dep_stubs()

# legged_gym has a benign circular import (legged_gym.utils.task_registry asks for
# legged_gym.envs.base.legged_robot_config, while legged_gym.envs/__init__.py asks
# for legged_gym.utils.task_registry). Entering through legged_gym.envs first --
# exactly what legged_gym/scripts/train.py and play.py do -- resolves it
# deterministically on every platform.
import legged_gym.envs.base.legged_robot_config  # noqa: F401, E402


def _make_selected_terrain_cfg(slope=1.0):
    from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

    cfg = LeggedRobotCfg.terrain()
    cfg.mesh_type = 'heightfield'
    cfg.curriculum = False
    cfg.selected = True
    cfg.horizontal_scale = 0.1
    cfg.vertical_scale = 0.005
    cfg.border_size = 0
    cfg.terrain_length = 8.
    cfg.terrain_width = 8.
    cfg.num_rows = 1
    cfg.num_cols = 1
    # 'type' is eval()ed in legged_gym/utils/terrain.py's module namespace, so it
    # must be a dotted name (terrain_utils.X) or a module-level function
    cfg.terrain_kwargs = dict(type='terrain_utils.pyramid_sloped_terrain', slope=slope)
    return cfg


def test_selected_terrain_uses_cfg_scales_and_kwargs():
    """selected=True must build the terrain from cfg scales + terrain_kwargs.

    Regression test for issue #37: selected_terrain() referenced the
    non-existent instance attributes self.vertical_scale /
    self.horizontal_scale and unpacked the non-existent attribute
    terrain_kwargs.terrain_kwargs, so any 'selected' terrain crashed with
    AttributeError before building a single sub-terrain.
    """
    from legged_gym.utils.terrain import Terrain

    terrain = Terrain(_make_selected_terrain_cfg(slope=1.0), num_robots=1)

    # pyramid_sloped_terrain must have been invoked with the remaining
    # terrain_kwargs (slope=1.0), producing non-zero heights
    assert np.max(np.abs(terrain.height_field_raw)) > 0


def test_selected_terrain_forwards_slope_kwarg():
    """The kwargs remaining after popping 'type' must reach the terrain fn."""
    from legged_gym.utils.terrain import Terrain

    shallow = Terrain(_make_selected_terrain_cfg(slope=0.5), num_robots=1)
    steep = Terrain(_make_selected_terrain_cfg(slope=1.0), num_robots=1)

    shallow_max = np.max(np.abs(shallow.height_field_raw))
    steep_max = np.max(np.abs(steep.height_field_raw))
    assert steep_max > shallow_max
    # pyramid height scales linearly with slope: 1.0 / 0.5
    assert abs(steep_max / shallow_max - 2.0) < 1e-6


if __name__ == "__main__":
    test_selected_terrain_uses_cfg_scales_and_kwargs()
    test_selected_terrain_forwards_slope_kwarg()
    print("test_terrain: 2 tests passed")

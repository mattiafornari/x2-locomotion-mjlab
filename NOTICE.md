# Third-party notices

* **mjlab** (Apache-2.0, https://github.com/mujocolab/mjlab) — simulation framework and the velocity
  task template; used as an installed dependency (`mjlab==1.6.1`), not copied. The task config in
  `src/x2_locomotion/tasks/velocity/env_cfgs.py` is built on top of `make_velocity_env_cfg()`.
* **MuJoCo / MuJoCo Warp** (Apache-2.0) and **NVIDIA Warp** (Apache-2.0), via mjlab.
* **rsl-rl-lib** (BSD-3-Clause, ETH Zurich / NVIDIA), via mjlab.
* **AgiBot X2 URDF/MJCF models and meshes** (Mulan Permissive Software License v2,
  https://github.com/AgibotTech/agibot_x2_urdf). Included only as a git submodule
  (`third_party/agibot_x2_urdf`, pinned commit); not copied into this repository. The MJCF is edited
  in memory at load time (`src/x2_locomotion/assets/x2.py`), nothing derived is redistributed.

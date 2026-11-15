# Running Fleet Memory on this laptop (ROS 2 Jazzy + CARLA 0.9.16)

The main `README.md` targets Ubuntu 22.04 + ROS 2 Humble + Python 3.10. This laptop
(MSI GE63, Ubuntu 24.04, RTX 2060 Mobile 6 GB + Intel UHD 630) runs **ROS 2 Jazzy +
Python 3.12**, so a few things are different. This file is the short version of
"how to start it again" plus a record of what was set up and why.

---

## Quick start (every time)

You need **two terminals**.

### Terminal 1 — CARLA server (on the NVIDIA GPU)

```bash
cd ~/CARLA
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
./CarlaUE4.sh -prefernvidia -quality-level=Low -windowed -ResX=1280 -ResY=720
```

Wait until the city (Town10HD) is visible in the CARLA window (~20–40 s).

> **Do not drop the first three lines.** Without them CARLA runs on the Intel chip:
> ~5 FPS, and it crashes when loading a map. See *Problems we hit* below.

### Terminal 2 — the Fleet Memory system

```bash
source ~/CARLA/fleet_env.sh

ros2 launch fleet_experiments fleet_memory.launch.py \
    num_cars:=3 memory_mode:=template seed:=0 rviz:=true run_name:=my_run \
    scenario:=$(ros2 pkg prefix fleet_experiments)/share/fleet_experiments/config/scenarios/town10_default.yaml
```

This starts the shared memory node, a hazard writer + memory reader per car, the
scenario runner and RViz. The camera in the CARLA window follows the active car.
The launch exits by itself when the last car finishes.

- **Results:** `~/fleet_memory_results/my_run/` (`cars.csv`, `hazards.csv`,
  `written_records.csv`, `memory_events.csv`, `trajectories/`, `config.json`).
- **Stop early:** `Ctrl+C` in terminal 2 (or `pkill -f "ros2 launch fleet_experiments"`).
- **Stop CARLA:** close its window, or `pkill -f CarlaUE4-Linux-Shipping`.

### Useful variations

| Want | Change |
|---|---|
| More cars (proposal uses 10) | `num_cars:=10` |
| No-memory baseline | `memory_mode:=none` |
| Oracle upper bound | `memory_mode:=oracle` |
| VLM text descriptions | `memory_mode:=vlm` — first `ollama pull qwen2.5vl:7b` (no vision model is installed yet) |
| No CARLA window needed (faster sweeps) | replace `-windowed -ResX=1280 -ResY=720` with `-RenderOffScreen` |
| Same route every run | the runner logs `Auto route: spawn 125 -> 21`; put that in the scenario YAML with `route.mode: spawn_points` |

### Quick health checks

```bash
# is CARLA up, on which map, how fast? (expect Town10HD_Opt, ~30 FPS)
PYTHONPATH=~/CARLA/pylib /usr/bin/python3 -c "
import carla,time; c=carla.Client('localhost',2000); c.set_timeout(20); w=c.get_world()
print(w.get_map().name); t=time.time(); n=0
while time.time()-t<3: w.wait_for_tick(); n+=1
print('FPS', n/3)"

# is it on the NVIDIA GPU? (expect CarlaUE4-Linux-Shipping using ~4 GB)
nvidia-smi

# ROS side
ros2 topic list | grep ego_
ros2 run fleet_memory memory_cli list
```

---

## What was set up (one-time, already done)

No project source code was changed. Everything below is environment/setup.

| # | What | Why |
|---|---|---|
| 1 | **Downloaded CARLA 0.9.16** to `~/CARLA` (8.3 GB, from the official `carla-releases` bucket) | 0.9.16 is the first release with a **Python 3.12** client, which ROS Jazzy needs. 0.9.15 (in `requirements.txt`) only supports ≤ 3.10. |
| 2 | **Unpacked the CARLA client wheel** into `~/CARLA/pylib` (`carla-0.9.16-cp312-…whl` from `~/CARLA/PythonAPI/carla/dist/`) | The system Python has no `pip`, and the project's `setup_carla_path()` only looks for old `.egg` files, not wheels. Putting `~/CARLA/pylib` on `PYTHONPATH` makes `import carla` work. |
| 3 | `sudo apt install python3-networkx python3-shapely` | Needed by CARLA's `agents` package (`BehaviorAgent`, `GlobalRoutePlanner`). |
| 4 | **Built the workspace in the repo**: `colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3` | The `(pytorch)` conda env puts its own `python3` first on `PATH`, which breaks the message build (`No module named 'em'`). Forcing `/usr/bin/python3` fixes it. `build/`, `install/`, `log/` are gitignored. |
| 5 | **Created `~/CARLA/fleet_env.sh`** | One `source` sets up a terminal: leaves conda, uses system Python, sets `CARLA_ROOT` and `PYTHONPATH`, sources ROS Jazzy + the workspace, `cd`s into the repo. |
| 6 | **Run CARLA on the NVIDIA GPU** (the env vars + `-prefernvidia` in the quick start) | The laptop is in PRIME `on-demand` mode, so programs default to the Intel GPU. |
| 7 | **Use the Town10 scenario** instead of the default Town03 | Switching maps (Town10HD → Town03) crashed CARLA twice — most likely out of GPU memory (Town10HD alone uses 4.3 of 6 GB). Town10HD is already loaded at startup, so the runner never switches maps. The proposal allows Town03 or Town10. |

### Contents of `~/CARLA/fleet_env.sh`

```bash
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
unset VIRTUAL_ENV
export PATH=/usr/bin:$PATH
export CARLA_ROOT=$HOME/CARLA
export PYTHONPATH=$HOME/CARLA/pylib:$PYTHONPATH
source /opt/ros/jazzy/setup.bash
source <repo>/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
cd <repo>
```

### Optional extras (not installed)

- `pip install sentence-transformers` → better text similarity in the memory (falls
  back to a hashing embedder without it; everything still works).
- A vision model for `memory_mode:=vlm`: `ollama pull qwen2.5vl:7b` (Ollama is
  already running; current models `all-minilm`, `defiant-9b`, `qwen38-turbo` are not VLMs).

---

## Problems we hit (and the fix)

| Symptom | Cause | Fix |
|---|---|---|
| `colcon build` fails: `No module named 'em'` | conda's `python3` used instead of the system one | build with `-DPython3_EXECUTABLE=/usr/bin/python3` / use `fleet_env.sh` |
| CARLA window "does nothing", ~5 FPS, `nvidia-smi` shows ~115 MB | running on the Intel GPU (PRIME on-demand) | NVIDIA offload env vars + `-prefernvidia` |
| Runner: `time-out of 30000ms while waiting for the simulator` at `load_world` | CARLA crashed while switching to Town03 | start on Town10HD and use `town10_default.yaml` |
| Map argument `/Game/Carla/Maps/Town03` on the CARLA command line is ignored | CARLA 0.9.16 still opens Town10HD | not needed with the Town10 scenario |
| Ctrl+C on CARLA leaves the old server running; a new one silently fails | port 2000 still taken | `pkill -f CarlaUE4-Linux-Shipping`, check `pgrep -fa CarlaUE4-Linux-Shipping` is empty |

---

## Open issues found in the first CARLA run (`town10_first`, 3 cars)

What **worked** end to end: route search (765 m, spawn 125 → 21), hazard spawning,
car 1 driving with `BehaviorAgent`, cone detection → near-miss → stop → bypass,
memory writes, 0 collisions, car 1 arrived in 100 s.

What **needs fixing** before experiments mean anything:

1. **Noisy acceleration → fake "jerk"/"hard brake" surprises.**
   `scenario_runner.py:736` computes `accel = (v_fwd - prev_v) / dt` every 50 ms; the
   writer then differentiates again for jerk. CARLA's autopilot jitter (even when
   standing still) becomes 25–146 m/s³ spikes. Car 1 logged **20 hard brakes and wrote
   14 memories** where ~3 were expected. Fix: smooth the speed / use the attached IMU,
   and ignore the first ~2 s after spawn.
2. **Real events get lost to the 5 s writer cooldown.** The cone near-miss came
   right after a fake jerk event, so the cones were stored as `road_damage`
   instead of `lane_closure`. Mostly solved by fixing (1).
3. **Pedestrian never stepped out** (no `walker` event near (-84.4, -130.9)). Check
   `PedestrianCrossing` in `hazards.py` (trigger distance / walker control).
4. **Pothole jolt not clearly detected** at (79.7, -134.7); re-check `jolt_gain`
   after (1).
5. **Speed-cap buffer too small for the pedestrian** (found in the 2-D demo):
   `ego_driver.py` reaches the cap only `cap_buffer_m = 5` m before the hazard, but the
   walker steps out ~18 m before it. ~20 m worked in the demo.
6. **Lane-change makes the memory refute itself** (found in the 2-D demo): a car that
   changes lane early because of a lane-closure memory never sees the cones, counts a
   "clean pass" and refutes the memory. A pass should not refute a memory the car was
   acting on (`memory_reader/feedback.py`, `PassTracker`).
7. **"Near-miss" with TTC 0.00 while bypassing the cones.** Both cars logged one at
   (9.8, -138.8) while driving *around* the cones: perception reports them in the path
   at distance 0. Not a real near-miss; ignore obstacles the car is already alongside
   or bypassing.

8. **CARLA server froze between car 2 and car 3.** ~40 s after "Car 2 done" the server
   stopped answering (process alive, 0 % CPU, no kernel/GPU errors). The runner died with
   an uncaught `carla::client::TimeoutException` (`terminate called`, exit -6), so
   `cleanup()` never ran and CARLA was left in synchronous mode. Recovery:
   `pkill -9 -f CarlaUE4-Linux-Shipping` and restart. Cars 1–2 results were saved.
   To do: catch the timeout and exit cleanly; consider `carla_timeout_s: 60`; if it
   repeats between cars, inspect the car hand-over (ego/sensor destroy, `inter_car_gap_s`
   ticks); if random, try `-RenderOffScreen`.

### First-run numbers (`town10_first`) — distorted by issue 1

| | Car 1 | Car 2 |
|---|---|---|
| Arrival time | 100.4 s | 122.2 s (+22 %) |
| Near-misses | 1/3 | 1/3 (bypass artifact, issue 7) |
| Hard brakes | 20 | 33 |
| Collisions | 0 | 0 |
| Memories written | 14 | 20 |

Car 2 obeyed ~7 fake `road_damage` memories (slowed to 15–20 km/h at each), so it was
slower and braked more. The "fleet improves" result can only appear after issue 1 is fixed.

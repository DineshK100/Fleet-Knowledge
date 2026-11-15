# Fleet Memory: Cars that Learn from Each Other

A retrieval-augmented fleet data engine for hazard-aware autonomous vehicles, written as ROS 2 Humble packages on top of the CARLA simulator.

When a car has a surprise (hard brake, near-miss, collision or a violent jolt), a vision-language model (VLM) writes a short description of the hazard. The description goes into a shared, location-indexed fleet memory. Later cars query that memory along their planned route and slow down, change lane or reroute before they reach the hazard.

```
 Car A sensors ──► surprise trigger ──► VLM event writer ──write──► shared fleet memory
 (camera, state,   (hard brake, TTC,     (JSON: type, text,         (spatial grid + vector index,
  obstacle, coll.)  collision, jerk)      lane, severity, advice)    merge, decay, prune)
                                                                             │ retrieve
 Car B route look-ahead query ──► retrieve & re-rank ──► policy adapter ◄────┘
 (pos, heading, context text)     (route distance,       (speed cap / lane change / reroute)
                                   heading, conf., recency, sim)   └── confirm / refute ──► memory
```

## Repository layout (colcon workspace)

| Package | Role |
|---|---|
| `src/fleet_msgs` | Interfaces: `HazardMemory`, `MemoryEvent`, `SurpriseEvent`, `VehicleState`, `ObstacleInfo`, `CollisionEvent`, `HazardConstraint`, `DrivingAdvice`; services `WriteMemory`, `QueryMemory`, `MemoryFeedback` |
| `src/fleet_memory` | One shared memory node (the fleet's "cloud"). Spatial grid index plus a vector index (sentence-transformers embeddings, with a hashing fallback; FAISS is used if installed). It merges duplicates, decays confidence (`c ← c·e^{-λΔt}`), handles confirm/refute and prunes records. It also publishes RViz markers. `memory_cli` lets you inspect or inject records. |
| `src/hazard_writer` | One node per car. The surprise trigger fires on decel > 0.4 g, TTC < 1.5 s, a collision, or a jerk spike. A ~3 s camera ring buffer feeds keyframes to the VLM, which runs on any OpenAI-compatible endpoint (Ollama, vLLM, …). The reply is a fixed JSON schema; if the VLM fails, a template record is used. |
| `src/memory_reader` | One node per car. Queries the memory at 2 Hz and scores each hit with `s = w_d f(d) + w_h cos(Δψ) + w_c c + w_r e^{-Δt/τ} + w_s sim`, where `d` is measured along the planned route. Keeps the top-k hits above `s_min`. The rule-based adapter (speed cap, lane change or reroute) is the main policy; an optional LLM adapter is the stretch goal. Also sends confirm/refute feedback after passing a hazard. |
| `src/fleet_experiments` | Scenario runner (CARLA client, hazards, N cars in a row, vehicle I/O, metrics), ego driver (CARLA `BehaviorAgent` plus the memory hooks), simulated perception, launch file, configs, RViz, the experiment sweep (E1–E4) and plotting. |

### Design notes / deviations from the proposal

- **Vehicle ↔ ROS interface.** The scenario runner talks to CARLA through the Python API and publishes each car's topics under `/ego_k/...` itself, instead of using the generic `carla_ros_bridge`. The risk table in the proposal lists this as the fallback ("keep a PythonAPI fallback behind the same ROS 2 interfaces"). It avoids the CARLA/ROS 2 bridge version problems and lets us drive cars with `BehaviorAgent`. All published positions are in the ROS `map` frame: CARLA coordinates with y flipped, same convention as the bridge.
- **Perception.** Obstacles (for braking, TTC and near-miss) come from CARLA ground truth, each limited by its own visibility range. Cones are seen late, the stalled car is hidden behind a curve, and the pothole can't be perceived at all. This keeps the scenarios reproducible and makes the memory the only way to know about a hazard early.
- **Hazard placement.** Hazards are not tied to hand-picked coordinates. Each scenario YAML lists hazard types with an `after_m` distance. The runner finds a route (deterministic search) on which every hazard fits its constraints: intersection approach, multi-lane road, after a curve, … It then logs the route so you can pin it.

## 1. Setup on the VM (Ubuntu 22.04)

1. **ROS 2 Humble** (desktop): <https://docs.ros.org/en/humble/Installation.html>
2. **CARLA server** (0.9.15 or 0.9.16), e.g. extracted to `~/CARLA`:
   ```bash
   export CARLA_ROOT=~/CARLA          # put this in ~/.bashrc; the runner uses it to find PythonAPI/carla/agents
   ```
3. **Python dependencies** (for the system Python 3.10 that ROS uses):
   ```bash
   sudo apt install python3-pip python3-numpy python3-yaml python3-matplotlib python3-opencv
   pip3 install -r requirements.txt   # carla client, networkx/shapely (CARLA agents), optional extras
   ```
   The `carla` pip package must match the server version (`pip3 install carla==0.9.15`, or `0.9.16`). If pip has no wheel for Python 3.10 at that version, install the `.whl`/`.egg` from `$CARLA_ROOT/PythonAPI/carla/dist`.
   `sentence-transformers` is optional. Without it the memory uses a built-in hashing embedder, and everything still works.
4. **VLM server.** The simplest option is [Ollama](https://ollama.com):
   ```bash
   ollama pull qwen2.5vl:7b        # VLM for the event writer
   ollama pull qwen2.5:7b          # text LLM for the (stretch) LLM adapter
   ```
   Any OpenAI-compatible server works. With vLLM, use `vlm_base_url:=http://localhost:8000/v1 vlm_model:=Qwen/Qwen2.5-VL-7B-Instruct`. For a hosted API, set its URL and export the key in `OPENAI_API_KEY`. Without a VLM, use `memory_mode:=template`.

## 2. Build

```bash
cd <this repo>
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 3. Run

```bash
# terminal 1: CARLA server (low quality is enough)
$CARLA_ROOT/CarlaUE4.sh -quality-level=Low        # add -RenderOffScreen on a headless VM

# terminal 2: the whole experiment with one launch command
ros2 launch fleet_experiments fleet_memory.launch.py num_cars:=10 memory_mode:=vlm seed:=0 rviz:=true
```

The launch file starts `fleet_memory`, one `hazard_writer` and one `memory_reader` per car (`/ego_1 … /ego_N`), and the `scenario_runner`, plus RViz if you ask for it. It exits when the last car finishes. Results go to `~/fleet_memory_results/<run_name>/`:

| File | Content |
|---|---|
| `cars.csv` | one row per car: near-misses, hard brakes (/km, /encounter), collisions, arrival time, comfort, unnecessary slow-downs, retrieval precision/recall, lead distance, lane changes / reroutes |
| `hazards.csv` | per car × hazard: min TTC, min distance, peak decel/jerk, speed at the hazard, retrieved?, lead distance |
| `written_records.csv` | every memory a car wrote (VLM text), matched to the ground-truth hazard → description accuracy |
| `memory_events.csv` | insert / merge / confirm / refute / prune log of the memory |
| `advice.csv` | what each car retrieved and how it acted on it |
| `trajectories/car_XX.csv` | 20 Hz trajectory: speed, accel, target / cap, maneuver, TTC |
| `config.json` | parameters, scenario, chosen route, hazard positions |

Useful launch arguments: `memory_mode:=vlm|template|oracle|none`, `adapter:=rules|llm`, `scenario:=<yaml>`, `vlm_model:=…`, `vlm_base_url:=…`, `false_memories:=5`, `record_video:=true`, `dataset_dir:=~/vlm_dataset`, `carla_host:=…`. All the tuning knobs live in `src/fleet_experiments/config/params.yaml`.

**First run on a new map.** The runner logs `Auto route: spawn i -> j`. Put those values in the scenario YAML (`route.mode: spawn_points`) so every later run uses exactly the same route. Hazard behaviour (visibility, trigger distance, pothole jolt, cruise speed) is set in the scenario YAML and `params.yaml`. Expect to tune it in the first week, as the proposal says.

### Inspecting the memory

```bash
ros2 run fleet_memory memory_cli list
ros2 run fleet_memory memory_cli query --x 120 --y -40 --radius 80 --text "pedestrian at intersection"
ros2 run fleet_memory memory_cli inject --type lane_closure --x 120 --y -40 --desc "cones in right lane"
ros2 topic echo /ego_2/driving_advice
```

## 4. Experiments (Section 4.2)

```bash
ros2 run fleet_experiments run_sweep --experiment E1 --seeds 0 1 2 --num-cars 10   # fleet learning curve
ros2 run fleet_experiments run_sweep --experiment E2 --seeds 0 1 2                 # none / oracle / template / vlm
ros2 run fleet_experiments run_sweep --experiment E3 --seeds 0 1 2                 # false + stale memories
ros2 run fleet_experiments run_sweep --experiment E4 --seeds 0 1 2                 # look-ahead, top-k, heading, decay, LLM adapter
ros2 run fleet_experiments plot_results ~/fleet_memory_results/E2 --baseline none
```

`run_sweep --dry-run` prints the commands without running them. `plot_results` writes three things to `<dir>/plots/`:
- `learning_curves.png`: each metric against car index, mean ± 95 % CI over seeds.
- `speed_profile_<config>.png`: car 1 against the last car along the route.
- `summary.md`: car 1 vs cars ≥ 2, the comparison to the baseline, a check of the proposal's *target* criteria (≥ 50 % fewer near-misses and hard brakes, < 10 % longer arrival time), VLM type accuracy and false-memory clean-up.

### Demo video

With `record_video:=true`, each car saves chase-camera frames with the retrieved memory text drawn on top:

```bash
ffmpeg -framerate 10 -i ~/fleet_memory_results/<run>/video/car_01/frame_%05d.jpg -pix_fmt yuv420p car01.mp4
ffmpeg -framerate 10 -i ~/fleet_memory_results/<run>/video/car_05/frame_%05d.jpg -pix_fmt yuv420p car05.mp4
ffmpeg -i car01.mp4 -i car05.mp4 -filter_complex hstack side_by_side.mp4
```

### Real dash-cam images (stretch)

```bash
ros2 run hazard_writer dashcam_eval --images ~/dashcam/imgs --labels ~/dashcam/labels.csv --model qwen2.5vl:7b
```

## 5. Tests

The memory store, trigger, VLM parsing, scoring, policy, feedback, metrics and hazard placement are plain Python and are unit-tested without ROS or CARLA:

```bash
colcon test --packages-select fleet_memory hazard_writer memory_reader fleet_experiments && colcon test-result --verbose
# or, without building:
cd src && PYTHONPATH=fleet_memory:hazard_writer:memory_reader:fleet_experiments python3 -m pytest fleet_*/test hazard_writer/test memory_reader/test
```

## 6. ROS interface summary

| Name | Type | Direction |
|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | runner → all (`use_sim_time`) |
| `/ego_k/vehicle_state` | `fleet_msgs/VehicleState` | runner → writer, reader |
| `/ego_k/camera/image` | `sensor_msgs/Image` (bgra8) | runner → writer |
| `/ego_k/obstacle`, `/ego_k/collision` | `fleet_msgs/ObstacleInfo`, `CollisionEvent` | runner → writer |
| `/ego_k/odometry`, `/ego_k/imu`, TF `map→ego_k/base_link` | standard | runner → RViz |
| `/ego_k/planned_route` | `nav_msgs/Path` | runner → reader |
| `/ego_k/surprise_events` | `fleet_msgs/SurpriseEvent` | writer → reader |
| `/ego_k/driving_advice` | `fleet_msgs/DrivingAdvice` | reader → runner (driver) |
| `/fleet_memory/{write,query,feedback,reset,save}` | services | writers / readers / runner |
| `/fleet_memory/events`, `/fleet_memory/markers` | `MemoryEvent`, `MarkerArray` | memory → runner, RViz |

## 7. Things to check first on the VM

- The pieces that touch CARLA (`scenario_runner.py`, `ego_driver.py`, `hazards.py`, `perception.py`) target the 0.9.13+ Python API and `agents` package. They could not be run while writing the code, so the first run is the integration test. Watch the log for spawn failures (props, van, walker) and adjust offsets in the scenario YAML.
- If the auto route search fails on a map, lower `min_length_m`, change `after_m`, or disable a hazard.
- If car 1 doesn't hard-brake or nearly miss at a hazard, tune `cruise_speed_kmh`, the hazard's `visibility_m`, the pedestrian `appear_distance_m` or the pothole `jolt_gain`. If later cars are too cautious, tune the speed caps (`memory_reader/policy.py`) and `s_min`.
- VLM latency doesn't slow the simulation. Writes happen in a background thread, and the runner waits for pending writes before the next car starts (`writer_flush_timeout_s`).

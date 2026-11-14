"""Start the whole Fleet Memory experiment with one command.

    ros2 launch fleet_experiments fleet_memory.launch.py num_cars:=10 memory_mode:=vlm seed:=0

Starts the shared fleet memory, one hazard writer and one memory reader per car
(/ego_1 ... /ego_N), the CARLA scenario runner and (optionally) RViz. The
launch shuts down when the scenario runner finishes.

memory_mode:
  vlm       VLM-written memories (main method)
  template  memories without VLM text: type + location only (baseline E2c)
  oracle    memory pre-filled with every ground-truth hazard (upper bound, E2b)
  none      memory disabled: every car is the "first car" (baseline E2a)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _bool(v: str) -> bool:
    return str(v).lower() in ('1', 'true', 'yes', 'on')


def launch_setup(context, *args, **kwargs):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    share = get_package_share_directory('fleet_experiments')
    params_file = lc('params_file') or os.path.join(share, 'config', 'params.yaml')
    scenario = lc('scenario') or os.path.join(share, 'config', 'scenarios', 'town03_default.yaml')
    n = int(lc('num_cars'))
    mode = lc('memory_mode')
    if mode not in ('vlm', 'template', 'oracle', 'none'):
        raise RuntimeError(f'unknown memory_mode {mode}')
    writer_mode = lc('writer_mode') or ('template' if mode == 'template' else 'vlm')
    adapter = 'none' if mode == 'none' else lc('adapter')
    sim = {'use_sim_time': True}

    actions = [Node(
        package='fleet_memory', executable='memory_node', name='fleet_memory', output='screen',
        parameters=[params_file, sim])]

    writer_over = {'writer_mode': writer_mode}
    if lc('vlm_base_url'):
        writer_over['vlm_base_url'] = lc('vlm_base_url')
    if lc('vlm_model'):
        writer_over['vlm_model'] = lc('vlm_model')
    if lc('dataset_dir'):
        writer_over['dataset_dir'] = lc('dataset_dir')
    reader_over = {'adapter': adapter}
    if lc('llm_base_url'):
        reader_over['llm_base_url'] = lc('llm_base_url')
    if lc('llm_model'):
        reader_over['llm_model'] = lc('llm_model')

    for k in range(1, n + 1):
        ns = f'ego_{k}'
        actions.append(Node(package='hazard_writer', executable='writer_node', name='hazard_writer',
                            namespace=ns, output='screen',
                            parameters=[params_file, sim, writer_over]))
        actions.append(Node(package='memory_reader', executable='reader_node', name='memory_reader',
                            namespace=ns, output='screen',
                            parameters=[params_file, sim, reader_over]))

    runner_over = {
        'num_cars': n, 'seed': int(lc('seed')), 'memory_mode': mode, 'scenario_file': scenario,
        'carla_host': lc('carla_host'), 'carla_port': int(lc('carla_port')),
        'record_video': _bool(lc('record_video')), 'results_dir': lc('results_dir'),
        'run_name': lc('run_name'), 'config_name': lc('config_name'),
        'false_memories': int(lc('false_memories')),
    }
    runner = Node(package='fleet_experiments', executable='scenario_runner', name='scenario_runner',
                  output='screen', parameters=[params_file, runner_over])
    actions.append(runner)
    actions.append(RegisterEventHandler(OnProcessExit(
        target_action=runner,
        on_exit=[EmitEvent(event=Shutdown(reason='scenario runner finished'))])))

    if _bool(lc('rviz')):
        actions.append(Node(package='rviz2', executable='rviz2', name='rviz2',
                            arguments=['-d', os.path.join(share, 'rviz', 'fleet_memory.rviz')],
                            parameters=[sim]))
    return actions


def generate_launch_description():
    args = [
        ('num_cars', '10', 'number of cars sent one after another'),
        ('memory_mode', 'vlm', 'vlm | template | oracle | none'),
        ('adapter', 'rules', 'policy adapter: rules | llm'),
        ('writer_mode', '', 'override hazard writer mode: vlm | template | off'),
        ('seed', '0', 'random seed (weather, pedestrian timing, traffic)'),
        ('scenario', '', 'scenario YAML (default: config/scenarios/town03_default.yaml)'),
        ('params_file', '', 'parameter YAML (default: config/params.yaml)'),
        ('carla_host', 'localhost', 'CARLA server host'),
        ('carla_port', '2000', 'CARLA server port'),
        ('vlm_base_url', '', 'OpenAI-compatible endpoint of the VLM (e.g. Ollama / vLLM)'),
        ('vlm_model', '', 'VLM model name'),
        ('llm_base_url', '', 'OpenAI-compatible endpoint of the LLM adapter'),
        ('llm_model', '', 'LLM model name for the LLM adapter'),
        ('results_dir', '~/fleet_memory_results', 'where results are written'),
        ('run_name', '', 'results sub-directory (default: timestamp_mode_seed)'),
        ('config_name', '', 'label used when plotting (default: memory_mode)'),
        ('false_memories', '0', 'inject N false memories before car 1 (E3)'),
        ('dataset_dir', '', 'save VLM keyframes + records here'),
        ('record_video', 'false', 'save chase-camera frames with memory overlay'),
        ('rviz', 'false', 'start RViz'),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=desc) for n, d, desc in args] +
        [OpaqueFunction(function=launch_setup)])

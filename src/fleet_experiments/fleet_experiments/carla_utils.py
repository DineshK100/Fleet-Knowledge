"""CARLA helpers: import path set-up and CARLA <-> ROS coordinate conversion.

CARLA uses a left-handed frame (x forward, y right, z up, yaw clockwise in
degrees). ROS uses a right-handed frame, so - like the official CARLA-ROS
bridge - we flip the y axis and the sign of the yaw at the boundary. Every
position that leaves the scenario runner (topics, memories, metrics) is in the
ROS ``map`` frame.
"""

import glob
import math
import os
import sys


def setup_carla_path():
    """Make ``import carla`` and ``import agents`` work from ``$CARLA_ROOT``."""
    root = os.environ.get('CARLA_ROOT', '')
    if root:
        api = os.path.join(root, 'PythonAPI', 'carla')
        if os.path.isdir(api) and api not in sys.path:
            sys.path.append(api)
        major, minor = sys.version_info[:2]
        # old CARLA releases ship .egg files; newer ones ship wheels (pip install them)
        eggs = glob.glob(os.path.join(api, 'dist', f'carla-*py{major}.{minor}*.egg'))
        for egg in eggs:
            if egg not in sys.path:
                sys.path.append(egg)


def carla_to_ros_xy(x: float, y: float):
    return float(x), -float(y)


def ros_to_carla_xy(x: float, y: float):
    return float(x), -float(y)


def carla_yaw_to_ros(yaw_deg: float) -> float:
    return -math.radians(yaw_deg)


def speed_mps(actor) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def forward_speed(actor) -> float:
    v = actor.get_velocity()
    f = actor.get_transform().get_forward_vector()
    return v.x * f.x + v.y * f.y + v.z * f.z


def actor_kind(actor) -> str:
    tid = getattr(actor, 'type_id', '') or ''
    if tid.startswith('walker'):
        return 'walker'
    if tid.startswith('vehicle'):
        return 'vehicle'
    if tid.startswith('static') or tid.startswith('traffic'):
        return 'static'
    return 'other'


def yaw_to_quaternion(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


WEATHER_PRESETS = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'ClearSunset', 'CloudySunset',
                   'WetCloudyNoon', 'SoftRainNoon', 'ClearNight']


def weather_for_seed(carla, seed: int, name: str = 'auto'):
    if name and name != 'auto':
        preset = name
    else:
        preset = WEATHER_PRESETS[seed % len(WEATHER_PRESETS)]
    return preset, getattr(carla.WeatherParameters, preset, carla.WeatherParameters.ClearNoon)


def weather_text(preset: str) -> str:
    p = preset.lower()
    tod = 'night' if 'night' in p else ('sunset' if 'sunset' in p else 'daytime')
    wx = 'rain' if ('rain' in p or 'wet' in p) else ('cloudy' if 'cloud' in p else 'clear')
    return f'{wx} {tod}'

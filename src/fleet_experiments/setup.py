from glob import glob
from setuptools import setup

package_name = 'fleet_experiments'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/scenarios', glob('config/scenarios/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dinesh Karnati',
    maintainer_email='drkarnat@ncsu.edu',
    description='CARLA scenario runner, vehicle interface, metrics, launch files and plotting for the Fleet Memory experiments.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scenario_runner = fleet_experiments.scenario_runner:main',
            'plot_results = fleet_experiments.plot_results:main',
            'run_sweep = fleet_experiments.run_sweep:main',
        ],
    },
)

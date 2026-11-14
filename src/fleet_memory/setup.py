from setuptools import setup

package_name = 'fleet_memory'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dinesh Karnati',
    maintainer_email='drkarnat@ncsu.edu',
    description='Shared fleet memory node: spatial + vector index of hazard memories with merging, confidence decay and pruning.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'memory_node = fleet_memory.memory_node:main',
            'memory_cli = fleet_memory.memory_cli:main',
        ],
    },
)

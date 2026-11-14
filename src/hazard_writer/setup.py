from setuptools import setup

package_name = 'hazard_writer'

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
    description='Per-vehicle surprise trigger and VLM event writer for the Fleet Memory system.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'writer_node = hazard_writer.writer_node:main',
            'dashcam_eval = hazard_writer.dashcam_eval:main',
        ],
    },
)

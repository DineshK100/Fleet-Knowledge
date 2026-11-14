from setuptools import setup

package_name = 'memory_reader'

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
    description='Per-vehicle memory retrieval, scoring and policy adapter (rule-based and LLM) for the Fleet Memory system.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reader_node = memory_reader.reader_node:main',
        ],
    },
)

from setuptools import setup

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name='sst',
    version='0.0.1',
    py_modules=['sst'],
    install_requires=REQUIREMENTS,
    entry_points={
        'console_scripts': [
            'sst = sst:cli',
        ],
    },
)
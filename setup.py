try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


SHORT = 'smartenergy'

setup(
    name='smartenergy',
    packages=[
        'smartenergy'
    ],
    url='https://github.com/javiermas/smart-energy',
    classifiers=(
        'Programming Language :: Python :: 3.6'
    ),
    description=SHORT,
    long_description=SHORT,
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)

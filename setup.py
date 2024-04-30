# Based on the sample at https://github.com/pypa/sampleproject/blob/master/setup.py

from setuptools import setup, find_packages, find_namespace_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
try:
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = "No README found."

setup(
    # Customize as needed
    name='chi-fexp',
    version='0.0.0',
    description='My description.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # Fix this to new repository url
    url='https://github.com/jpeoples/chi-fexp',
    author='Jacob J. Peoples',
    author_email='jacob.peoples@queensu.ca',

    # If you want to use a module rather than entire package, list them here
    # py_modules=['module_file_name'], 
    # This will find packages for you, excluding the folders listed.
    packages=find_namespace_packages(include=["chi.*"]), # Required

    python_requires='>=3.0',
    install_requires=[], # List your dependencies here

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
    },
    # If any entry points, list them here, see https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    entry_points = {}
)

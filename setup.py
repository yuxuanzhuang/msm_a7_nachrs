#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Yuxuan Zhuang",
    author_email='yuxuan.zhuang@dbb.su.se',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Scripts for MSM building of a7 nicotinic receptor",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='msm_a7_nachrs',
    name='msm_a7_nachrs',
    packages=find_packages(include=['msm_a7_nachrs', 'msm_a7_nachrs.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/yuxuanzhuang/msm_a7_nachrs',
    version='0.1.0',
    zip_safe=False,
)

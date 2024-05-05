====================================================
Scripts for system building of a7 nicotinic receptor
====================================================

Accompanying scripts for the paper
"Symmetry-adapted Markov state models of closing, opening, and desensitizing in α7 nicotinic acetylcholine receptors"
https://www.biorxiv.org/content/10.1101/2023.12.04.569956v1.

Please check https://github.com/yuxuanzhuang/sym_msm for
the essential tools to perform markov state models on symmetric systems

For scripts that reproduce the figures, go to https://zenodo.org/records/8424868

* Free software: BSD license

Folders
-------
- prep: Scripts to build the system from scratch.
- dask_worker: Scripts to run dask workers on a cluster
- datafile: Trajectories of structural interpolation with CLimber. 
- manuscript: Scripts to generate figures and tables for the manuscript
- utils: Utility functions for the project
- migrated: Old codes that were moved to https://github.com/yuxuanzhuang/sym_msm

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

# Debiased EEG/MEG source imaging

This repo contains the python source code for the published paper "Debiased Estimation and Inference for Spatial-Temporal EEG/MEG Source Imaging" in ([IEEE TMI, 2025](https://ieeexplore.ieee.org/document/10768920)).

The development of accurate electroencephalography (EEG) and magnetoencephalography (MEG) source imaging algorithm is of great importance for functional brain research and non-invasive presurgical evaluation of epilepsy. In practice, the challenge arises from the fact that the number of measurement channels is far less than the number of candidate source locations, rendering the inverse problem ill-posed. A widely used approach is to introduce a regularization term into the objective function, which inevitably biased the estimated amplitudes towards zero, leading to an inaccurate estimation of the estimator's variance.

In this repo, our goal is to propose a novel debiased EEG/MEG source imaging (DeESI) algorithm for detecting sparse brain activities, which corrects the estimation bias in signal amplitude, dipole orientation and depth. The DeESI extends the idea of group Lasso by incorporating both the matrix Frobenius norm and the L1-norm, which guarantees the estimators are only sparse over sources while maintains smoothness in time and orientation.

## Package dependencies:

* pandas: 1.3.2
* numpy: 1.26.1
* scipy: 1.11.3
* statsmodels: 0.14.1
* matplotlib: 3.4.3
* mne: 1.5.1

## Usage:

To generate the simulation results in the paper, simply run the following command:

```bash
python simulation_study.py
```
where simulation_study.py is the main script for the simulation study, ADMM_GroupLASSO.py contains the implementation of the DeESI algorithm, and data_utils.py contains the utility functions for data generation and performance evaluation.



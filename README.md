# Robust Best-arm Identification for Linear Bandits

This repo contains code for the paper "A/B Testing and Best-arm Identification for Linear Bandits with Robustness to Non-stationarity".

Library requirements: `NumPy`, `Matplotlib`

To run the experiment based on Yahoo!'s dataset, user needs to first download Yahoo! Webscope Dataset R6A from https://webscope.sandbox.yahoo.com/catalog.php?datatype=r and then runs file `process_yahoo.py` and  `get_instance_yahoo.py` in order to get `yahoo_data_pca.npz`. After these stpes, run `run_adv_yahoo.py` will give the experiment results of Yahoo!'s dataset.

To run stationary benchmark example, run `run_sto_soare.py`; to run non-stationary multivariate testing example, run `run_adv_multi.py`; to run non-stationary benchmark example, run `run_adv_soare.py`; to run malicious and stationary multivariate testing single examples, run `run_single.py`.

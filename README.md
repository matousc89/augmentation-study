# Source Codes for Box-secure TA Augmentation Study

Install all requirements in order to run experiments.
Most likely you will need to adjust torch `device`
and `batch size` according to your machine
in the scripts mentioned below.

## Run experiment with Box-secure TA policy

- **Dataset**: billet, saw, tubes
- **Policy**: none, ta_ex, ta_co

```
source .venv/bin/activate
cd datasets
python -u train_policy.py <dataset> <policy>
deactivate
```

## Run experiment for single augmentation

Parameters:
- **Dataset**: billet, saw, tubes

```
source .venv/bin/activate
cd datasets
python -u train_single_aug.py <dataset>
deactivate
```

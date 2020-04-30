# gb_rf_evolution
This repository performs hyperparameter tuning for tree based ensembles (random forest, gradient boosting).

instalation: `pip install git+https://github.com/EvanBagis/gb_rf_evolution.git`

Credits to the repos below.

[]('https://github.com/harvitronix') repository: <https://github.com/harvitronix/neural-network-genetic-algorithm>

[]('https://github.com/subpath') repository: <https://github.com/subpath/neuro-evolution>

Example of usage:

1. Create dictionary with parameters

```python

from gb_rf_evolution import evolution

params = { 'boosting_type':['gbdt', 'dart'], 
           'num_leaves':[31, 41, 51], 
           'learning_rate':[0.1, 0.15, 0,2], 
           'n_estimators':[50, 100, 200], 
           'subsample_for_bin':[100000, 200000, 300000], 
           'objective':['regression'],
           'colsample_bytree':[0.5, 0.7, 1.0], 
           'n_jobs':[-1],
           'max_bin':[100, 1000, 10000],
           'num_iterations':[100, 200, 300],
           'extra_trees':[True, False],
           'reg_sqrt':[True, False] }
```

```python
# x_train, y_train, x_test, y_test - prepared data

search = evolution.gb_rf_evolution(generations = 10, population = 10, params=params)

search.evolve(x_train, y_train, x_test, y_test)
```

```bash
100%|██████████| 10/10 [05:37<00:00, 29.58s/it]
100%|██████████| 10/10 [03:55<00:00, 25.55s/it]
100%|██████████| 10/10 [02:05<00:00, 15.05s/it]
100%|██████████| 10/10 [01:37<00:00, 14.03s/it]
100%|██████████| 10/10 [02:49<00:00, 22.53s/it]
100%|██████████| 10/10 [02:37<00:00, 23.14s/it]
100%|██████████| 10/10 [02:36<00:00, 21.37s/it]
100%|██████████| 10/10 [01:57<00:00, 18.56s/it]
100%|██████████| 10/10 [02:42<00:00, 25.29s/it]
```

```bash
"best coefficient of determination: 0.79,
best params: {'epochs': 35, 'batch_size': 40, 'n_layers': 2, 'n_neurons': 20, 'dropout': 0.1, 'optimizers': 'nadam', 'activations': 'relu'}"
```

## or you can call it with

```python
search.best_params
```

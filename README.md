# README
Data analytics using python

## Installation

...


### Used libraries and styles
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Supported Analytics

- Forecast
- Clustering
- Anomaly Detection
- Classification


## Usage

### Natively in python

Example:

```python
import pylytic

df = ...

res = pylytic.process(
    df=df,
    analytic='forecast',
    timeseries_columns=['date'],
    # optional kwargs:
    n_predictions=6,  # num of forecasts to make
    model = '<name of model>',
    ...
)

print(res.error.mae)
print(res.df_pred)
```

### As a Service

TODO

## Authors
[Johannes Schöck](https://github.com/JSchoeck)

### Building the project
- Uses build package: pip install build
- Invoke build: python -m build

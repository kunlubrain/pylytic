# README
Data analytics using python

## Installation

...

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

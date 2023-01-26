# Project Title: pylytic
# Project Description: Analytics module to integrate into data handling solution.
# Author:  Johannes Schöck
# Date: 11/01/2023
# Github: https://github.com/JSchoeck
# Linkedin: https://www.linkedin.com/in/johannes-schoeck/
# StackOverflow: https://stackoverflow.com/users/3960182/johannes-sch%c3%b6ck?tab=profile
#%%
from typing import Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error

# from sklearn.model_selection import TimeSeriesSplit


class TsDict(TypedDict):
    datetime_col: str
    target_col: str
    forecast_freq: str
    forecast_period: int


def example_timeseries_data(type="sales") -> pd.DataFrame:
    """Load example data from Prophet module.

    Args:
        type (str, optional): Specify which example dataset to load. Defaults to "sales".

    Raises:
        NotImplemented: If specified dataset type is not implemented.

    Returns:
        pd.DataFrame: Example dataset of specified type.
    """
    if type == "sales":
        # Daily sales
        return pd.read_csv(
            "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_retail_sales.csv"
        )
    elif type == "visits":
        # Log of daily Wikipedia page visits
        return pd.read_csv(
            "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
        )
    else:
        raise NotImplementedError(f"Example dataset of type {type} is not implemented.")


def timeseries_prediction(df: pd.DataFrame, kwargs: TsDict) -> Tuple[pd.DataFrame, int]:
    """Wrapper-function for timeseries models to predict a time range based on historic data.
    Currently the Prophet library is used.

    Args:
        df (pd.DataFrame): Historical timeseries data to use for training and prediction.
        kwargs (dict): Arguments passed from calling module. Needs to include:
            # TODO: reference nomenclature / data type
            forecast_period (str): Time unit of the forecast (i.e. days, seconds...)
            forecast_length (int): Number of periods to forecast.

    Returns:
        pd.Series: Predicted data
    """
    if df.empty:  # If no data has been passed, load example dataset of a chosen type
        df = example_timeseries_data()
        kwargs = {
            "datetime_col": "ds",
            "target_col": "y",
            "forecast_freq": "D",
            "forecast_period": 28,
            # TODO: how to handle additional feature cols? Are they automatically used?
        }
        example_mode = True
    else:
        example_mode = False
    print(df.head())

    # TODO: handle missing parameters: infer best parameters for forecast_freq and _period based on dataset time horizon

    # Transform data
    df["y"], lam = boxcox(df[kwargs["target_col"]])  # type: ignore
    # With: CV MAE: 823, CV MSE: 1_433_816  --> calculated from m, not df_forecast! inverse transform required
    # Without: CV MAE: 13_050, CV MSE: 363_033_310
    print(df.head())

    # Define timeseries prediction model
    m = Prophet()  # TODO: define add seasonality, holiday and other features
    m.fit(df)

    # TODO: split df into segments / split segments into training and test data
    # TODO: define default model parameters
    # TODO: handle / add features
    # Custom seasonality: m.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # order is # of parameters

    # Create timeseries for future data to be predicted
    # TODO: Add check if kwargs["forecast_freq"] is a valid freq for pd.date_range or add try to catch error.
    future = m.make_future_dataframe(
        freq=kwargs["forecast_freq"], periods=kwargs["forecast_period"]
    )

    # Predict future data
    df_forecast = m.predict(future)
    print(df_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

    # Inverse transformation
    df_forecast[["yhat", "yhat_upper", "yhat_lower"]] = df_forecast[
        ["yhat", "yhat_upper", "yhat_lower"]
    ].apply(lambda x: inv_boxcox(x, lam))

    # TODO: Iterative training / testing of model of time periods / stratified validation
    horizon = "1 Y"
    df_cv = cross_validation(m, horizon)
    # df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
    df_p = performance_metrics(df_cv)
    print(df_p)
    fig = plot_cross_validation_metric(df_cv, metric="mape")
    # plt.show()
    # With CV
    mae = mean_absolute_error(df_cv["y"], df_cv["yhat"])
    mse = mean_squared_error(df_cv["y"], df_cv["yhat"])
    print(f"CV MAE: {mae}, CV MSE: {mse}")
    # https://facebook.github.io/prophet/docs/diagnostics.html

    # Plot results, only if running in example mode?
    print(f"example_mode: {example_mode}")
    if example_mode:
        fig1 = m.plot(df_forecast)
        # plt.show()
        fig2 = m.plot_components(df_forecast)
    plt.show()

    # TODO: Output:
    # dataframe with only forecast rows
    # accuracy / evaluation result
    # lower/upper confidence intervals (or another metric for confidence)
    df_result = (
        pd.DataFrame()
    )  # Datetime column, prediction (yhat), yhat_lower, yhat_upper, maybe error metric like MAE
    accuracy = 0  # MAE or similar ?
    # What does the enduser really care about? Probably size of the confidence intervall, not measures for training data
    # Any way to measure confidence in %, i.e. with 95% confidence? p-value relevant?
    # TODO: compare delta of yhat_lower and _upper for example dataset with different hyperparameters
    # TODO: find out if prophet has an overall measure for its prediction confidence
    return (df_result, accuracy)


if __name__ == "__main__":
    kwargs = TsDict(
        datetime_col="ds", target_col="y", forecast_freq="D", forecast_period=28
    )
    timeseries_prediction(pd.DataFrame(), kwargs)

# %%

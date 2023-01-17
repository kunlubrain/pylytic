# Project Title: pylytic
# Project Description: Analytics module to integrate into data handling solution.
# Author:  Johannes Schöck
# Date: 11/01/2023
# Github: https://github.com/JSchoeck
# Linkedin: https://www.linkedin.com/in/johannes-schoeck/
# StackOverflow: https://stackoverflow.com/users/3960182/johannes-sch%c3%b6ck?tab=profile

from typing import Tuple

import numpy as np
import pandas as pd

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric


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


def timeseries_prediction(df: pd.DataFrame, kwargs: dict) -> Tuple[pd.DataFrame, int]:
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
        example_mode = True
    else:
        example_mode = False
    df.head()

    # TODO: split df into segments
    # TODO: split segments into training and test data
    # TODO: define default model parameters
    # TODO: handle / add features
    # Custom seasonality: m.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # order is # of parameters

    # Define timeseries prediction model
    m = Prophet()  # TODO: define add seasonality, holiday and other features
    m.fit(df)

    # Create timeseries for future data to be predicted
    # TODO: Add check if kwargs["forecast_freq"] is a valid freq for pd.date_range or add try to catch error.
    future = m.make_future_dataframe(
        freq=kwargs["forecast_freq"], periods=kwargs["forecast_period"]
    )

    # Predict future data
    forecast = m.predict(future)
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

    # TODO: Iterative training / testing of model of time periods / stratified validation
    # df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
    # df_p = performance_metrics(df_cv)
    # fig = plot_cross_validation_metric(df_cv, metric='mape')
    # https://facebook.github.io/prophet/docs/diagnostics.html

    # Plot results, only if running in example mode?
    if example_mode:
        fig1 = m.plot(forecast)
        fig2 = m.plot_components(forecast)

    # TODO: Output:
    # dataframe with only forecast rows
    # accuracy / evaluation result
    # lower/upper confidence intervals (or another metric for confidence)
    df_result = (
        pd.DataFrame()
    )  # Datetime column, prediction (yhat), yhat_lower, yhat_upper, maybe error metric like MAE
    accuracy = 0  # MAE or similar ?
    return (df_result, accuracy)

# Project Title: pylytic
# Project Description: Analytics module to integrate into data handling solution.
# Author:  Johannes Schöck
# Date: 11/01/2023
# Github: https://github.com/JSchoeck
# Linkedin: https://www.linkedin.com/in/johannes-schoeck/
# StackOverflow: https://stackoverflow.com/users/3960182/johannes-sch%c3%b6ck?tab=profile

# NOTE:
# - User should be able to specify options, but start with minimum necessary default parameters
# - Specify required input variables for each task (calling module needs to take care of providing them)


#%%
from typing import Tuple

import numpy as np
import pandas as pd
from anomaly_detection import AdDict, anomaly_detection
from classification import ClaDict, classification
from clustering import CluDict, clustering
from regression import ReDict, regression
from timeseries_prediction import TsDict, timeseries_prediction


def main(df: pd.DataFrame, kwargs: dict) -> Tuple[pd.DataFrame, int]:
    """Calls the function for the specified task and passes data and parameters along.

    Args:
        df (pd.DataFrame): Dataset
        kwargs (dict): Keyword arguments

    Raises:
        NotImplementedError: Raised for prediction tasks that are not implemented.

    Returns:
        pd.DataFrame: Predicted data and metrics with the same index as the input dataset.
    """
    task = kwargs.pop("task")
    # NOTE: TypedDict kwargs has different keys for each task, so "# type: ignore" is needed
    if task == "anomaly_detection":
        df_result, accuracy = anomaly_detection()  # type: ignore
    elif task == "classification":
        df_result, accuracy = classification()  # type: ignore
    elif task == "clustering":
        df_result, accuracy = clustering()  # type: ignore
    elif task == "regression":
        df_result, accuracy = regression()  # type: ignore
    elif task == "timeseries_prediction":
        df_result, accuracy = timeseries_prediction(df, TsDict(**kwargs))
    else:
        raise NotImplementedError(f"The prediction task {task} is not implemented.")
    return df_result, accuracy


if __name__ == "__main__":
    # TODO: Example call, replace with parameters from calling module
    kwargs = {
        "task": "timeseries_prediction",
        "forecast_freq": "D",  # Any valid frequency for pd.date_range, such as 'D' or 'M'.
        "forecast_period": 365,
    }
    main(pd.DataFrame(), kwargs)

# %%

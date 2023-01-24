# Project Title: pylytic
# Project Description: Analytics module to integrate into data handling solution.
# Author:  Johannes Schöck
# Date: 11/01/2023
# Github: https://github.com/JSchoeck
# Linkedin: https://www.linkedin.com/in/johannes-schoeck/
# StackOverflow: https://stackoverflow.com/users/3960182/johannes-sch%c3%b6ck?tab=profile

import context
import pandas as pd
import pytest
from core import main
from timeseries_prediction import TsDict, timeseries_prediction


def test_main():
    # Test case 1: Check if the function raises an error when an unknown task is passed
    with pytest.raises(NotImplementedError) as e:
        main(pd.DataFrame(), {"task": "notimplemented"})
    assert e.typename == "NotImplementedError"

    # Test case 2: Check if the function returns a DataFrame and an int when a valid task is passed
    # df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    # df_result, accuracy = main(df, {"task": "timeseries_prediction"})
    # assert isinstance(df_result, pd.DataFrame)
    # assert isinstance(accuracy, int)


# TODO: Does each of the existing prediction tasks, using example data, yield the correct output type?
# def test_anomaly_detection():
#     # assert


# def test_classification():
#     # assert


# def test_clustering():
#     # assert


# def test_regression():
#     # assert


def test_timeseries_prediction_fails_in_testing(mock_timeseries_data):
    with pytest.raises(RuntimeError, match="Error during optimization!") as e:
        kwargs = kwargs = TsDict(
            datetime_col="ds", target_col="y", forecast_freq="D", forecast_period=28
        )
        timeseries_prediction(mock_timeseries_data, kwargs)

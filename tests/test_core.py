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
from timeseries_prediction import timeseries_prediction


def test_main():
    with pytest.raises(NotImplementedError) as e:
        main(pd.DataFrame(), kwargs={"task": "notimplemented"})
    assert e.typename == "NotImplementedError"


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
        timeseries_prediction(
            mock_timeseries_data, {"forecast_freq": "D", "forecast_period": 28}
        )

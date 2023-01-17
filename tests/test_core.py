# Project Title: pylytic
# Project Description: Analytics module to integrate into data handling solution.
# Author:  Johannes Schöck
# Date: 11/01/2023
# Github: https://github.com/JSchoeck
# Linkedin: https://www.linkedin.com/in/johannes-schoeck/
# StackOverflow: https://stackoverflow.com/users/3960182/johannes-sch%c3%b6ck?tab=profile

import context
import pytest
import pandas as pd

# TODO: remove, only for debugging imports
# import sys
# print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)

from core import main


# TODO: Does the module return expected values?
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


# def test_timeseries_prediction():
# timeseries_prediction(mock_timeseries_data, {"forecast_freq": "D", "forecast_period": 28})
# assert
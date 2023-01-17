# Project Title: pylytic
# Project Description: Analytics module to integrate into data handling solution.
# Author:  Johannes Schöck
# Date: 11/01/2023
# Github: https://github.com/JSchoeck
# Linkedin: https://www.linkedin.com/in/johannes-schoeck/
# StackOverflow: https://stackoverflow.com/users/3960182/johannes-sch%c3%b6ck?tab=profile

from typing import Tuple

import context
import pytest
import pandas as pd

from timeseries_prediction import example_timeseries_data, timeseries_prediction


def test_example_timeseries_data():
    assert example_timeseries_data().empty == False
    assert isinstance(example_timeseries_data(), pd.DataFrame)
    with pytest.raises(NotImplementedError) as e:
        example_timeseries_data(type="notimplemented")
    assert e.typename == "NotImplementedError"


# # TODO: fix, so CMDSTAN is same in pytest and module
# import os
# os.environ["CMDSTAN"] = "C:/Users/Johannes/anaconda3/envs/pylytic/Library/bin/cmdstan"
# def test_timeseries_prediction(mock_timeseries_data):
#     # kwargs = {"forecast_freq": "D", "forecast_period": 28}
#     # result = timeseries_prediction(mock_timeseries_data, kwargs)
#     # assert isinstance(result, Tuple)
#     result = timeseries_prediction(example_timeseries_data(), {"forecast_freq": "D", "forecast_period": 28})
#     assert isinstance(result, Tuple)

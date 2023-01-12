# Project Title: pylytic
# Project Description: Analytics module to integrate into data handling solution.
# Author:  Johannes Schöck
# Date: 11/01/2023
# Github: https://github.com/JSchoeck
# Linkedin: https://www.linkedin.com/in/johannes-schoeck/
# StackOverflow: https://stackoverflow.com/users/3960182/johannes-sch%c3%b6ck?tab=profile

import pytest

from pylytic.timeseries_prediction import *


@pytest.fixture
def mock_timeseries_data():
    return example_timeseries_data()


def test_example_timeseries_data():
    assert example_timeseries_data().empty == False
    with pytest.raises(NotImplementedError) as e:
        example_timeseries_data(type="notimplemented")
    assert e.typename == "NotImplementedError"


# def test_timeseries_prediction():
# assert

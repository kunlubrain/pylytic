# You can use a conftest.py file to create your fixtures
# (setup and tear down code) for reuse across your test modules.

# TODO: fix, so CMDSTAN is same in pytest and module?
import os

import context
import pytest
from timeseries_prediction import example_timeseries_data

os.environ["CMDSTAN"] = "C:/Users/Johannes/anaconda3/envs/pylytic/Library/bin/cmdstan"


@pytest.fixture(scope="session")
def mock_timeseries_data():
    yield example_timeseries_data()

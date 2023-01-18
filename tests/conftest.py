# You can use a conftest.py file to create your fixtures
# (setup and tear down code) for reuse across your test modules.

import context
import pytest
from timeseries_prediction import example_timeseries_data


@pytest.fixture
def mock_timeseries_data():
    return example_timeseries_data()

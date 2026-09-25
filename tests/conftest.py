"""pytest 公共夹具。"""

import pytest

from app.states import expand_state


@pytest.fixture(scope="session")
def demo_inlet():
    """示范工况进口：35 °C、50%RH、常压。"""
    return expand_state(35.0, 101325.0, rh=0.5)

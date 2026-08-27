import logging

import pytest


@pytest.fixture(autouse=True)
def _quiet_sage_logging():
    """Keep code-under-test logging out of test output. Individual logging tests
    add/remove their own handlers and restore afterwards."""
    root = logging.getLogger("sage")
    saved_handlers, saved_propagate, saved_level = root.handlers[:], root.propagate, root.level
    root.handlers[:] = [logging.NullHandler()]
    root.propagate = False
    root.setLevel(logging.DEBUG)
    yield
    root.handlers[:] = saved_handlers
    root.propagate = saved_propagate
    root.setLevel(saved_level)

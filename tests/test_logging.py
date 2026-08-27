"""utils.logging_setup: run-scoped file logging, dedup, UI fan-out."""
import logging

import pytest

from utils.logging_setup import (
    ROOT,
    _RateLimitFilter,
    _UI_HANDLER,
    add_console_subscriber,
    get_logger,
    remove_console_subscriber,
    run_log_file,
)


@pytest.fixture
def ui_handler_attached():
    root = logging.getLogger(ROOT)
    root.addHandler(_UI_HANDLER)
    yield
    root.removeHandler(_UI_HANDLER)


def test_get_logger_namespacing():
    assert get_logger("optimization.engine").name == "sage.optimization.engine"
    assert get_logger("sage.foo").name == "sage.foo"


def test_run_log_file_writes_and_cleans_up(tmp_path):
    root = logging.getLogger(ROOT)
    before = list(root.handlers)
    log = get_logger("test.run")

    with run_log_file(str(tmp_path), logging.DEBUG):
        log.info("hello")
        log.debug("detail %d", 7)

    assert list(root.handlers) == before          # handler removed on exit
    text = (tmp_path / "run.log").read_text()
    assert "hello" in text and "detail 7" in text


def test_run_log_file_cleans_up_on_exception(tmp_path):
    root = logging.getLogger(ROOT)
    before = len(root.handlers)
    with pytest.raises(RuntimeError):
        with run_log_file(str(tmp_path)):
            get_logger("test.boom").info("x")
            raise RuntimeError("boom")
    assert len(root.handlers) == before


def test_dedup_collapses_debug_burst(tmp_path):
    log = get_logger("test.spam")
    with run_log_file(str(tmp_path), logging.DEBUG):
        for i in range(50):
            log.debug("failed at %.1fmm", i * 3.0)     # same template
        log.debug("something else")
    lines = (tmp_path / "run.log").read_text().splitlines()
    failed = [ln for ln in lines if "failed at" in ln and "suppressed" not in ln]
    assert len(failed) == _RateLimitFilter.K
    assert any("suppressed 47 more" in ln for ln in lines)
    assert any("something else" in ln for ln in lines)


def test_dedup_never_limits_info(tmp_path):
    log = get_logger("test.gen")
    with run_log_file(str(tmp_path), logging.INFO):
        for i in range(20):
            log.info("gen %d", i)                       # identical template, INFO
    gen_lines = [ln for ln in (tmp_path / "run.log").read_text().splitlines()
                 if ln.strip().endswith(tuple(f"gen {i}" for i in range(20)))]
    assert len(gen_lines) == 20


def test_opt_level_drops_debug(tmp_path):
    log = get_logger("test.hot")
    with run_log_file(str(tmp_path), logging.INFO):
        log.debug("per-solve noise")
        log.info("kept")
    text = (tmp_path / "run.log").read_text()
    assert "per-solve noise" not in text and "kept" in text


def test_ui_subscriber_receives_lines(tmp_path, ui_handler_attached):
    got = []
    add_console_subscriber(got.append)
    try:
        with run_log_file(str(tmp_path), logging.DEBUG):
            get_logger("test.ui").info("shown in ui")
            get_logger("test.ui").debug("not shown")     # UI handler is INFO
    finally:
        remove_console_subscriber(got.append)
    assert any("shown in ui" in line for line in got)
    assert not any("not shown" in line for line in got)


def test_ui_subscriber_exception_does_not_break_emit(tmp_path, ui_handler_attached):
    def boom(_):
        raise ValueError("nope")

    good = []
    add_console_subscriber(boom)
    add_console_subscriber(good.append)
    try:
        with run_log_file(str(tmp_path)):
            get_logger("test.ui2").info("still delivered")
    finally:
        remove_console_subscriber(boom)
        remove_console_subscriber(good.append)
    assert any("still delivered" in line for line in good)


def test_generation_logger_emits_per_call(caplog):
    from optimization.engine import _GenerationLogger
    import types

    algo = types.SimpleNamespace(
        n_gen=4,
        evaluator=types.SimpleNamespace(n_eval=24),
        opt=[1, 2, 3],
    )
    with caplog.at_level(logging.INFO, logger="sage.optimization.engine"):
        _GenerationLogger().notify(algo)
    assert "gen 4" in caplog.text and "24 designs" in caplog.text

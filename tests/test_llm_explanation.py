import http.client
import json
from datetime import date
from urllib.error import HTTPError

import pandas as pd
import pytest

from app.llm_explanation import (
    IncompleteExplanationError,
    build_explanation_payload,
    explain_decisions,
)


def sample_frame():
    return pd.DataFrame([{
        "sku": "SKU001",
        "as_of_date": date(2026, 9, 20),
        "daily_demand": 2.0,
        "current_stock": 3.0,
        "inventory_position": 8.0,
        "reorder_point": 12.0,
        "recommended_order_qty": 4.0,
        "procurement_recommended_qty": 10.0,
        "reason": "IN_TRANSIT_INSUFFICIENT",
        "internal_note": "must not be sent",
    }])


def test_payload_is_allowlisted_and_does_not_change_frame():
    frame = sample_frame()
    before = frame.copy(deep=True)
    payload = build_explanation_payload(frame)
    assert payload == [{
        "sku": "SKU001",
        "as_of_date": "2026-09-20",
        "daily_demand": 2.0,
        "current_stock": 3.0,
        "inventory_position": 8.0,
        "reorder_point": 12.0,
        "recommended_order_qty": 4.0,
        "procurement_recommended_qty": 10.0,
        "reason": "IN_TRANSIT_INSUFFICIENT",
    }]
    pd.testing.assert_frame_equal(frame, before)
    assert "internal_note" not in payload[0]


def test_requires_api_key_and_model():
    frame = sample_frame()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        explain_decisions(frame, api_key="", model="test-model")
    with pytest.raises(RuntimeError, match="SCOS_LLM_MODEL"):
        explain_decisions(frame, api_key="test-key", model="")


def test_sends_only_projection_and_returns_text(monkeypatch):
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps(
            {"choices": [{"message": {"content": "SKU001：系统建议采购 10。"}, "finish_reason": "stop"}]}
        ).encode()

    seen = {}
    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode())
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("app.llm_explanation.urlopen", fake_urlopen)
    text = explain_decisions(
        sample_frame(), api_key="test-key", base_url="https://example.test/v1",
        model="test-model", timeout=3,
    )
    assert text == "SKU001：系统建议采购 10。"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    user_message = seen["body"]["messages"][1]["content"]
    assert "must not be sent" not in user_message
    assert seen["body"]["temperature"] == 0
    assert seen["timeout"] == 3


def test_rejects_malformed_response(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b'{"choices": []}'
    monkeypatch.setattr("app.llm_explanation.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="did not contain"):
        explain_decisions(sample_frame(), api_key="test-key", model="test-model")


def test_rejects_invalid_json_body(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b"not-json"
    monkeypatch.setattr("app.llm_explanation.urlopen", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="not valid JSON"):
        explain_decisions(sample_frame(), api_key="test-key", model="test-model")


def test_rejects_empty_body(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b""
    monkeypatch.setattr("app.llm_explanation.urlopen", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="not valid JSON"):
        explain_decisions(sample_frame(), api_key="test-key", model="test-model")


# --- integration with the deterministic pipeline / demo -------------------

class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": self._content}, "finish_reason": "stop"}]}
        ).encode()


def _pipeline_frame():
    """Run the real deterministic chain: demand -> inventory -> procurement."""
    from app.data_loader import (
        load_data_metadata,
        load_in_transit,
        load_inventory,
        load_orders,
        load_procurement_constraints,
    )
    from app.demand import calculate_demand
    from app.inventory import analyze_inventory_frame
    from app.procurement import apply_procurement

    as_of = date(2026, 9, 20)
    demand = calculate_demand(load_orders(), as_of, coverage_start=load_data_metadata())
    batch = analyze_inventory_frame(
        load_inventory(), demand, in_transit_df=load_in_transit(), as_of_date=as_of
    )
    return apply_procurement(batch.frame, load_procurement_constraints())


def test_explanation_call_does_not_change_quantities(monkeypatch):
    frame = sample_frame()
    before = frame.copy(deep=True)
    monkeypatch.setattr(
        "app.llm_explanation.urlopen", lambda *a, **k: _FakeResponse("解释文本")
    )
    explain_decisions(frame, api_key="test-key", model="test-model")
    pd.testing.assert_frame_equal(frame, before)


def test_pipeline_quantities_survive_explanation(monkeypatch):
    frame = _pipeline_frame()
    columns = ["sku", "status", "recommended_order_qty", "procurement_recommended_qty"]
    before = frame[columns].copy(deep=True)
    monkeypatch.setattr(
        "app.llm_explanation.urlopen", lambda *a, **k: _FakeResponse("解释文本")
    )
    explain_decisions(frame, api_key="test-key", model="test-model")
    pd.testing.assert_frame_equal(before, frame[columns])


def _enable(monkeypatch, api_key="test-key", model="test-model"):
    monkeypatch.setenv("SCOS_LLM_EXPLANATION", "1")
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setenv("SCOS_LLM_MODEL", model)


def test_demo_runs_normally_when_llm_disabled(capsys, monkeypatch):
    from app.demo import main

    monkeypatch.delenv("SCOS_LLM_EXPLANATION", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SCOS_LLM_MODEL", raising=False)
    main(as_of_date=date(2026, 9, 20))
    out = capsys.readouterr().out
    assert "分析完成" in out           # deterministic report still runs
    assert "大模型辅助解释" not in out  # opt-in block skipped by default


def test_demo_survives_llm_failure(capsys, monkeypatch):
    from app.demo import main
    from urllib.error import URLError

    _enable(monkeypatch)

    def boom(*args, **kwargs):
        raise URLError("network down")

    monkeypatch.setattr("app.llm_explanation.urlopen", boom)
    main(as_of_date=date(2026, 9, 20))
    out = capsys.readouterr().out
    assert "分析完成" in out
    assert "[LLM_EXPLANATION_UNAVAILABLE]" in out


def test_demo_prints_explanation_on_success(capsys, monkeypatch):
    from app.demo import main

    _enable(monkeypatch)
    monkeypatch.setattr(
        "app.llm_explanation.urlopen", lambda *a, **k: _FakeResponse("SKU001：系统建议采购 10。")
    )
    main(as_of_date=date(2026, 9, 20))
    out = capsys.readouterr().out
    assert "大模型辅助解释" in out
    assert "SKU001：系统建议采购 10。" in out
    assert "分析完成" in out


def test_api_key_never_appears_in_output(capsys, monkeypatch):
    from app.demo import main

    _enable(monkeypatch, api_key="secret-key-xyz-123")
    monkeypatch.setattr(
        "app.llm_explanation.urlopen", lambda *a, **k: _FakeResponse("解释文本")
    )
    main(as_of_date=date(2026, 9, 20))
    out = capsys.readouterr().out
    assert "secret-key-xyz-123" not in out


# --- phase 2.1: timeout configuration ------------------------------------

class _OkResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()


def test_default_request_timeout_is_90_seconds(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["timeout"] = timeout
        return _OkResponse()

    monkeypatch.setattr("app.llm_explanation.urlopen", fake_urlopen)
    explain_decisions(sample_frame(), api_key="test-key", model="test-model")  # no explicit timeout
    assert seen["timeout"] == 90


def test_explicit_timeout_is_still_honoured(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["timeout"] = timeout
        return _OkResponse()

    monkeypatch.setattr("app.llm_explanation.urlopen", fake_urlopen)
    explain_decisions(sample_frame(), api_key="test-key", model="test-model", timeout=7)
    assert seen["timeout"] == 7


def test_urlopen_timeout_becomes_runtime_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("app.llm_explanation.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="timed out"):
        explain_decisions(sample_frame(), api_key="test-key", model="test-model")


def test_no_automatic_retry_on_timeout(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr("app.llm_explanation.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError):
        explain_decisions(sample_frame(), api_key="test-key", model="test-model")
    assert calls["n"] == 1, "a timeout must not be retried"


# --- phase 2.8: finish_reason contract ------------------------------------


def _response_bytes(choice: dict) -> bytes:
    return json.dumps({"choices": [choice]}).encode()


def _patch_urlopen(monkeypatch, payload, captured=None):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    def fake(request, timeout):
        if captured is not None:
            captured["calls"] = captured.get("calls", 0) + 1
        return _Resp()

    monkeypatch.setattr("app.llm_explanation.urlopen", fake)


def _explain(monkeypatch, choice, captured=None):
    _patch_urlopen(monkeypatch, _response_bytes(choice), captured)
    return explain_decisions(sample_frame(), api_key="test-key", model="test-model")


def test_finish_reason_stop_returns_the_text(monkeypatch):
    text = _explain(monkeypatch, {"message": {"content": "完整解释文本"}, "finish_reason": "stop"})
    assert text == "完整解释文本"


def test_finish_reason_length_fails_and_never_returns_the_truncated_text(monkeypatch):
    with pytest.raises(IncompleteExplanationError, match="length"):
        _explain(monkeypatch, {"message": {"content": "被截断的解释"}, "finish_reason": "length"})


def test_finish_reason_unknown_fails(monkeypatch):
    with pytest.raises(IncompleteExplanationError, match="unknown"):
        _explain(monkeypatch, {"message": {"content": "上游未报告结束原因"}, "finish_reason": "unknown"})


def test_missing_finish_reason_fails(monkeypatch):
    with pytest.raises(IncompleteExplanationError, match="None"):
        _explain(monkeypatch, {"message": {"content": "没有结束原因的文本"}})


@pytest.mark.parametrize("reason", ["content_filter", "tool_calls", "", "STOP"])
def test_any_other_finish_reason_fails(monkeypatch, reason):
    with pytest.raises(IncompleteExplanationError):
        _explain(monkeypatch, {"message": {"content": "文本"}, "finish_reason": reason})


def test_incomplete_explanation_error_is_a_runtime_error(monkeypatch):
    # app/demo.py catches RuntimeError, so the subclass keeps that behaviour
    with pytest.raises(RuntimeError):
        _explain(monkeypatch, {"message": {"content": "文本"}, "finish_reason": "length"})


@pytest.mark.parametrize(
    ("choice", "message"),
    [
        ({}, "did not contain"),
        ({"message": {}}, "did not contain"),
        ({"message": {"content": None}, "finish_reason": "stop"}, "content was empty"),
        ({"message": {"content": 42}, "finish_reason": "stop"}, "content was empty"),
        ({"message": {"content": "   "}, "finish_reason": "stop"}, "content was empty"),
    ],
)
def test_structural_failures_are_explicit(monkeypatch, choice, message):
    with pytest.raises(RuntimeError, match=message):
        _explain(monkeypatch, choice)


def test_empty_choices_fails(monkeypatch):
    _patch_urlopen(monkeypatch, json.dumps({"choices": []}).encode())
    with pytest.raises(RuntimeError, match="did not contain"):
        explain_decisions(sample_frame(), api_key="test-key", model="test-model")


@pytest.mark.parametrize("status", [502, 504])
def test_http_errors_fail_without_retry(monkeypatch, status):
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def fake(request, timeout):
        captured["calls"] = captured.get("calls", 0) + 1
        raise HTTPError("http://127.0.0.1/v1/chat/completions", status, "err", None, None)

    monkeypatch.setattr("app.llm_explanation.urlopen", fake)
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        explain_decisions(sample_frame(), api_key="test-key", model="test-model")
    assert captured["calls"] == 1, "an HTTP error must not be retried"


def test_timeout_fails_without_retry(monkeypatch):
    captured = {}

    def fake(request, timeout):
        captured["calls"] = captured.get("calls", 0) + 1
        raise TimeoutError("timed out")

    monkeypatch.setattr("app.llm_explanation.urlopen", fake)
    with pytest.raises(RuntimeError, match="timed out"):
        explain_decisions(sample_frame(), api_key="test-key", model="test-model")
    assert captured["calls"] == 1


# --- deterministic output must be untouched -------------------------------


def _deterministic_prefix(stdout: str) -> str:
    """Everything the demo prints before the optional explanation block.

    The explanation block starts with a blank line and a dashed separator, so those are
    trimmed; the remaining text is exactly the deterministic report.
    """
    return stdout.split("大模型辅助解释")[0].rstrip().rstrip("-").rstrip()


def test_deterministic_output_is_identical_for_every_explanation_outcome(capsys, monkeypatch):
    from app.demo import main

    def run(explanation_enabled, urlopen_impl):
        if explanation_enabled:
            monkeypatch.setenv("SCOS_LLM_EXPLANATION", "1")
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            monkeypatch.setenv("SCOS_LLM_MODEL", "test-model")
        else:
            monkeypatch.delenv("SCOS_LLM_EXPLANATION", raising=False)
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)
            monkeypatch.delenv("SCOS_LLM_MODEL", raising=False)
        if urlopen_impl is not None:
            monkeypatch.setattr("app.llm_explanation.urlopen", urlopen_impl)
        main(as_of_date=date(2026, 9, 20))
        return capsys.readouterr().out

    def failing(request, timeout):
        raise HTTPError("http://127.0.0.1/v1/chat/completions", 502, "err", None, None)

    def succeeding(request, timeout):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return _response_bytes({"message": {"content": MOCK_EXPLANATION}, "finish_reason": "stop"})

        return _Resp()

    disabled = run(False, None)
    failed = run(True, failing)
    succeeded = run(True, succeeding)

    prefix_disabled = _deterministic_prefix(disabled)
    prefix_failed = _deterministic_prefix(failed)
    prefix_succeeded = _deterministic_prefix(succeeded)

    # the deterministic part must be byte-identical in all three outcomes
    assert prefix_failed == prefix_disabled
    assert prefix_succeeded == prefix_disabled

    # ...and it must contain the real business output, not an empty prefix
    assert "核心决策" in prefix_disabled
    assert "建议采购量" in prefix_disabled
    assert "采购建议量(L2)" in prefix_disabled
    assert "SKU001" in prefix_disabled
    assert "分析完成" in prefix_disabled

    # only the optional block differs
    assert "大模型辅助解释" not in disabled
    assert "[LLM_EXPLANATION_UNAVAILABLE]" in failed
    assert MOCK_EXPLANATION in succeeded
    assert MOCK_EXPLANATION not in prefix_succeeded


MOCK_EXPLANATION = "这是模型生成的解释文本"

# --- phase 2.10: transport failures must stay inside the RuntimeError contract ---


def _raising_urlopen(exc):
    def fake(request, timeout):
        raise exc

    return fake


@pytest.mark.parametrize(
    "exc",
    [
        http.client.BadStatusLine("garbage-from-upstream"),
        http.client.IncompleteRead(b"partial", 100),
        http.client.LineTooLong("a" * 10),
    ],
)
def test_http_client_exceptions_become_runtime_errors(monkeypatch, exc):
    """BadStatusLine / IncompleteRead are HTTPException: neither OSError nor ValueError.

    Without an explicit branch they escaped explain_decisions unchanged, and the only
    caller (app/demo.py) catches just (RuntimeError, ValueError), so the demo died with
    a traceback instead of degrading. The response is also already broken at this point,
    so no retry may happen.
    """
    calls = {"n": 0}

    def fake(request, timeout):
        calls["n"] += 1
        raise exc

    monkeypatch.setattr("app.llm_explanation.urlopen", fake)
    with pytest.raises(RuntimeError, match="request failed"):
        explain_decisions(sample_frame(), api_key="test-key", model="test-model")
    assert calls["n"] == 1


def test_demo_degrades_on_bad_status_line(capsys, monkeypatch):
    from app.demo import main

    _enable(monkeypatch)
    monkeypatch.setattr(
        "app.llm_explanation.urlopen",
        _raising_urlopen(http.client.BadStatusLine("garbage-from-upstream")),
    )
    main(as_of_date=date(2026, 9, 20))
    out = capsys.readouterr().out
    assert "分析完成" in out
    assert "[LLM_EXPLANATION_UNAVAILABLE]" in out


def test_base_url_without_a_scheme_is_a_runtime_error(monkeypatch):
    """Request() raises ValueError for a scheme-less URL; the contract says RuntimeError."""

    def never_called(*args, **kwargs):
        raise AssertionError("no request may be sent for an invalid base_url")

    monkeypatch.setattr("app.llm_explanation.urlopen", never_called)
    with pytest.raises(RuntimeError, match="base_url is not a valid URL"):
        explain_decisions(
            sample_frame(), api_key="test-key", base_url="localhost", model="test-model"
        )

# --- phase 3.3: request construction stays inside the RuntimeError contract ---


def _never_called_urlopen(*args, **kwargs):
    raise AssertionError("no request may be sent when the payload cannot be built or serialized")


def _frame_with_reason(value):
    return pd.DataFrame([{"sku": "SKU001", "reason": value}])


def test_unserializable_frame_value_is_a_runtime_error(monkeypatch):
    """A set in the frame makes json.dumps raise TypeError; callers only know RuntimeError."""
    monkeypatch.setattr("app.llm_explanation.urlopen", _never_called_urlopen)
    with pytest.raises(RuntimeError, match="could not be serialized") as excinfo:
        explain_decisions(_frame_with_reason({1, 2}), api_key="test-key", model="test-model")
    assert "TypeError" in str(excinfo.value)


def test_non_finite_float_is_a_runtime_error_without_echoing_the_value(monkeypatch):
    """allow_nan=False turns inf into ValueError, whose own message echoes the value."""
    monkeypatch.setattr("app.llm_explanation.urlopen", _never_called_urlopen)
    with pytest.raises(RuntimeError, match="could not be serialized") as excinfo:
        explain_decisions(_frame_with_reason((float("inf"),)), api_key="test-key", model="test-model")
    assert "ValueError" in str(excinfo.value)
    assert "inf" not in str(excinfo.value)  # the encoder message must not be forwarded


def test_frame_without_sku_is_a_runtime_error(monkeypatch):
    """The projection raises ValueError for a missing sku column; the caller contract is RuntimeError."""
    monkeypatch.setattr("app.llm_explanation.urlopen", _never_called_urlopen)
    with pytest.raises(RuntimeError, match="payload is invalid"):
        explain_decisions(pd.DataFrame([{"reason": "x"}]), api_key="test-key", model="test-model")


def test_demo_degrades_when_the_request_cannot_be_serialized(capsys, monkeypatch):
    """End to end: an unserializable frame must degrade, and the deterministic prefix must not change."""
    from app.demo import main

    def run(explanation_enabled):
        if explanation_enabled:
            _enable(monkeypatch)
        else:
            for name in ("SCOS_LLM_EXPLANATION", "OPENAI_API_KEY", "SCOS_LLM_MODEL"):
                monkeypatch.delenv(name, raising=False)
        main(as_of_date=date(2026, 9, 20))
        return capsys.readouterr().out

    disabled = run(False)

    # the projection is replaced by one the JSON encoder rejects, so the failure happens
    # inside explain_decisions and must not reach the user as a traceback
    monkeypatch.setattr(
        "app.llm_explanation.build_explanation_payload",
        lambda frame: [{"sku": "SKU001", "reason": {1, 2}}],
    )
    monkeypatch.setattr("app.llm_explanation.urlopen", _never_called_urlopen)
    failed = run(True)

    assert "分析完成" in failed
    assert "[LLM_EXPLANATION_UNAVAILABLE]" in failed
    assert "could not be serialized" in failed
    assert _deterministic_prefix(failed) == _deterministic_prefix(disabled)

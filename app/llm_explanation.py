"""Optional LLM explanations for deterministic inventory decisions.

The LLM receives an allowlisted projection of an already-computed decision frame.
It must never calculate, replace, or modify inventory/procurement quantities.
No external dependency is required; the client uses the OpenAI-compatible HTTP API.
"""
from __future__ import annotations

import http.client
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Keep this list narrow: only fields useful for explaining an existing decision.
EXPLANATION_FIELDS = (
    "sku",
    "as_of_date",
    "demand_basis",
    "daily_demand",
    "daily_demand_30d",
    "daily_demand_7d",
    "demand_trend",
    "current_stock",
    "safety_stock",
    "lead_time_days",
    "in_transit_stock",
    "overdue_in_transit",
    "inventory_position",
    "reorder_point",
    "shortage_qty",
    "status",
    "recommended_order_qty",
    "reason",
    "procurement_recommended_qty",
    "procurement_reason",
    "provisional",
    "data_quality_status",
)

SYSTEM_PROMPT = """你是库存补货决策解释助手，不是决策引擎。
只根据用户提供的结构化字段解释已有结果，不得重新计算、修改或推荐不同的数量。
不得补造缺失事实、成本、供应商信息或业务规则。区分数据事实、系统规则和不确定性。
优先简洁说明：当前状态、系统给出该结果的直接依据、在途/数据质量风险、需要人工核实的事项。
所有数量必须原样引用输入字段；若字段缺失，明确说明无法判断。不要声称已经下单。
"""


def build_explanation_payload(frame) -> list[dict]:
    """Project the existing result frame; never mutate it or calculate new values."""
    present = [field for field in EXPLANATION_FIELDS if field in frame.columns]
    if "sku" not in present:
        raise ValueError("decision frame must contain sku")
    records = []
    for row in frame[present].to_dict("records"):
        records.append({key: _json_value(value) for key, value in row.items()})
    return records


class IncompleteExplanationError(RuntimeError):
    """Raised when the upstream response cannot be treated as a complete explanation.

    ``explain_decisions`` accepts only ``finish_reason == "stop"``. A missing field, or
    ``length`` / ``unknown`` / any other value, raises this error (a ``RuntimeError``
    subclass) so callers never receive a truncated or unfinished answer as if it were a
    complete explanation.
    """


def explain_decisions(
    frame,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 90,
) -> str:
    """Explain existing decisions using an OpenAI-compatible chat-completions API.

    Configure OPENAI_API_KEY and SCOS_LLM_MODEL; optionally set SCOS_LLM_BASE_URL.
    Call this only after deterministic calculations have completed.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    model = model or os.getenv("SCOS_LLM_MODEL")
    base_url = (base_url or os.getenv("SCOS_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM explanations")
    if not model:
        raise RuntimeError("SCOS_LLM_MODEL is required for LLM explanations")

    # Building the request must not leak a raw TypeError/ValueError: app/demo.py only
    # relies on the RuntimeError contract. The two failure sources stay separate because
    # json.dumps echoes the offending value in its own message (frame data), while the
    # projection raises our fixed text.
    try:
        payload = build_explanation_payload(frame)
    except ValueError as exc:  # e.g. a decision frame without a sku column
        raise RuntimeError(f"LLM explanation payload is invalid: {exc}") from exc
    try:
        body = json.dumps(
            {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "请解释以下已完成的库存决策。输入是唯一事实来源，不要改动任何数量：\n"
                        + json.dumps(payload, ensure_ascii=False, allow_nan=False),
                    },
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        # exc is deliberately not interpolated: its text can echo frame values.
        raise RuntimeError(
            f"LLM explanation request could not be serialized ({type(exc).__name__})"
        ) from exc
    try:
        request = Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
    except ValueError as exc:  # e.g. a base_url without a scheme ("localhost")
        raise RuntimeError(f"LLM explanation base_url is not a valid URL: {base_url!r} ({exc})") from exc
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"LLM explanation request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"LLM explanation service unavailable: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise RuntimeError(f"LLM explanation request failed: {exc}") from exc
    except http.client.HTTPException as exc:
        # BadStatusLine / IncompleteRead / LineTooLong are HTTPException, which is
        # neither OSError nor ValueError: without this branch they escape as a
        # non-RuntimeError and the caller's degradation path never runs.
        raise RuntimeError(f"LLM explanation request failed: {exc}") from exc
    except ValueError as exc:  # invalid UTF-8 / invalid or empty JSON body
        raise RuntimeError(f"LLM explanation response was not valid JSON: {exc}") from exc

    try:
        choice = result["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM response did not contain choices[0].message.content") from exc

    # Only a completed generation counts as a full explanation. Everything else
    # (length / unknown / missing / content_filter / ...) fails loudly instead of handing
    # a truncated or unfinished answer back to the caller.
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if finish_reason != "stop":
        raise IncompleteExplanationError(
            "LLM explanation is not a complete answer (finish_reason="
            f"{finish_reason!r}); only 'stop' is accepted"
        )

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response content was empty")
    return content.strip()


def _json_value(value):
    """Convert common pandas/numpy values to JSON-safe native values."""
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    try:
        if value != value:  # NaN / NaT
            return None
    except (TypeError, ValueError):
        pass
    return value

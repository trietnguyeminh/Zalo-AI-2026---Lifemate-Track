"""Execute a supplied proposal list in one environment with an explicit ledger.

This is a public contract example, not an LLM planner. The actual competition
runner must use the organizers' original ToolEnv and its returned state hash.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Callable, Iterable, Protocol, TextIO


class ToolEnvironment(Protocol):
    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def get_tool_calls(self) -> list[dict[str, Any]]: ...
    def get_state_hash(self) -> str: ...


def execute_proposals(sample_id: str, make_env: Callable[[], ToolEnvironment],
                      proposals: Iterable[dict[str, Any]], ledger: TextIO) -> dict[str, Any]:
    """No reset/replay/retry. Invalid proposals are logged and never executed.

    A tool exception leaves a diagnostic ledger and raises; no complete submission
    is claimed. Call logging requires a reliable writable stream. It is not an
    exactly-once distributed transaction or a crash-recovery implementation.
    """
    def record(event: str, **fields: Any) -> None:
        ledger.write(json.dumps({"id": sample_id, "event": event, **fields},
                                ensure_ascii=False, allow_nan=False) + "\n")
        ledger.flush()

    env = make_env()
    record("env_created", constructors=1, history_replayed=False)
    actual_calls: list[dict[str, Any]] = []
    for proposal in proposals:
        call = copy.deepcopy(proposal)
        name, arguments = call["name"], call["arguments"]
        validation = env.validate(name, copy.deepcopy(arguments))
        record("validation", call=call, result=validation)
        if not validation.get("ok"):
            continue
        record("call_started", call=call)
        try:
            result = env.call(name, copy.deepcopy(arguments))
        except Exception as exc:
            record("call_exception", call=call, exception_type=type(exc).__name__)
            raise
        actual_calls.append(call)
        record("call_result", call=call, result=result)
        if name == "sys.ask_missing_info" and result.get("executed", True) and result.get("ok", True):
            break
    submitted = env.get_tool_calls()
    if submitted != actual_calls:
        raise ValueError("SDK calls disagree with the execution ledger")
    return {"id": sample_id, "tool_calls": submitted, "state_hash": env.get_state_hash()}

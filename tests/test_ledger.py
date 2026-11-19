import copy
import io
import json
import unittest

from lifemate_portfolio.ledger import execute_proposals


class FixtureEnvironment:
    def __init__(self, *, corrupt=False, fail=False):
        self.calls, self.corrupt, self.fail = [], corrupt, fail

    def validate(self, name, arguments):
        return {"ok": name in {"fixture.read", "fixture.write", "sys.ask_missing_info"}}

    def call(self, name, arguments):
        if self.fail:
            raise RuntimeError("injected tool failure")
        self.calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
        arguments["tool_internal_mutation"] = True
        return {"ok": True, "executed": True}

    def get_tool_calls(self):
        return [] if self.corrupt else copy.deepcopy(self.calls)

    def get_state_hash(self):
        return "fixture_only_not_an_organizer_hash"


class LedgerTests(unittest.TestCase):
    def run_fixture(self, proposals, **kwargs):
        constructors = []
        def factory():
            constructors.append(True)
            return FixtureEnvironment(**kwargs)
        log = io.StringIO()
        result = execute_proposals("synthetic", factory, proposals, log)
        return result, [json.loads(s) for s in log.getvalue().splitlines()], constructors

    def test_one_env_and_all_invocations_logged(self):
        proposals = [{"name": "fixture.read", "arguments": {}},
                     {"name": "fixture.write", "arguments": {"value": 3}}]
        result, events, constructors = self.run_fixture(proposals)
        self.assertEqual(len(constructors), 1)
        self.assertEqual(result["tool_calls"], [e["call"] for e in events if e["event"] == "call_result"])
        self.assertEqual(result["tool_calls"], proposals)

    def test_invalid_proposal_not_executed(self):
        result, events, _ = self.run_fixture([{"name": "not_installed", "arguments": {}}])
        self.assertEqual(result["tool_calls"], [])
        self.assertFalse(any(e["event"] == "call_started" for e in events))

    def test_successful_ask_stops_before_later_write(self):
        result, _, _ = self.run_fixture([{"name": "sys.ask_missing_info", "arguments": {}},
                                         {"name": "fixture.write", "arguments": {}}])
        self.assertEqual(len(result["tool_calls"]), 1)

    def test_tool_mutation_does_not_change_original_proposals(self):
        proposals = [{"name": "fixture.write", "arguments": {"value": 3}}]
        before = copy.deepcopy(proposals)
        self.run_fixture(proposals)
        self.assertEqual(proposals, before)

    def test_ledger_disagreement_rejects_result(self):
        with self.assertRaises(ValueError):
            self.run_fixture([{"name": "fixture.read", "arguments": {}}], corrupt=True)

    def test_exception_keeps_diagnostic_and_never_retries(self):
        constructions = []
        def factory():
            constructions.append(True)
            return FixtureEnvironment(fail=True)
        log = io.StringIO()
        with self.assertRaises(RuntimeError):
            execute_proposals("synthetic", factory, [{"name": "fixture.read", "arguments": {}}], log)
        events = [json.loads(s) for s in log.getvalue().splitlines()]
        self.assertEqual(len(constructions), 1)
        self.assertEqual(events[-1]["event"], "call_exception")

    def test_empty_plan_makes_no_app_calls(self):
        result, events, constructors = self.run_fixture([])
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(len(constructors), 1)
        self.assertEqual(len(events), 1)

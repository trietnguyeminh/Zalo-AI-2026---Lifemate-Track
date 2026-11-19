"""No audio, model weights, competition data or organizer SDK are used here."""
from pathlib import Path
import hashlib
import io
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate_portfolio.ledger import execute_proposals


class DemoEnvironment:
    """Invented fixture. Its demo_ digest is NOT a competition state hash."""
    def __init__(self):
        self.light_on = False
        self.calls = []

    def validate(self, name, arguments):
        valid = (name == "demo.list_lights" and arguments == {}) or (
            name == "demo.set_light" and arguments == {"id": "demo_lamp", "on": True})
        return {"ok": valid}

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments.copy()})
        if name == "demo.set_light":
            self.light_on = arguments["on"]
            return {"ok": True, "executed": True}
        return {"ok": True, "lights": [{"id": "demo_lamp", "on": self.light_on}]}

    def get_tool_calls(self):
        return self.calls.copy()

    def get_state_hash(self):
        return "demo_" + hashlib.sha256(str(self.light_on).encode()).hexdigest()


if __name__ == "__main__":
    log = io.StringIO()
    proposals = [{"name": "demo.list_lights", "arguments": {}},
                 {"name": "demo.set_light", "arguments": {"id": "demo_lamp", "on": True}}]
    result = execute_proposals("synthetic_demo", DemoEnvironment, proposals, log)
    print("SYNTHETIC CONTRACT DEMO — not audio/LLM inference")
    print(json.dumps(result, indent=2))
    print("Ledger events:", len(log.getvalue().splitlines()))

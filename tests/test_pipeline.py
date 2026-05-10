"""
Unit tests for pipeline.py.

Tests cover PipelineResult defaults and the Pipeline.run() control flow
using fully mocked agents — no LLM calls, no vector store.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stubs — prevent transitive imports from loading real clients
# ---------------------------------------------------------------------------

def _make_stubs():
    config_mod = types.ModuleType("config")
    config_mod.cfg = type("cfg", (), {
        "LLM_API_KEY": "test",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_MODEL": "gpt-4o-mini",
        "MAX_TOKENS": 8192,
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "CHUNK_SIZE": 800,
        "CHROMA_DIR": "/tmp/test_chroma",
        "DOCS_ROOT": __import__("pathlib").Path("/tmp/test_docs"),
    })()
    sys.modules.setdefault("config", config_mod)

    for name in ("openai", "tenacity", "chromadb", "fastembed", "tqdm"):
        sys.modules.setdefault(name, types.ModuleType(name))

    # agents package stub so pipeline.py can import agent classes
    agents_mod = types.ModuleType("agents")
    for cls in ("RequirementsAgent", "ArchitectureAgent", "DataModelerAgent",
                "DiagramAgent", "CriticAgent"):
        setattr(agents_mod, cls, MagicMock)
    sys.modules.setdefault("agents", agents_mod)

    critic_mod = types.ModuleType("agents.critic_agent")
    critic_mod.extract_action_items = lambda text: "- Fix security gaps\n- Add rate limiting"
    sys.modules.setdefault("agents.critic_agent", critic_mod)


_make_stubs()

from pipeline import Pipeline, PipelineResult, PASS_THRESHOLD  # noqa: E402


class TestPipelineResult(unittest.TestCase):

    def test_defaults(self):
        r = PipelineResult()
        self.assertEqual(r.requirements, "")
        self.assertEqual(r.architecture, "")
        self.assertEqual(r.data_model, "")
        self.assertEqual(r.diagrams, "")
        self.assertEqual(r.critique, "")
        self.assertEqual(r.score, 0)
        self.assertEqual(r.rounds, 0)
        self.assertEqual(r.errors, {})

    def test_errors_are_independent(self):
        r1 = PipelineResult()
        r2 = PipelineResult()
        r1.errors["foo"] = "bar"
        self.assertNotIn("foo", r2.errors)

    def test_pass_threshold_value(self):
        self.assertEqual(PASS_THRESHOLD, 7)


class TestPipelineRun(unittest.TestCase):

    def _make_pipeline(self):
        p = Pipeline.__new__(Pipeline)
        p.requirements_agent = MagicMock()
        p.architecture_agent = MagicMock()
        p.data_modeler_agent = MagicMock()
        p.diagram_agent = MagicMock()
        p.critic_agent = MagicMock()
        return p

    def test_high_score_single_round(self):
        """Critic score >= 7 should produce a 1-round result."""
        p = self._make_pipeline()
        p.requirements_agent.run.return_value = "SRS text"
        p.architecture_agent.run.return_value = "Arch XML"
        p.data_modeler_agent.run.return_value = "Data model"
        p.critic_agent.run.return_value = ("Good critique", 8)
        p.diagram_agent.run.return_value = "Diagrams"

        result = p.run("build a ride-sharing app")

        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.score, 8)
        self.assertEqual(result.requirements, "SRS text")
        self.assertEqual(result.errors, {})
        # architecture agent should have been called exactly once
        self.assertEqual(p.architecture_agent.run.call_count, 1)

    def test_low_score_triggers_round_2(self):
        """Critic score < 7 should cause a second pass over arch + data model."""
        p = self._make_pipeline()
        p.requirements_agent.run.return_value = "SRS text"
        p.architecture_agent.run.return_value = "Arch XML"
        p.data_modeler_agent.run.return_value = "Data model"
        p.critic_agent.run.side_effect = [
            ("Weak critique", 5),   # round 1
            ("Better critique", 8), # round 2
        ]
        p.diagram_agent.run.return_value = "Diagrams"

        result = p.run("build a ride-sharing app")

        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.score, 8)
        self.assertEqual(p.architecture_agent.run.call_count, 2)
        self.assertEqual(p.data_modeler_agent.run.call_count, 2)

    def test_requirements_override_skips_agent(self):
        """Passing requirements_override should skip the requirements agent."""
        p = self._make_pipeline()
        p.architecture_agent.run.return_value = "Arch XML"
        p.data_modeler_agent.run.return_value = "Data model"
        p.critic_agent.run.return_value = ("Critique", 9)
        p.diagram_agent.run.return_value = "Diagrams"

        result = p.run("anything", requirements_override="My SRS")

        p.requirements_agent.run.assert_not_called()
        self.assertEqual(result.requirements, "My SRS")

    def test_requirements_failure_stops_pipeline(self):
        """An exception in the requirements agent should abort immediately."""
        p = self._make_pipeline()
        p.requirements_agent.run.side_effect = RuntimeError("API down")

        result = p.run("build something")

        self.assertIn("requirements", result.errors)
        self.assertEqual(result.architecture, "")
        p.architecture_agent.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

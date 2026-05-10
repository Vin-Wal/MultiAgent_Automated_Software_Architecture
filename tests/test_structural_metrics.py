"""
Unit tests for eval/structural.py.

Validates the four automated metrics: EARS compliance, NFR measurability,
section completeness, and decision quality. These run on string inputs only
and require no LLM or vector store.
"""
import sys
import types
import unittest


# ---------------------------------------------------------------------------
# Stubs — structural.py imports config and agents.base indirectly
# ---------------------------------------------------------------------------

def _make_stubs():
    config_mod = types.ModuleType("config")
    config_mod.cfg = type("cfg", (), {
        "LLM_API_KEY": "test",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_MODEL": "gpt-4o-mini",
        "MAX_TOKENS": 8192,
    })()
    sys.modules.setdefault("config", config_mod)

    for name in ("openai", "tenacity", "chromadb", "fastembed"):
        sys.modules.setdefault(name, types.ModuleType(name))


_make_stubs()

from eval.structural import (  # noqa: E402
    ears_compliance_rate,
    nfr_measurability,
    section_completeness,
    decision_quality,
)


SAMPLE_SRS = """
# Software Requirements Specification (SRS)

## 1. PROJECT OVERVIEW
Overview text.

## 2. FUNCTIONAL REQUIREMENTS
FR-001: The system shall match riders with available drivers when a ride is requested.
FR-002: The system shall send a notification when a driver accepts a request.
FR-003: The system should provide fast matching.

## 3. NON-FUNCTIONAL REQUIREMENTS
NFR-001: The system shall support 500000 concurrent users.
NFR-002: The system shall achieve 99.99% uptime.
NFR-003: Performance must be acceptable.

## 4. CONSTRAINTS
Constraint text.

## 5. STAKEHOLDERS
Stakeholder text.
"""

SAMPLE_ARCH_XML = """
<architecture_document>
  <design_decisions>
    <decision>
      <title>Event-driven messaging</title>
      <chosen>Kafka</chosen>
      <rationale>Handles high-throughput async events reliably.</rationale>
      <trade_offs>Adds operational overhead compared to REST.</trade_offs>
    </decision>
    <decision>
      <title>Database choice</title>
      <chosen>PostgreSQL</chosen>
      <rationale>ACID guarantees needed for payment records.</rationale>
      <trade_offs>Vertical scaling limit at very high write volumes.</trade_offs>
    </decision>
    <decision>
      <title>Caching</title>
      <chosen>Redis</chosen>
    </decision>
  </design_decisions>
</architecture_document>
"""


class TestEARSCompliance(unittest.TestCase):

    def test_full_compliance(self):
        srs = "FR-001: The system shall match riders when a ride is requested.\n" \
              "FR-002: The system shall notify users when a driver is assigned."
        rate = ears_compliance_rate(srs)
        self.assertAlmostEqual(rate, 1.0)

    def test_zero_compliance(self):
        srs = "The system should be fast.\nPerformance must be good.\nUsers want reliability."
        rate = ears_compliance_rate(srs)
        self.assertAlmostEqual(rate, 0.0)

    def test_partial_compliance(self):
        rate = ears_compliance_rate(SAMPLE_SRS)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)

    def test_empty_srs(self):
        self.assertEqual(ears_compliance_rate(""), 0.0)


class TestNFRMeasurability(unittest.TestCase):

    def test_measurable_nfr(self):
        srs = "NFR-001: The system shall support 500000 concurrent users.\n" \
              "NFR-002: The system shall achieve 99.99% uptime."
        rate = nfr_measurability(srs)
        self.assertAlmostEqual(rate, 1.0)

    def test_unmeasurable_nfr(self):
        srs = "NFR-001: Performance must be acceptable.\n" \
              "NFR-002: The system must be reliable."
        rate = nfr_measurability(srs)
        self.assertAlmostEqual(rate, 0.0)

    def test_mixed_measurability(self):
        rate = nfr_measurability(SAMPLE_SRS)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)

    def test_no_nfrs_returns_zero(self):
        srs = "FR-001: The system shall do something when triggered."
        self.assertEqual(nfr_measurability(srs), 0.0)


class TestSectionCompleteness(unittest.TestCase):

    def test_all_sections_present(self):
        score = section_completeness(SAMPLE_SRS)
        self.assertAlmostEqual(score, 1.0)

    def test_missing_sections(self):
        srs = "## 1. PROJECT OVERVIEW\nSome text.\n## 2. FUNCTIONAL REQUIREMENTS\nFR-001: ..."
        score = section_completeness(srs)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_empty_srs(self):
        self.assertEqual(section_completeness(""), 0.0)


class TestDecisionQuality(unittest.TestCase):

    def test_decisions_with_rationale_and_tradeoffs(self):
        # 2 out of 3 decisions have both rationale and trade_offs
        score = decision_quality(SAMPLE_ARCH_XML)
        self.assertAlmostEqual(score, 2 / 3, places=2)

    def test_no_decisions(self):
        xml = "<architecture_document><overview>text</overview></architecture_document>"
        self.assertEqual(decision_quality(xml), 0.0)

    def test_all_complete_decisions(self):
        xml = """
        <architecture_document>
          <design_decisions>
            <decision>
              <title>T1</title><chosen>X</chosen>
              <rationale>Because of A.</rationale>
              <trade_offs>Costs B.</trade_offs>
            </decision>
          </design_decisions>
        </architecture_document>
        """
        self.assertAlmostEqual(decision_quality(xml), 1.0)

    def test_malformed_xml_returns_zero(self):
        self.assertEqual(decision_quality("not xml at all"), 0.0)


if __name__ == "__main__":
    unittest.main()

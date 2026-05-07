"""
Structural metrics computed by regex and XML parsing — no LLM calls.
"""
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


# ── regex patterns ────────────────────────────────────────────────────────────

# Matches the EARS conditional clause that must follow "shall"
_EARS_PATTERN = re.compile(
    r"\bshall\b.{5,150}?\b(when|while|if|unless|where|as soon as|after)\b",
    re.IGNORECASE,
)
_FR_LINE   = re.compile(r"\bFR-\d+", re.IGNORECASE)
_NFR_LINE  = re.compile(r"\bNFR-\d+", re.IGNORECASE)

# Any digit adjacent to a recognised unit = measurable NFR
_MEASURABLE = re.compile(
    r"\d[\d,\.]*\s*"
    r"(%|ms\b|s\b|seconds?\b|minutes?\b|hours?\b|days?\b"
    r"|KB\b|MB\b|GB\b|TB\b|req/s|rps\b|TPS\b|rpm\b"
    r"|concurrent\b|users?\b|requests?\b|calls?\b)",
    re.IGNORECASE,
)

_SRS_SECTIONS = [
    "PROJECT OVERVIEW",
    "FUNCTIONAL REQUIREMENTS",
    "NON-FUNCTIONAL REQUIREMENTS",
    "EXTERNAL INTERFACES",
    "CONSTRAINTS",
]


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SRSMetrics:
    ears_total:     int = 0
    ears_compliant: int = 0
    nfr_total:      int = 0
    nfr_measurable: int = 0
    sections_found: list[str] = field(default_factory=list)
    sections_missing: list[str] = field(default_factory=list)

    @property
    def ears_rate(self) -> float:
        return self.ears_compliant / self.ears_total if self.ears_total else 0.0

    @property
    def nfr_measurable_rate(self) -> float:
        return self.nfr_measurable / self.nfr_total if self.nfr_total else 0.0

    @property
    def section_completeness(self) -> float:
        total = len(self.sections_found) + len(self.sections_missing)
        return len(self.sections_found) / total if total else 0.0


@dataclass
class ArchMetrics:
    xml_valid:                  bool = False
    component_count:            int  = 0
    total_decisions:            int  = 0
    decisions_with_rationale:   int  = 0
    decisions_with_tradeoffs:   int  = 0
    has_nfr_strategy:           bool = False
    has_data_flow:              bool = False

    @property
    def decision_quality(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return (self.decisions_with_rationale + self.decisions_with_tradeoffs) / (
            2 * self.total_decisions
        )


@dataclass
class DataModelMetrics:
    xml_valid:                  bool = False
    store_count:                int  = 0
    stores_with_justification:  int  = 0
    has_normalization_notes:    bool = False
    has_tradeoffs:              bool = False

    @property
    def justification_rate(self) -> float:
        return self.stores_with_justification / self.store_count if self.store_count else 0.0


@dataclass
class DiagramMetrics:
    arch_present:         bool = False
    sequence_present:     bool = False
    er_present:           bool = False
    arch_node_count:      int  = 0
    sequence_msg_count:   int  = 0


@dataclass
class StructuralReport:
    srs:        SRSMetrics       = field(default_factory=SRSMetrics)
    arch:       ArchMetrics      = field(default_factory=ArchMetrics)
    data_model: DataModelMetrics = field(default_factory=DataModelMetrics)
    diagrams:   DiagramMetrics   = field(default_factory=DiagramMetrics)

    def summary_score(self) -> float:
        """Aggregate 0–1 score across all structural dimensions."""
        scores = [
            self.srs.ears_rate,
            self.srs.nfr_measurable_rate,
            self.srs.section_completeness,
            float(self.arch.xml_valid),
            min(self.arch.component_count / 6, 1.0),   # 6+ components = full marks
            self.arch.decision_quality,
            float(self.arch.has_nfr_strategy),
            float(self.arch.has_data_flow),
            float(self.data_model.xml_valid),
            self.data_model.justification_rate,
            float(self.diagrams.arch_present),
            float(self.diagrams.sequence_present),
        ]
        return round(sum(scores) / len(scores), 4)

    def to_dict(self) -> dict:
        return {
            "ears_rate":            self.srs.ears_rate,
            "nfr_measurable_rate":  self.srs.nfr_measurable_rate,
            "section_completeness": self.srs.section_completeness,
            "ears_total":           self.srs.ears_total,
            "ears_compliant":       self.srs.ears_compliant,
            "nfr_total":            self.srs.nfr_total,
            "nfr_measurable":       self.srs.nfr_measurable,
            "sections_missing":     self.srs.sections_missing,
            "arch_xml_valid":       self.arch.xml_valid,
            "component_count":      self.arch.component_count,
            "total_decisions":      self.arch.total_decisions,
            "decision_quality":     self.arch.decision_quality,
            "has_nfr_strategy":     self.arch.has_nfr_strategy,
            "has_data_flow":        self.arch.has_data_flow,
            "dm_xml_valid":         self.data_model.xml_valid,
            "store_count":          self.data_model.store_count,
            "justification_rate":   self.data_model.justification_rate,
            "diagram_arch":         self.diagrams.arch_present,
            "diagram_sequence":     self.diagrams.sequence_present,
            "diagram_er":           self.diagrams.er_present,
            "arch_node_count":      self.diagrams.arch_node_count,
            "summary_score":        self.summary_score(),
        }


# ── scoring functions ─────────────────────────────────────────────────────────

def score_srs(srs_text: str) -> SRSMetrics:
    m = SRSMetrics()
    for line in srs_text.splitlines():
        if _FR_LINE.search(line) and "shall" in line.lower():
            m.ears_total += 1
            if _EARS_PATTERN.search(line):
                m.ears_compliant += 1
        if _NFR_LINE.search(line):
            m.nfr_total += 1
            if _MEASURABLE.search(line):
                m.nfr_measurable += 1

    upper = srs_text.upper()
    for section in _SRS_SECTIONS:
        if section in upper:
            m.sections_found.append(section)
        else:
            m.sections_missing.append(section)
    return m


def score_architecture(arch_xml: str) -> ArchMetrics:
    m = ArchMetrics()
    try:
        root = ET.fromstring(arch_xml.strip())
        m.xml_valid = True
    except ET.ParseError:
        # Fall back to regex counts so we still get partial signal
        m.component_count = len(re.findall(r"<component>", arch_xml))
        m.total_decisions = len(re.findall(r"<decision>", arch_xml))
        m.has_nfr_strategy = "<nfr_strategy>" in arch_xml
        m.has_data_flow = "<data_flow>" in arch_xml
        return m

    m.component_count  = len(root.findall(".//component"))
    m.has_nfr_strategy = root.find("nfr_strategy") is not None
    m.has_data_flow    = root.find("data_flow") is not None

    for decision in root.findall(".//decision"):
        m.total_decisions += 1
        rat = decision.find("rationale")
        if rat is not None and (rat.text or "").strip():
            m.decisions_with_rationale += 1
        trd = decision.find("trade_offs")
        if trd is not None and (trd.text or "").strip():
            m.decisions_with_tradeoffs += 1
    return m


def score_data_model(dm_xml: str) -> DataModelMetrics:
    m = DataModelMetrics()
    try:
        root = ET.fromstring(dm_xml.strip())
        m.xml_valid = True
    except ET.ParseError:
        m.store_count = len(re.findall(r"<strategy>", dm_xml))
        m.has_normalization_notes = "<normalization_notes>" in dm_xml
        m.has_tradeoffs = "<trade_offs>" in dm_xml
        return m

    strategies = root.findall(".//strategy")
    m.store_count = len(strategies)
    for s in strategies:
        why = s.find("why_chosen")
        if why is not None and len((why.text or "").strip()) > 20:
            m.stores_with_justification += 1

    m.has_normalization_notes = root.find("normalization_notes") is not None
    m.has_tradeoffs           = root.find("trade_offs") is not None
    return m


def score_diagrams(diagrams_xml: str) -> DiagramMetrics:
    m = DiagramMetrics()
    m.arch_present     = 'type="architecture"' in diagrams_xml
    m.sequence_present = 'type="sequence"' in diagrams_xml
    m.er_present       = 'type="er"' in diagrams_xml

    arch_block = re.search(
        r'type="architecture".*?```mermaid\s*\n(.*?)\n```',
        diagrams_xml, re.DOTALL,
    )
    if arch_block:
        m.arch_node_count = len(re.findall(r'\["[^"]+"\]', arch_block.group(1)))

    seq_block = re.search(
        r'type="sequence".*?```mermaid\s*\n(.*?)\n```',
        diagrams_xml, re.DOTALL,
    )
    if seq_block:
        m.sequence_msg_count = len(re.findall(r"->?>", seq_block.group(1)))

    return m


def score_all(
    requirements: str,
    architecture: str,
    data_model:   str,
    diagrams:     str,
) -> StructuralReport:
    return StructuralReport(
        srs        = score_srs(requirements),
        arch       = score_architecture(architecture),
        data_model = score_data_model(data_model),
        diagrams   = score_diagrams(diagrams),
    )

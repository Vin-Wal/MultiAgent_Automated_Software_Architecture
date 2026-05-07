"""
Automatic scenario generator.

Generates N diverse, realistic system briefs using the LLM, covering a wide
range of domains, scales, compliance requirements, and architectural challenges.
Saves to eval/generated_scenarios.json so they can be reused without
regenerating on every run.

Usage
-----
# Generate 80 scenarios and save
python -m eval.scenario_generator --count 80

# Regenerate even if file exists
python -m eval.scenario_generator --count 80 --force
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from agents.base import call_llm

OUTPUT_FILE = Path(__file__).parent / "generated_scenarios.json"

# ── domain pool — used to ensure diversity ────────────────────────────────────

DOMAINS = [
    # Healthcare & life sciences
    "hospital patient management", "telemedicine platform", "clinical trial management",
    "pharmacy inventory", "mental health app", "medical imaging analysis",
    # Fintech & banking
    "digital banking", "cryptocurrency exchange", "insurance claims processing",
    "peer-to-peer lending", "stock trading platform", "expense management SaaS",
    # E-commerce & retail
    "multi-vendor marketplace", "subscription box service", "luxury goods auction",
    "grocery delivery", "second-hand goods platform", "B2B procurement portal",
    # Logistics & supply chain
    "last-mile delivery tracking", "warehouse management", "fleet management",
    "cold-chain logistics", "freight forwarding", "customs clearance portal",
    # Education & edtech
    "online learning platform", "university student portal", "corporate LMS",
    "coding bootcamp platform", "K-12 school management", "language learning app",
    # IoT & smart systems
    "smart home automation", "industrial IoT monitoring", "smart city traffic",
    "energy grid management", "precision agriculture", "connected vehicle platform",
    # Social & communication
    "professional networking", "live streaming platform", "community forum",
    "event management platform", "dating app", "alumni network",
    # Government & public sector
    "e-government citizen portal", "tax filing system", "public transit ticketing",
    "emergency dispatch system", "court case management", "benefits administration",
    # SaaS & developer tools
    "CI/CD pipeline platform", "API gateway management", "multi-tenant CRM",
    "project management tool", "observability platform", "low-code app builder",
    # Gaming & entertainment
    "multiplayer online game backend", "game analytics platform", "esports tournament",
    "music streaming", "video-on-demand", "podcast hosting platform",
    # AI & data
    "ML model serving platform", "data lake management", "real-time analytics dashboard",
    "document intelligence SaaS", "recommendation engine", "fraud detection system",
    # Other
    "legal document management", "HR management system", "property management",
    "travel booking aggregator", "food delivery marketplace", "sports betting platform",
]

COMPLIANCE = [
    "HIPAA", "PCI-DSS", "GDPR", "SOC 2 Type II", "ISO 27001",
    "FedRAMP", "CCPA", "FERPA", "FINRA", "WCAG 2.1 AA",
]

SCALES = [
    "10,000 daily active users", "100,000 concurrent users",
    "1 million registered users", "50,000 transactions per minute",
    "500 enterprise clients", "10 million monthly active users",
    "real-time sub-100ms latency requirements", "99.99% uptime SLA",
]

# ── prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are generating realistic software project briefs for architecture evaluation.
Each brief is what a product manager would hand to a solutions architect.

For each scenario output EXACTLY this JSON structure (no markdown, pure JSON array):
[
  {
    "name": "snake_case_identifier_max_30_chars",
    "label": "Short Display Name (3-5 words)",
    "input": "Detailed 3-5 sentence brief describing: what the system does, the scale (numbers), key technical challenges, compliance requirements if any, and 2-3 specific functional requirements."
  },
  ...
]

Rules:
- Every brief must mention at least one specific number (users, TPS, latency, etc.)
- Every brief must name at least one specific technology challenge or constraint
- Vary domains, scales, and compliance requirements across scenarios
- Names must be unique snake_case, max 30 chars
- Do NOT repeat domains across the batch
"""


def _generate_batch(domains_batch: list[str], batch_num: int) -> list[dict]:
    """Generate one batch of scenarios for the given domains."""
    domain_list = "\n".join(f"- {d}" for d in domains_batch)
    prompt = (
        f"Generate exactly {len(domains_batch)} software project briefs, "
        f"one for each domain listed below. Use the domain to inspire the brief "
        f"but make it specific and realistic.\n\nDomains:\n{domain_list}\n\n"
        f"Output a JSON array of {len(domains_batch)} objects."
    )
    raw = call_llm(_SYSTEM, prompt, max_tokens=len(domains_batch) * 200)

    # Extract JSON array
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        print(f"  [batch {batch_num}] failed to parse JSON, retrying ...")
        return []

    try:
        items = json.loads(m.group(0))
        return [
            {
                "name":  str(item.get("name",  f"scenario_{batch_num}_{i}"))[:30],
                "label": str(item.get("label", f"Scenario {batch_num}-{i}")),
                "input": str(item.get("input", "")),
            }
            for i, item in enumerate(items)
            if item.get("input")
        ]
    except json.JSONDecodeError as e:
        print(f"  [batch {batch_num}] JSON decode error: {e}")
        return []


def generate_scenarios(count: int = 80) -> list[dict]:
    """
    Generate `count` diverse scenarios. Domains are sampled without replacement
    to maximise diversity. Calls LLM in batches of 10.
    """
    import random
    random.seed(42)

    # Build a domain list of exactly `count` entries (cycle if needed)
    pool = DOMAINS.copy()
    random.shuffle(pool)
    while len(pool) < count:
        extra = DOMAINS.copy()
        random.shuffle(extra)
        pool.extend(extra)
    selected_domains = pool[:count]

    BATCH_SIZE = 10
    all_scenarios: list[dict] = []
    batches = [selected_domains[i:i+BATCH_SIZE]
               for i in range(0, len(selected_domains), BATCH_SIZE)]

    print(f"  Generating {count} scenarios in {len(batches)} batches of {BATCH_SIZE} ...")

    for i, batch in enumerate(batches, 1):
        print(f"  batch {i}/{len(batches)} ...", end=" ", flush=True)
        items = _generate_batch(batch, i)
        # Ensure unique names
        existing_names = {s["name"] for s in all_scenarios}
        for item in items:
            if item["name"] in existing_names:
                item["name"] = f"{item['name'][:27]}_{i}"
            all_scenarios.append(item)
            existing_names.add(item["name"])
        print(f"{len(items)} generated  (total: {len(all_scenarios)})")
        if i < len(batches):
            time.sleep(0.5)   # small pause to avoid rate limits

    return all_scenarios[:count]


def load_or_generate(count: int = 80, force: bool = False) -> list[dict]:
    """Load from cache if available, otherwise generate."""
    if OUTPUT_FILE.exists() and not force:
        scenarios = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        if len(scenarios) >= count:
            print(f"  Loaded {count} scenarios from cache ({OUTPUT_FILE.name})")
            return scenarios[:count]
        print(f"  Cache has {len(scenarios)} scenarios, need {count} — regenerating ...")

    scenarios = generate_scenarios(count)
    OUTPUT_FILE.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    print(f"  Saved {len(scenarios)} scenarios → {OUTPUT_FILE}")
    return scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if cache exists")
    args = parser.parse_args()

    from agents.base import call_llm  # noqa — ensure importable
    scenarios = load_or_generate(args.count, force=args.force)
    print(f"\n  Done — {len(scenarios)} scenarios ready in {OUTPUT_FILE}")

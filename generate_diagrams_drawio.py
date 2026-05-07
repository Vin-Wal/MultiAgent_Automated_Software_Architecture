"""
Generate clean draw.io XML files — one per stage.
Open at https://app.diagrams.net → File → Export As → PNG (2×)

Run:
    python generate_diagrams_drawio.py
"""
from pathlib import Path

OUT = Path("diagrams")
OUT.mkdir(exist_ok=True)

# ── id counter ────────────────────────────────────────────────────────────────
_ctr = 2   # 0 and 1 reserved for root cells

def nid():
    global _ctr
    _ctr += 1
    return str(_ctr)

def reset():
    global _ctr
    _ctr = 2

# ── low-level builders ────────────────────────────────────────────────────────

def _attr(s: str) -> str:
    """Escape a string for use as an XML attribute value (not HTML-encoded label)."""
    return s.replace("&", "&amp;").replace('"', "&quot;")

def _wrap(body: str, W=1700, H=900) -> str:
    return (
        f'<mxfile host="app.diagrams.net"><diagram id="d" name="d">'
        f'<mxGraphModel dx="1200" dy="800" grid="0" gridSize="10" guides="1" '
        f'tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" '
        f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
        f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
        f'{body}'
        f'</root></mxGraphModel></diagram></mxfile>'
    )

def _geo(x, y, w, h, rel=0) -> str:
    r = ' relative="1"' if rel else ""
    return f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}"{r} as="geometry"/>'

def box(label_html: str, x, y, w, h, fill, stroke,
        font=11, bold=True, fc="#ffffff", rounding=10, shadow=1) -> tuple:
    """Solid coloured box. label_html is raw HTML (already entity-encoded for XML)."""
    i = nid()
    fw = 1 if bold else 0
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"fontColor={fc};fontSize={font};fontStyle={fw};"
        f"arcSize={rounding};shadow={shadow};"
    )
    xml = (
        f'<mxCell id="{i}" value="{label_html}" style="{style}" '
        f'vertex="1" parent="1">{_geo(x, y, w, h)}</mxCell>'
    )
    return i, xml

def outline_box(label_html: str, x, y, w, h, fill, stroke,
                font=10, bold=False, fc="#333333", rounding=10) -> tuple:
    """Light outline box."""
    i = nid()
    fw = 1 if bold else 0
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"fontColor={fc};fontSize={font};fontStyle={fw};"
        f"arcSize={rounding};shadow=0;"
    )
    xml = (
        f'<mxCell id="{i}" value="{label_html}" style="{style}" '
        f'vertex="1" parent="1">{_geo(x, y, w, h)}</mxCell>'
    )
    return i, xml

def icon_box(label_html: str, x, y, w, h, shape, fill, stroke,
             font=10, fc="#333333") -> tuple:
    """Shape-icon cell (image-style, label below)."""
    i = nid()
    style = (
        f"shape={shape};html=1;whiteSpace=wrap;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"fontColor={fc};fontSize={font};fontStyle=1;"
        f"verticalAlign=bottom;shadow=0;"
    )
    xml = (
        f'<mxCell id="{i}" value="{label_html}" style="{style}" '
        f'vertex="1" parent="1">{_geo(x, y, w, h)}</mxCell>'
    )
    return i, xml

def cylinder(label_html: str, x, y, w, h, fill, stroke, fc="#ffffff", font=11) -> tuple:
    i = nid()
    style = (
        f"shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"fontColor={fc};fontSize={font};fontStyle=1;shadow=1;"
    )
    xml = (
        f'<mxCell id="{i}" value="{label_html}" style="{style}" '
        f'vertex="1" parent="1">{_geo(x, y, w, h)}</mxCell>'
    )
    return i, xml

def doc_shape(label_html: str, x, y, w, h, fill, stroke, fc="#333333", font=10) -> tuple:
    i = nid()
    style = (
        f"shape=mxgraph.flowchart.document;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"fontColor={fc};fontSize={font};fontStyle=1;shadow=1;"
    )
    xml = (
        f'<mxCell id="{i}" value="{label_html}" style="{style}" '
        f'vertex="1" parent="1">{_geo(x, y, w, h)}</mxCell>'
    )
    return i, xml

def section_header(text: str, x, y, w, h, fill, stroke) -> tuple:
    i = nid()
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=0;"
        f"fontColor=#ffffff;fontSize=12;fontStyle=1;"
        f"arcSize=4;verticalAlign=middle;"
    )
    xml = (
        f'<mxCell id="{i}" value="{text}" style="{style}" '
        f'vertex="1" parent="1">{_geo(x, y, w, h)}</mxCell>'
    )
    return i, xml

def section_bg(x, y, w, h, fill, stroke) -> tuple:
    i = nid()
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
        f"arcSize=4;opacity=30;"
    )
    xml = (
        f'<mxCell id="{i}" value="" style="{style}" '
        f'vertex="1" parent="1">{_geo(x, y, w, h)}</mxCell>'
    )
    return i, xml

def formula(text: str, x, y, w, h) -> tuple:
    i = nid()
    style = (
        f"rounded=1;whiteSpace=wrap;html=1;"
        f"fillColor=#f5f5f5;strokeColor=#666666;strokeWidth=1.5;"
        f"fontColor=#1a1a2e;fontSize=10;fontStyle=5;"   # 5 = bold+italic (monospace-ish)
        f"arcSize=6;"
    )
    xml = (
        f'<mxCell id="{i}" value="{text}" style="{style}" '
        f'vertex="1" parent="1">{_geo(x, y, w, h)}</mxCell>'
    )
    return i, xml

def label_only(text: str, x, y, w, h, fc="#555555", font=9, italic=True) -> tuple:
    i = nid()
    style = (
        f"text;html=1;align=center;verticalAlign=middle;"
        f"fontColor={fc};fontSize={font};fontStyle={'2' if italic else '0'};"
    )
    xml = (
        f'<mxCell id="{i}" value="{text}" style="{style}" '
        f'vertex="1" parent="1">{_geo(x, y, w, h)}</mxCell>'
    )
    return i, xml

def arrow_between(src_id: str, tgt_id: str, label="",
                  color="#555555", dashed=False, lw=2,
                  ex=1, ey=0.5, nx=0, ny=0.5) -> str:
    i = nid()
    dash = "dashed=1;dashPattern=8 4;" if dashed else ""
    safe_label = _attr(label)
    style = (
        f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;"
        f"entryX={nx};entryY={ny};entryDx=0;entryDy=0;"
        f"strokeColor={color};strokeWidth={lw};fontSize=9;fontStyle=1;"
        f"fontColor={color};{dash}"
    )
    xml = (
        f'<mxCell id="{i}" value="{safe_label}" style="{style}" '
        f'edge="1" source="{src_id}" target="{tgt_id}" parent="1">'
        f'<mxGeometry relative="1" as="geometry"/></mxCell>'
    )
    return xml

def arrow_pts(x1, y1, x2, y2, label="", color="#555555", lw=2,
              mid_x=None) -> str:
    """Free arrow with explicit start/end points."""
    i = nid()
    safe_label = _attr(label)
    style = (
        f"rounded=1;strokeColor={color};strokeWidth={lw};"
        f"fontSize=9;fontStyle=1;fontColor={color};"
        f"endArrow=block;endFill=1;"
    )
    mid = ""
    if mid_x is not None:
        mid = f'<Array as="points"><mxPoint x="{mid_x}" y="{y1}"/><mxPoint x="{mid_x}" y="{y2}"/></Array>'
    xml = (
        f'<mxCell id="{i}" value="{safe_label}" style="{style}" '
        f'edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
        f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/>'
        f'{mid}'
        f'</mxGeometry></mxCell>'
    )
    return xml

# ── COLOURS ───────────────────────────────────────────────────────────────────
BLU  = "#1a73e8"   # requirements
IND  = "#4338ca"   # architecture
PUR  = "#7c3aed"   # data modeler
AMB  = "#b45309"   # diagram / amber
RED  = "#dc2626"   # critic / danger
GRN  = "#059669"   # success / chromadb
SLT  = "#475569"   # slate / neutral
LBL  = "#dbeafe"   # light blue bg
LIN  = "#e0e7ff"   # light indigo bg
LPU  = "#f3e8ff"   # light purple bg
LAM  = "#fef3c7"   # light amber bg
LRE  = "#fee2e2"   # light red bg
LGR  = "#d1fae5"   # light green bg
LSL  = "#f1f5f9"   # light slate bg


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1  —  Corpus Ingestion & Semantic Chunking
# ══════════════════════════════════════════════════════════════════════════════

def d1():
    reset()
    body = ""

    # ── header ────────────────────────────────────────────────────────────
    _, c = section_header("Stage 1 — Corpus Ingestion &amp; Semantic Chunking",
                          20, 15, 1660, 44, "#1e293b", "#1e293b")
    body += c

    # ══ SECTION A — Knowledge Base ═══════════════════════════════════════
    _, c = section_bg(20, 75, 370, 700, LBL, BLU)
    body += c
    _, c = label_only("A — Knowledge Base", 20, 82, 370, 28,
                       fc=BLU, font=11, italic=False)
    body += c

    corps = [
        ("requirements", "IEEE 29148 · EARS · ISO 25010",  BLU, LBL, 120),
        ("architecture",  "AWS WAF · Microservices · CQRS", IND, LIN, 270),
        ("data_modeler",  "CAP theorem · 3NF · DB design",  PUR, LPU, 420),
        ("critic",        "OWASP · NIST CSF · STRIDE",       RED, LRE, 570),
    ]
    corp_ids = []
    for name, sub, col, bg, y in corps:
        lbl = f"&lt;b&gt;{name}&lt;/b&gt;&lt;br/&gt;&lt;font style='font-size:9px'&gt;{sub}&lt;/font&gt;"
        ci, c = outline_box(lbl, 40, y, 330, 110, bg, col,
                            font=11, bold=False, fc=col)
        body += c
        corp_ids.append(ci)

    # ══ SECTION B — Semantic Chunker ══════════════════════════════════════
    _, c = section_bg(410, 75, 820, 700, LIN, IND)
    body += c
    _, c = label_only("B — Semantic Chunker (rag/chunker.py)", 410, 82, 820, 28,
                       fc=IND, font=11, italic=False)
    body += c

    p1_i, c = box("Pass 1 — Structural Split&lt;br/&gt;"
                  "&lt;font style='font-size:9px'&gt;Split raw text at paragraph boundaries (\\n\\n)&lt;/font&gt;",
                  460, 120, 700, 90, IND, IND, font=11)
    body += c

    _, c = formula("sim( pᵢ ,  pᵢ₊₁ )  =  ( eᵢ · eᵢ₊₁ )  /  ( ‖eᵢ‖ · ‖eᵢ₊₁‖ )  ≥  τ  =  0.72",
                   460, 248, 700, 46)
    body += c

    p2_i, c = box("Pass 2 — Cosine Similarity Merge&lt;br/&gt;"
                  "&lt;font style='font-size:9px'&gt;Merge consecutive paragraphs while sim ≥ τ&lt;/font&gt;",
                  460, 322, 700, 90, IND, IND, font=11)
    body += c

    _, c = formula("ē_merged  =  ( ē_cur  +  ē_new )  /  2          max_chars  =  800",
                   460, 450, 700, 46)
    body += c

    p3_i, c = box("Pass 3 — Overflow Split&lt;br/&gt;"
                  "&lt;font style='font-size:9px'&gt;Split any chunk &gt; 800 chars at nearest sentence boundary&lt;/font&gt;",
                  460, 524, 700, 90, IND, IND, font=11)
    body += c

    em_i, c = outline_box(
        "Embedding Model: &lt;b&gt;BAAI/bge-small-en-v1.5&lt;/b&gt;&lt;br/&gt;"
        "&lt;font style='font-size:9px'&gt;384-dimensional dense vectors · runs locally via fastembed&lt;/font&gt;",
        460, 650, 700, 90, LSL, SLT, font=10, bold=False, fc="#1e293b")
    body += c

    # ══ SECTION C — Vector Store ══════════════════════════════════════════
    _, c = section_bg(1250, 75, 410, 700, LGR, GRN)
    body += c
    _, c = label_only("C — Vector Store", 1250, 82, 410, 28,
                       fc=GRN, font=11, italic=False)
    body += c

    db_i, c = cylinder(
        "ChromaDB&lt;br/&gt;&lt;font style='font-size:9px'&gt;"
        "4 collections&lt;br/&gt;~500 chunks each&lt;br/&gt;"
        "HNSW index&lt;br/&gt;Lazy indexing&lt;/font&gt;",
        1300, 250, 310, 340, GRN, GRN, fc="#ffffff", font=12)
    body += c

    # ── arrows ────────────────────────────────────────────────────────────
    # All corpora → pass 1
    for ci in corp_ids:
        body += arrow_between(ci, p1_i, color=IND,
                              ex=1, ey=0.5, nx=0, ny=0.5)

    # pass1 → formula → pass2 → formula → pass3 → embed
    for src, tgt in [(p1_i, p2_i), (p2_i, p3_i), (p3_i, em_i)]:
        body += arrow_between(src, tgt, color=IND,
                              ex=0.5, ey=1, nx=0.5, ny=0)

    # embed → chromadb
    body += arrow_between(em_i, db_i, label="chunks + embeddings",
                          color=GRN, lw=3, ex=1, ey=0.5, nx=0, ny=0.5)

    return _wrap(body, W=1700, H=800)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2  —  Runtime RAG Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def d2():
    reset()
    body = ""

    _, c = section_header("Stage 2 — Runtime RAG Retrieval  (fires once per agent call)",
                          20, 15, 1660, 44, "#1e293b", "#1e293b")
    body += c

    # ── User Brief ────────────────────────────────────────────────────────
    brief_i, c = box(
        "USER BRIEF&lt;br/&gt;+&lt;br/&gt;Prior Outputs",
        30, 300, 160, 160, SLT, SLT, font=11)
    body += c
    _, c = label_only("(plain English&lt;br/&gt;system description)", 30, 468, 160, 40,
                       fc=SLT, font=8)
    body += c

    # ── 4 parallel queries ────────────────────────────────────────────────
    _, c = section_bg(220, 100, 280, 590, LIN, IND)
    body += c
    _, c = label_only("4 Parallel Queries", 220, 108, 280, 24, fc=IND, font=10, italic=False)
    body += c

    q_defs = [
        ("Broad", "General domain coverage", BLU, LBL, 140),
        ("Domain-focused", "Agent-specific terminology", IND, LIN, 270),
        ("Constraint-driven", "NFRs &amp; compliance", PUR, LPU, 400),
        ("Context-grounded", "From prior agent outputs", SLT, LSL, 530),
    ]
    q_ids = []
    for title, sub, col, bg, y in q_defs:
        lbl = f"&lt;b&gt;{title}&lt;/b&gt;&lt;br/&gt;&lt;font style='font-size:8px'&gt;{sub}&lt;/font&gt;"
        qi, c = outline_box(lbl, 232, y, 256, 90, bg, col, font=10, fc=col)
        body += c
        q_ids.append(qi)
        body += arrow_between(brief_i, qi, color=col, lw=1.5,
                              ex=1, ey=0.5, nx=0, ny=0.5)

    # ── Embedding model ───────────────────────────────────────────────────
    emb_i, c = box(
        "EMBEDDING&lt;br/&gt;MODEL&lt;br/&gt;&lt;br/&gt;"
        "&lt;font style='font-size:9px'&gt;BAAI/bge-small-en-v1.5&lt;br/&gt;384-dim vectors&lt;/font&gt;",
        580, 290, 180, 200, IND, IND, font=12)
    body += c
    _, c = formula("q_emb  =  embed(query)  ∈  ℝ³⁸⁴", 560, 520, 220, 38)
    body += c
    for qi in q_ids:
        body += arrow_between(qi, emb_i, color=IND, lw=1.5,
                              ex=1, ey=0.5, nx=0, ny=0.5)

    # ── ChromaDB ──────────────────────────────────────────────────────────
    db_i, c = cylinder(
        "ChromaDB&lt;br/&gt;HNSW Search&lt;br/&gt;"
        "&lt;font style='font-size:9px'&gt;agent&apos;s own collection only&lt;/font&gt;",
        850, 290, 180, 200, GRN, GRN, fc="#ffffff", font=11)
    body += c
    _, c = formula("score(q,c)  =  (q·c) / (‖q‖·‖c‖)   →   top-K chunks", 820, 520, 240, 38)
    body += c
    body += arrow_between(emb_i, db_i, label="query vector",
                          color=GRN, lw=2, ex=1, ey=0.5, nx=0, ny=0.5)

    # ── Deduplication ─────────────────────────────────────────────────────
    ded_i, c = box(
        "DEDUPLICATE&lt;br/&gt;by Chunk ID&lt;br/&gt;&lt;br/&gt;"
        "&lt;font style='font-size:9px'&gt;4 × K  →  unique set&lt;/font&gt;",
        1120, 290, 170, 200, AMB, AMB, font=11)
    body += c
    body += arrow_between(db_i, ded_i, label="top-K × 4",
                          color=AMB, lw=2, ex=1, ey=0.5, nx=0, ny=0.5)

    # ── Agent prompt ──────────────────────────────────────────────────────
    pmt_i, c = box(
        "AGENT PROMPT&lt;br/&gt;&lt;br/&gt;"
        "&lt;font style='font-size:9px'&gt;"
        "&lt;retrieved_context&gt;&lt;br/&gt;"
        "  ...relevant chunks...&lt;br/&gt;"
        "&lt;/retrieved_context&gt;&lt;/font&gt;",
        1380, 270, 200, 240, BLU, BLU, font=11)
    body += c
    body += arrow_between(ded_i, pmt_i, label="unique chunks",
                          color=BLU, lw=3, ex=1, ey=0.5, nx=0, ny=0.5)

    # footer
    _, c = label_only(
        "Each of the 5 agents independently queries its own dedicated collection. "
        "4 query strategies maximise recall for diverse phrasings of the same information need.",
        20, 640, 1660, 40, fc="#64748b", font=9)
    body += c

    return _wrap(body, W=1700, H=720)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3  —  Multi-Agent Pipeline & Critique Loop
# ══════════════════════════════════════════════════════════════════════════════

def d3():
    reset()
    body = ""

    _, c = section_header(
        "Stage 3 — Multi-Agent Pipeline &amp; Critique Loop",
        20, 15, 1760, 44, "#1e293b", "#1e293b")
    body += c

    # agent definitions: label, color, x-centre
    AGENTS = [
        ("REQUIREMENTS&lt;br/&gt;AGENT",  BLU, 130),
        ("ARCHITECTURE&lt;br/&gt;AGENT",  IND, 450),
        ("DATAMODELER&lt;br/&gt;AGENT",   PUR, 770),
        ("DIAGRAM&lt;br/&gt;AGENT",       AMB, 1090),
        ("CRITIC&lt;br/&gt;AGENT",        RED, 1410),
    ]

    OUTPUTS = [
        ("SRS&lt;br/&gt;(Markdown)",               BLU, LBL),
        ("Architecture&lt;br/&gt;(XML)",            IND, LIN),
        ("Data Model&lt;br/&gt;(XML)",              PUR, LPU),
        ("Diagrams&lt;br/&gt;(XML)",                AMB, LAM),
        ("Critique&lt;br/&gt;+ Score /10",          RED, LRE),
    ]

    CORPORA = [
        ("requirements&lt;br/&gt;corpus",  BLU, LBL, True),
        ("architecture&lt;br/&gt;corpus",  IND, LIN, True),
        ("data_modeler&lt;br/&gt;corpus",  PUR, LPU, True),
        (None, None, None, False),
        ("critic&lt;br/&gt;corpus",        RED, LRE, True),
    ]

    PASS_LABELS = [
        "SRS.md",
        "SRS.md + Arch.xml",
        "SRS + Arch + DataModel",
        "All 3 docs",
    ]

    AW, AH = 220, 130   # agent box size
    OW, OH = 190, 90    # output box size
    CW, CH = 180, 70    # corpus box size

    agent_ids, out_ids, corp_ids = [], [], []

    for (albl, acol, ax), (olbl, ocol, obg), (clbl, ccol, cbg, has_c) in \
            zip(AGENTS, OUTPUTS, CORPORA):

        # agent box
        ai, c = box(albl, ax - AW//2, 360, AW, AH, acol, acol, font=12)
        body += c
        agent_ids.append(ai)

        # output box (above)
        oi, c = outline_box(olbl, ax - OW//2, 200, OW, OH, obg, ocol,
                            font=10, bold=True, fc=ocol)
        body += c
        out_ids.append(oi)
        body += arrow_between(ai, oi, color=acol,
                              ex=0.5, ey=0, nx=0.5, ny=1, lw=1.5)

        # corpus box (below, dashed)
        if has_c:
            ci, c = outline_box(clbl, ax - CW//2, 560, CW, CH, cbg, ccol,
                                font=9, bold=False, fc=ccol)
            body += c
            corp_ids.append(ci)
            body += arrow_between(ci, ai, color=ccol, dashed=True,
                                  ex=0.5, ey=0, nx=0.5, ny=1, lw=1.5)
        else:
            corp_ids.append(None)

    # arrows between agents
    ub_i, c = box("USER&lt;br/&gt;BRIEF", 20, 380, 110, 90, SLT, SLT, font=11)
    body += c
    body += arrow_between(ub_i, agent_ids[0], color=SLT, lw=2.5,
                          ex=1, ey=0.5, nx=0, ny=0.5)

    for i, lbl in enumerate(PASS_LABELS):
        body += arrow_between(agent_ids[i], agent_ids[i+1], label=lbl,
                              color="#1e293b", lw=2.5,
                              ex=1, ey=0.5, nx=0, ny=0.5)

    # RAG label
    _, c = label_only(
        "↑  RAG retrieval — each agent queries its own collection  ↑",
        200, 650, 1200, 28, fc=SLT, font=9, italic=True)
    body += c

    # ══ CRITIQUE LOOP ═════════════════════════════════════════════════════
    _, c = section_bg(20, 710, 1760, 230, LRE, RED)
    body += c
    _, c = section_header(
        "Critique Loop  —  Round 2 triggered if Critic score &lt; θ = 7/10  (max 2 rounds)",
        20, 710, 1760, 38, RED, RED)
    body += c

    # score < 7 box
    dec_i, c = box("score &lt; 7 ?", 1510, 770, 180, 80, RED, RED, font=12)
    body += c
    body += arrow_pts(1500, 425, 1500, 770, label="score", color=RED, lw=2)

    # extract
    ext_i, c = box("Extract&lt;br/&gt;Action Items", 1170, 770, 180, 80, AMB, AMB, font=11)
    body += c
    body += arrow_between(dec_i, ext_i, label="YES", color=RED, lw=2,
                          ex=0, ey=0.5, nx=1, ny=0.5)

    # reinject
    rei_i, c = box("Reinject as&lt;br/&gt;prior_critique", 830, 770, 200, 80, IND, IND, font=11)
    body += c
    body += arrow_between(ext_i, rei_i, color=AMB, lw=2,
                          ex=0, ey=0.5, nx=1, ny=0.5)

    # re-run
    rer_i, c = box("Arch + DataModeler&lt;br/&gt;Re-run (Round 2)", 440, 770, 260, 80, BLU, BLU, font=11)
    body += c
    body += arrow_between(rei_i, rer_i, color=IND, lw=2,
                          ex=0, ey=0.5, nx=1, ny=0.5)

    # re-run → arch agent (loopback arrow with midpoints)
    body += arrow_pts(560, 770, 450, 490, label="Round 2", color=BLU, lw=2)
    body += arrow_pts(560, 770, 770, 490, color=BLU, lw=2)

    # final output
    fin_i, c = box("FINAL OUTPUT&lt;br/&gt;(score ≥ 7)  ✓",
                   1510, 880, 180, 50, GRN, GRN, font=11)
    body += c
    body += arrow_between(dec_i, fin_i, label="NO", color=GRN, lw=2,
                          ex=0.5, ey=1, nx=0.5, ny=0)

    # footer
    _, c = label_only(
        "Final artefacts:  SRS.md  +  Architecture.xml  +  DataModel.xml  +  Diagrams.xml  +  Critique.md",
        20, 955, 1760, 30, fc="#475569", font=9)
    body += c

    return _wrap(body, W=1800, H=1000)


# ── write files ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    files = [
        ("data_pipeline.drawio",  d1),
        ("rag_retrieval.drawio",   d2),
        ("agent_pipeline.drawio",  d3),
    ]
    for fname, fn in files:
        path = OUT / fname
        path.write_text(fn(), encoding="utf-8")
        print(f"  saved → {path}")

    print("""
Open at  https://app.diagrams.net
  File → Open from → Device → pick the .drawio file
  File → Export As → PNG → Scale 2x → Export
""")

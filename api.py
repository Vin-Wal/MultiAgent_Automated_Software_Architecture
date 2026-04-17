"""
api.py - FastAPI Web Interface
Run: uvicorn api:app --reload --port 8000
Open: http://localhost:8000
"""

import time
import zlib
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from main import run_pipeline

app = FastAPI(title="Multi-Agent Software Architect")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ArchRequest(BaseModel):
    user_input: str


class ArchResponse(BaseModel):
    user_input:   str
    requirements: str
    architecture: str
    data_model:   str
    critique:     str
    diagrams:     str
    elapsed_sec:  float


class DiagramRequest(BaseModel):
    uml: str


def plantuml_encode(uml_text: str) -> str:
    """Encode PlantUML text using the correct Deflate + base64 encoding."""
    compressed = zlib.compress(uml_text.encode("utf-8"), 9)[2:-4]
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    result = ""
    raw = base64.b64encode(compressed).decode("ascii")
    # Convert standard base64 to PlantUML's encoding
    std = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    for c in raw:
        if c == "=":
            continue
        idx = std.find(c)
        if idx >= 0:
            result += alphabet[idx]
        else:
            result += c
    return result


HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Multi-Agent Software Architect</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #2c3e50; }
        textarea { width: 100%; height: 120px; padding: 12px; font-size: 15px; border: 1px solid #ccc; border-radius: 6px; }
        button { background: #2c3e50; color: white; padding: 12px 28px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; margin-top: 10px; }
        button:hover { background: #34495e; }
        button:disabled { background: #95a5a6; cursor: not-allowed; }
        .section { background: white; border-radius: 8px; padding: 20px; margin-top: 20px; border-left: 4px solid #2c3e50; }
        .section h2 { color: #2c3e50; margin-top: 0; }
        pre { background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 13px; white-space: pre-wrap; }
        #status { margin-top: 15px; color: #7f8c8d; font-style: italic; }
        .diagram-img { max-width: 100%; border: 1px solid #ccc; border-radius: 4px; margin: 10px 0; display: block; }
        .diagram-label { font-weight: bold; color: #2c3e50; margin-top: 16px; }
    </style>
</head>
<body>
    <h1>Multi-Agent Software Architect</h1>
    <p>Describe your system. The pipeline generates:<br>
    <b>Requirements - Architecture - Data Model - Security Review - PlantUML Diagrams</b></p>
    <textarea id="input" placeholder="e.g. Build a ride-sharing app with GPS tracking, payments, and driver ratings..."></textarea>
    <br>
    <button id="btn" onclick="generate()">Generate Architecture</button>
    <div id="status"></div>
    <div id="output"></div>
    <script>
    function escapeHtml(text) {
        return text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    async function renderDiagrams(content) {
        var html = "";
        var parts = content.split("@startuml");
        for (var i = 1; i < parts.length; i++) {
            var endIdx = parts[i].indexOf("@enduml");
            if (endIdx === -1) continue;
            var uml = "@startuml" + parts[i].substring(0, endIdx + 7);
            try {
                var resp = await fetch("/encode_diagram", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({uml: uml})
                });
                var data = await resp.json();
                html += "<p class='diagram-label'>Diagram " + i + "</p>";
                html += "<img class='diagram-img' src='https://www.plantuml.com/plantuml/png/" + data.encoded + "' />";
            } catch(e) {
                html += "<p style='color:red'>Could not render diagram " + i + "</p>";
            }
        }
        return html;
    }

    function section(title, content) {
        return "<div class='section'><h2>" + title + "</h2><pre>" + escapeHtml(content) + "</pre></div>";
    }

    async function generate() {
        var input = document.getElementById("input").value.trim();
        if (!input) { alert("Please describe your system."); return; }
        var btn = document.getElementById("btn");
        btn.disabled = true;
        btn.textContent = "Generating...";
        document.getElementById("status").textContent = "Running 5 agents... (~2-4 minutes)";
        document.getElementById("output").innerHTML = "";
        try {
            var resp = await fetch("/generate", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({user_input: input})
            });
            if (!resp.ok) throw new Error((await resp.json()).detail);
            var data = await resp.json();
            document.getElementById("status").textContent = "Completed in " + data.elapsed_sec.toFixed(1) + "s";

            var html = section("Requirements", data.requirements);
            html += section("Architecture", data.architecture);
            html += section("Data Model", data.data_model);
            html += section("Critique", data.critique);

            var diagSection = "<div class='section'><h2>Diagrams</h2>";
            diagSection += await renderDiagrams(data.diagrams);
            diagSection += "<details><summary style='cursor:pointer;color:#7f8c8d;margin-top:16px'>Show PlantUML source</summary><pre>" + escapeHtml(data.diagrams) + "</pre></details></div>";
            html += diagSection;

            document.getElementById("output").innerHTML = html;
        } catch(e) {
            document.getElementById("status").textContent = "Error: " + e.message;
        } finally {
            btn.disabled = false;
            btn.textContent = "Generate Architecture";
        }
    }
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML


@app.post("/encode_diagram")
def encode_diagram(req: DiagramRequest):
    """Encode PlantUML text server-side using correct Deflate compression."""
    try:
        encoded = plantuml_encode(req.uml)
        return {"encoded": encoded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=ArchResponse)
def generate(req: ArchRequest):
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input cannot be empty")
    start = time.time()
    try:
        result = run_pipeline(req.user_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ArchResponse(**result, elapsed_sec=round(time.time() - start, 2))


@app.get("/health")
def health():
    return {"status": "ok"}

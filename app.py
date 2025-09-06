#!/usr/bin/env python3
import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain.docstore.document import Document  # older LC
except Exception:
    try:
        from langchain_core.documents import Document  # newer LC
    except Exception:
        raise

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# LLM providers
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

console = Console()

SYSTEM_INSTRUCTIONS = """You are a senior DevOps assistant.
Given raw CI/CD logs, you will:
1) Identify the primary failure reason(s) and the exact step that failed.
2) Extract key error lines (with line numbers if provided).
3) Suggest concrete fixes (commands/files to edit).
4) Classify severity (blocker/high/medium/low) and confidence (0-100%).

Respond with a concise, actionable summary for engineers.
"""

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are analyzing a CI/CD log chunk. Extract failures, error codes, and probable root causes.\n"
        "Be concise but capture important details.\n\n"
        "LOG CHUNK:\n{text}\n\n"
        "Chunk Summary:"
    ),
)

REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are combining multiple chunk summaries into a final, precise report.\n"
        "Follow this JSON schema exactly:\n"
        "{{\n"
        "  \"root_cause\": \"string\",\n"
        "  \"failed_step\": \"string\",\n"
        "  \"key_errors\": [\"string\", \"string\"],\n"
        "  \"suggested_fixes\": [\"string\", \"string\"],\n"
        "  \"severity\": \"blocker|high|medium|low\",\n"
        "  \"confidence\": 0-100,\n"
        "  \"human_readable\": \"markdown summary aimed at developers\"\n"
        "}}\n\n"
        "Chunk Summaries:\n{text}\n\n"
        "Final JSON:"
    ),
)

def get_llm():
    provider = os.getenv("PROVIDER", "openai").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    model = os.getenv("MODEL_NAME")  # optional; pick sensible defaults below

    if provider == "openai":
        if not model:
            model = "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=temperature)
    elif provider == "groq":
        if not model:
            model = "openai/gpt-oss-120b"
        return ChatGroq(model_name=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported PROVIDER: {provider} (use 'openai' or 'groq')")

def read_logs(paths: List[str]) -> str:
    all_text = []
    for p in paths:
        with open(p, "r", errors="ignore") as f:
            all_text.append(f"===== FILE: {p} =====\n" + f.read())
    return "\n\n".join(all_text)

def quick_highlights(text: str) -> dict:
    """
    Grep-style quick signals to complement LLM output.
    """
    lines = text.splitlines()
    patterns = [
        r"error[: ]", r"failed", r"failure", r"exception", r"traceback",
        r"npm ERR!", r"yarn ERR!", r"FATAL", r"panic:", r"segmentation fault",
        r"module not found", r"cannot find module", r"no such file or directory"
    ]
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    hits = [f"{i+1}: {ln}" for i, ln in enumerate(lines) if rx.search(ln)]
    return {
        "error_lines": hits[:50],  # cap
        "total_lines": len(lines),
        "error_like_count": len(hits)
    }

def summarize_logs(raw_text: str) -> dict:
    # Split logs into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400
    )
    docs = [Document(page_content=ch) for ch in splitter.split_text(raw_text)]

    llm = get_llm()

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=REDUCE_PROMPT,
        return_intermediate_steps=False,
        verbose=False,
    )

    result = chain.invoke({"input_documents": docs})
    # Try to parse JSON from the LLM
    text = result["output_text"].strip()
    # If the model wrapped JSON in markdown fences, strip them
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: embed as human_readable
        data = {
            "root_cause": "n/a",
            "failed_step": "n/a",
            "key_errors": [],
            "suggested_fixes": [],
            "severity": "medium",
            "confidence": 60,
            "human_readable": text
        }
    return data

def main():
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="LangChain CI Log Summarizer")
    parser.add_argument("--log", "-l", nargs="+", required=True, help="Path(s) to CI/CD log file(s)")
    parser.add_argument("--outdir", "-o", default="out", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw = read_logs(args.log)
    quick = quick_highlights(raw)
    data = summarize_logs(raw)

    now = datetime.utcnow().strftime("%Y-%m-%d_%H%M%SZ")
    json_path = Path(args.outdir) / f"summary_{now}.json"
    md_path = Path(args.outdir) / f"summary_{now}.md"
    hi_path = Path(args.outdir) / f"highlights_{now}.txt"

    with open(json_path, "w") as f:
        json.dump({"llm": data, "quick": quick}, f, indent=2)

    md = f"# CI/CD Log Summary ({now} UTC)\n\n"
    md += f"**Severity:** {data.get('severity','')}  \n"
    md += f"**Confidence:** {data.get('confidence','')}%\n\n"
    md += f"**Failed Step:** {data.get('failed_step','')}\n\n"
    md += "## Root Cause\n\n" + data.get("root_cause","") + "\n\n"
    md += "## Key Errors\n\n" + "\n".join(f"- {e}" for e in data.get("key_errors", [])) + "\n\n"
    md += "## Suggested Fixes\n\n" + "\n".join(f"- {e}" for e in data.get("suggested_fixes", [])) + "\n\n"
    md += "## Human Summary\n\n" + data.get("human_readable","") + "\n"
    with open(md_path, "w") as f:
        f.write(md)

    with open(hi_path, "w") as f:
        f.write("Quick Error-like Lines (first 50):\n")
        for line in quick["error_lines"]:
            f.write(line + "\n")

    console.print(Panel.fit("✅ Analysis complete", style="bold green"))
    console.print(f"[bold]JSON:[/bold] {json_path}")
    console.print(f"[bold]Markdown:[/bold] {md_path}")
    console.print(f"[bold]Highlights:[/bold] {hi_path}\n")

    console.print(Panel.fit("Markdown Preview", style="cyan"))
    console.print(Markdown(md))

if __name__ == "__main__":
    main()

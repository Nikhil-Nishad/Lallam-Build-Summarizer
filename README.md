# LangChain CI Log Summarizer (Quick Demo)

This is a tiny, interview-ready demo that shows how **LangChain** can be used in **DevOps/CI** to analyze and summarize build logs with actionable fixes.

## 1) Setup (10 minutes)

```bash
# 1. Create a virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env, set PROVIDER=openai or groq and the corresponding API key
```

## 2) Run on the provided sample log

```bash
python app.py --log sample_ci_log.txt --outdir out
```

Outputs:
- `out/summary_*.json` → structured JSON (root cause, failed step, fixes)
- `out/summary_*.md` → nicely formatted summary (use in PR comments / Slack)
- `out/highlights_*.txt` → grep-like error lines

## 3) Run on your CI logs

Pass multiple files if you like:
```bash
python app.py --log path/to/github_actions.log path/to/jest.txt --outdir out
```

## 4) Optional: Wire into GitHub Actions

Add a job step after your build/test steps:

```yaml
- name: Summarize CI logs with LangChain
  run: |
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    echo "PROVIDER=groq" >> .env
    echo "GROQ_API_KEY=${ secrets.GROQ_API_KEY }" >> .env
    python app.py --log $GITHUB_WORKSPACE/ci.log --outdir out

- name: Upload summary artifact
  uses: actions/upload-artifact@v4
  with:
    name: ci-summary
    path: out/
```

(You can also post `out/summary_*.md` as a PR comment using `gh api` or a bot account.)

## 5) What to say in the interview

- "I built a **LangChain-based CI assistant** that summarizes logs, pinpoints the failed step, and proposes fixes. It supports OpenAI or Groq backends and handles long logs via **map-reduce summarization** with chunking."
- "This can be extended with **agentic actions**: on failure, auto-open a GitHub issue, create a PR adding the missing dependency, or trigger a rollback in Kubernetes."

Good luck!

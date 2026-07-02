import csv
import json
import os
import subprocess
import sys
import tempfile

import gradio as gr

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_candidates.json")
RANK_SCRIPT  = os.path.join(os.path.dirname(__file__), "rank.py")
TEMP_DIR     = tempfile.gettempdir()


def load_sample():
    if os.path.exists(SAMPLE_PATH):
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def to_jsonl(raw_bytes, is_gz):
    if is_gz:
        return raw_bytes
    try:
        text = raw_bytes.decode("utf-8").strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return "\n".join(json.dumps(item) for item in parsed).encode("utf-8")
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return raw_bytes


def run_ranking(upload_file, json_text):
    # determine input source
    if upload_file is not None:
        src = upload_file.name if hasattr(upload_file, "name") else upload_file
        is_gz = src.endswith(".gz")
        with open(src, "rb") as f:
            raw = f.read()
        raw = to_jsonl(raw, is_gz)
        suffix = ".jsonl.gz" if is_gz else ".jsonl"
    elif json_text and json_text.strip():
        text = json_text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                text = "\n".join(json.dumps(item) for item in parsed)
        except json.JSONDecodeError:
            pass
        raw = text.encode("utf-8")
        suffix = ".jsonl"
    else:
        return [["error", "", "", "no input provided"]]

    # write input to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    # run ranker
    try:
        proc = subprocess.run(
            [sys.executable, RANK_SCRIPT, "--candidates", tmp_path, "--out", OUT_PATH],
            capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0:
            return [["error", "", "", proc.stderr or proc.stdout]]
    except subprocess.TimeoutExpired:
        return [["error", "", "", "ranking timed out (>120s)"]]
    except Exception as e:
        return [["error", "", "", str(e)]]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not os.path.exists(OUT_PATH):
        return [["error", "", "", "output file not produced"]]

    # parse CSV into table rows
    rows = []
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append([row["rank"], row["candidate_id"], row["score"], row["reasoning"]])

    if not rows:
        return [["info", "", "", "no candidates passed the filters"]]

    return rows


with gr.Blocks(title="Redrob Ranking Engine") as demo:
    gr.Markdown("**Redrob Candidate Ranking Engine** — Team: The defenders | Solo: Tanish M")

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="candidates file (.jsonl / .jsonl.gz / .json)")
            json_input = gr.Textbox(
                label="or paste JSON here",
                lines=8,
                value=load_sample()
            )
            run_btn = gr.Button("Run Ranking")

        with gr.Column(scale=2):
            output_table = gr.Dataframe(
                headers=["Rank", "Candidate ID", "Score", "Reasoning"],
                label="Results",
                wrap=True,
                interactive=False
            )

    run_btn.click(
        fn=run_ranking,
        inputs=[upload, json_input],
        outputs=[output_table]
    )

if __name__ == "__main__":
    demo.launch()
"""Run the golden dataset through the full agent and score it.

Usage:  python -m evals.run_evals            (needs OPENAI_API_KEY)
Exit code 1 if any threshold fails, so it can gate releases in CI.
"""
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

from evals import metrics
from evals.make_sample_pdf import main as make_pdf
from qpg.graph import build_graph, run
from qpg.ingest import build_index, document_sample, load_pdf, make_search
from qpg.llm import generator_llm, judge_llm
from qpg.prompts import PROMPT_VERSION

ROOT = Path(__file__).resolve().parent.parent


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    load_dotenv()
    golden = json.loads((ROOT / "evals" / "golden_set.json").read_text())
    pdf_path = ROOT / golden["document"]
    if not pdf_path.exists():
        make_pdf(pdf_path)

    pages = load_pdf(str(pdf_path))
    full_text = "\n".join(f"[page {p.metadata['page']}] {p.page_content}" for p in pages)
    index, _ = build_index(pages)
    graph = build_graph(generator_llm(), judge_llm(), make_search(index))
    judge = judge_llm()

    rows, guard_ok = [], []
    for case in golden["cases"]:
        start = time.time()
        with get_openai_callback() as usage:
            state = run(graph, case["instructions"], case["format_text"], case["difficulty"],
                        document_sample(pages))
        seconds = time.time() - start

        if "expect_blocked" in case:
            ok = bool(state.get("blocked")) == case["expect_blocked"]
            guard_ok.append(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {case['id']}: blocked={state.get('blocked')}")
            continue

        paper, key = state["paper"], state["answer_key"]
        scores = metrics.judge(judge, full_text, paper, key, case["difficulty"])
        row = {
            "id": case["id"],
            "format_pass": metrics.format_pass(paper, state.get("format_issues", []),
                                               case["expected_total_questions"],
                                               case["expected_total_marks"]),
            "must_cover": metrics.must_cover_rate(paper, case["must_cover"]),
            "citation_validity": metrics.citation_validity(key, state["evidence"]),
            "answer_alignment": metrics.answer_alignment(paper, key),
            "groundedness": scores.groundedness,
            "correctness": scores.correctness,
            "difficulty": scores.difficulty_match,
            "revisions": state.get("revisions", 0),
            "tokens": usage.total_tokens,
            "cost_usd": round(usage.total_cost, 5),
            "seconds": round(seconds, 1),
            "judge_reason": scores.reason,
        }
        rows.append(row)
        print(f"[done] {case['id']}: format={row['format_pass']} cover={row['must_cover']:.0%} "
              f"cite={row['citation_validity']:.0%} grounded={row['groundedness']}/5 "
              f"difficulty={row['difficulty']}/5 revisions={row['revisions']} "
              f"cost=${row['cost_usd']} {row['seconds']}s")

    summary = {
        "prompt_version": PROMPT_VERSION,
        "format_pass_rate": mean([r["format_pass"] for r in rows]),
        "must_cover_rate": mean([r["must_cover"] for r in rows]),
        "citation_validity": mean([r["citation_validity"] for r in rows]),
        "judge_groundedness": mean([r["groundedness"] for r in rows]),
        "judge_difficulty": mean([r["difficulty"] for r in rows]),
        "guardrail_accuracy": mean(guard_ok),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }
    failed = [m for m, t in golden["thresholds"].items() if summary[m] < t]

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        flag = "  <- FAIL" if k in failed else ""
        print(f"{k:20} {v:.2f}{flag}" if isinstance(v, float) else f"{k:20} {v}")

    out = ROOT / "evals" / "results"
    out.mkdir(exist_ok=True)
    (out / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps({"summary": summary, "cases": rows, "failed": failed}, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

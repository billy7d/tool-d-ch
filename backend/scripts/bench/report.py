"""Tổng hợp một hoặc nhiều lần chạy: thời gian theo loại lệnh gọi, token prompt mới/dùng lại, lý do chờ duyệt."""
import collections, json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"E:\tmp\td-bench\runs")
CJK = re.compile(r"[\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF\u0400-\u052F]")


def kind_of(c):
    system, user = c["system"], c["user"]
    if "semantic fidelity critic" in user[:700] or "semantic fidelity critic" in system[:200]: return "critic"
    if "naturalness critic" in user[:500] or "naturalness critic" in system[:200]: return "naturalness"
    if "editorial translator" in user[:500] or "editorial translator" in system[:200]: return "rewrite"
    if "Build structured source chapter memory" in system: return "chapter_memory"
    if "Summarize the key topic" in system: return "doc_profile"
    if "Câu cần sửa" in user: return "patch"
    if "Tuyệt đối không dùng chữ Hán" in system: return "script_fix"
    if "VALIDATION OR QA ISSUES" in user: return "repair"
    if "CURRENT SEGMENT" in user: return "segment"
    if "CURRENT SOURCE" in user or "Dịch tiêu đề" in user: return "translate"
    return "other"


def load(name):
    run = ROOT / name
    return (json.loads((run / "summary.json").read_text(encoding="utf-8")),
            json.loads((run / "calls.json").read_text(encoding="utf-8")),
            json.loads((run / "nodes.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    total = collections.Counter()
    for name in sys.argv[1:]:
        s, calls, nodes = load(name)
        print(f"\n=== {name} [{s.get('backend')}/{s.get('set')}]: {s['elapsed_s']:.0f}s, duyệt {s['approved']}, chờ {s['needs_review']}, ~{s['book_hours']}h/sách")
        agg = collections.defaultdict(collections.Counter)
        for c in calls:
            a = agg[kind_of(c)]
            cached = c.get("cached_tokens")
            a["n"] += 1; a["wall"] += c["wall"]; a["ptok"] += c["prompt_tokens"]; a["ps"] += c["prompt_s"]
            a["otok"] += c["output_tokens"]; a["os"] += c["output_s"]; a["cached"] += cached or 0; a["has_cached"] += cached is not None
        print(f"{'loại':15s}{'n':>5s}{'giây':>7s}{'%':>6s}{'ptok':>7s}{'mới':>7s}{'đọc s':>7s}{'otok':>6s}{'sinh s':>7s}")
        for kind, a in sorted(agg.items(), key=lambda kv: -kv[1]["wall"]):
            new = f"{(a['ptok'] - a['cached']) / a['n']:7.0f}" if a["has_cached"] else "      ?"
            print(f"{kind:15s}{a['n']:5d}{a['wall']:7.0f}{a['wall'] / s['elapsed_s'] * 100:6.1f}{a['ptok'] / a['n']:7.0f}{new}{a['ps'] / a['n']:7.1f}{a['otok'] / a['n']:6.0f}{a['os'] / a['n']:7.1f}")
        print(f"đọc prompt {sum(c['prompt_s'] for c in calls):.0f}s, sinh chữ {sum(c['output_s'] for c in calls):.0f}s, ngoài model {s['elapsed_s'] - sum(c['wall'] for c in calls):.0f}s")
        reasons = collections.Counter(tuple((n.get("draft") or {}).get("reasons") or ["?"]) for n in nodes if n["status"] != "TRANSLATED")
        print("lý do chờ duyệt:", reasons.most_common(12))
        foreign = [n["id"] for n in nodes if CJK.search((n.get("translation") or "") + ((n.get("draft") or {}).get("text") or "")) and not CJK.search(n["source"])]
        print("đoạn có chữ lạ trong bản dịch/nháp:", len(foreign), foreign[:10])
        for key in ("elapsed_s", "sample_chars", "approved", "needs_review", "nodes"):
            total[key] += s[key]
        total["book_chars"] = s["book_chars"]
    if len(sys.argv) > 2:
        print(f"\nTỔNG {len(sys.argv) - 1} lần chạy: {total['elapsed_s']:.0f}s, duyệt {total['approved']}/{total['nodes']}, chờ {total['needs_review']}, "
              f"~{total['elapsed_s'] / total['sample_chars'] * total['book_chars'] / 3600:.2f}h/sách")

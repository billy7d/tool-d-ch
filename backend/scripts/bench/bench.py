"""Chạy pipeline dịch thật trên vài cửa sổ đoạn của sách và ghi lại mọi lệnh gọi model.

Mỗi cửa sổ là PER đoạn liền nhau bắt đầu tại int(f * N). Dự án đo là DB mới trong runs/<run>/data,
không đụng dữ liệu thật. Thời gian cả sách ngoại suy theo số ký tự nguồn.
"""
import argparse, json, os, re, sqlite3, sys, time
from pathlib import Path

ROOT = Path(r"E:\tmp\td-bench")
SOURCE_DB = r"file:E:\tool-dịch\data\projects\a064fe21-1e43-4272-9548-b089cac5a77a\project.db?mode=ro"
SETS = {
    "a": "0.1,0.3,0.5,0.7,0.9", "b": "0.2,0.4,0.6,0.8,0.95", "h": "0.05,0.25,0.45,0.65,0.75",
    "f": "0.16,0.35,0.55,0.84,0.92", "hold1": "0.02,0.22,0.42,0.62,0.82", "hold2": "0.12,0.32,0.52,0.72,0.88",
}
parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--set", default="a", help="tên bộ cửa sổ hoặc danh sách phân số")
parser.add_argument("--per", type=int, default=8)
parser.add_argument("--repo", default=r"E:\tool-dịch\backend")
parser.add_argument("--backend", default="ollama", choices=["ollama", "llama"])
parser.add_argument("--url", default="")
parser.add_argument("--join", action="store_true", help="nối các đoạn bị ngắt trang trong từng cửa sổ")
parser.add_argument("--pairs", type=int, default=0, help="thay cho --set: N chỗ ngắt trang rải đều, mỗi cửa sổ gồm đoạn đứng trước và hai nửa")
parser.add_argument("--terms", default="", help="file JSON [{source_term, target_term}] nạp làm thuật ngữ mềm")
parser.add_argument("--book", default="", help="file EPUB/PDF/TXT nhập qua đường nhập thật của app thay cho DB nguồn")
parser.add_argument("--like", default="", help="dịch lại đúng các đoạn nguồn của lượt này (khớp bỏ qua khoảng trắng, dấu ` và số chú thích)")
parser.add_argument("--doc-type", default="GENERAL", help="document_type của dự án khi dùng --book")
parser.add_argument("--model", default="qwen2.5:7b", help="tên model Ollama hoặc tên file GGUF trong tools/models (bỏ .gguf)")
parser.add_argument("--env", action="append", default=[], help="KEY=VALUE đặt trước khi nạp app")
args = parser.parse_args()

sys.stdout.reconfigure(encoding="utf-8")
run_dir = ROOT / "runs" / args.run
if run_dir.exists():
    sys.exit(f"run {args.run} đã tồn tại")
(run_dir / "data").mkdir(parents=True)
os.environ["TOOL_DICH_DATA_DIR"] = str(run_dir / "data")
for item in args.env:
    key, _, value = item.partition("=")
    os.environ[key] = value
sys.path.insert(0, args.repo)

if args.book:
    from app.services.reconstruction.structure_builder import StructureBuilder
    book_path = Path(args.book)
    suffix = book_path.suffix.lower()
    if suffix == ".epub":
        from app.services.extractor.epub_extractor import EPUBExtractor as Extractor
    elif suffix == ".pdf":
        from app.services.extractor.pdf_extractor import PDFExtractor as Extractor
    else:
        from app.services.extractor.text_extractor import TextExtractor as Extractor
    extracted = Extractor(run_dir / "import").extract_document(book_path)
    document = StructureBuilder.build_canonical_document("bench", book_path.name, extracted["pages"], extracted["assets"])
    rows = [
        (node.type.value, node.content, json.dumps(node.source_mapping.model_dump()))
        for chapter in document.chapters for node in chapter.nodes
        if node.type.value in ("paragraph", "heading", "footnote")
    ]
    project_row = (document.metadata.title, args.doc_type.upper(), "BALANCED", "BALANCED")
else:
    source = sqlite3.connect(SOURCE_DB, uri=True)
    rows = source.execute(
        "select n.node_type, n.content, n.source_mapping from nodes n join chapters c on c.id=n.chapter_id "
        "where n.node_type in ('paragraph','heading','footnote') order by c.order_index, n.order_index"
    ).fetchall()
    project_row = source.execute("select title, document_type, translation_mode, qa_level from projects").fetchone()
book_chars = sum(len(content) for _, content, _ in rows)
fractions = [float(value) for value in SETS.get(args.set, args.set).split(",")]
picked = []
windows = [rows[int(fraction * len(rows)):int(fraction * len(rows)) + args.per] for fraction in fractions]
if args.pairs:
    from app.services.reconstruction.page_break_joiner import Fragment, find_joins
    maps = [json.loads(mapping or "{}") for _, _, mapping in rows]
    breaks = [extra for keeper, extra in find_joins([
        Fragment(kind, content, int(m.get("source_page_start") or 0), int(m.get("source_page_end") or 0))
        for (kind, content, _), m in zip(rows, maps)]) if keeper == extra - 1 and extra >= 2]
    step = len(breaks) / args.pairs
    windows = [rows[breaks[int(i * step + step / 2)] - 2:breaks[int(i * step + step / 2)] + 1] for i in range(args.pairs)]
if args.like:
    # Dịch đúng các đoạn của một lượt trước (so cũ/mới trên cùng nội dung dù bước nhập đã đổi cách tách đoạn).
    def _key(text):
        return re.sub("[^A-Za-z]", "", text or "").lower()
    position = {}
    for index, row in enumerate(rows):
        position.setdefault(_key(row[1]), index)
    earlier = json.loads((ROOT / "runs" / args.like / "nodes.json").read_text(encoding="utf-8"))
    matched = [position[_key(node["source"])] for node in earlier if _key(node["source"]) in position]
    print(f"--like {args.like}: khớp {len(matched)}/{len(earlier)} đoạn", flush=True)
    windows = [[rows[index] for index in matched]]
for window in windows:
    if args.join:
        from app.services.reconstruction.page_break_joiner import Fragment, find_joins, join_text
        pages = [json.loads(mapping or "{}") for _, _, mapping in window]
        texts = [content for _, content, _ in window]
        fragments = [Fragment(kind, text, int(page.get("source_page_start") or 0), int(page.get("source_page_end") or 0))
                     for (kind, _, _), text, page in zip(window, texts, pages)]
        absorbed = set()
        for keeper, extra in find_joins(fragments):
            texts[keeper] = join_text(texts[keeper], texts[extra])
            absorbed.add(extra)
        window = [(row[0], texts[i], row[2]) for i, row in enumerate(window) if i not in absorbed]
    picked.extend((kind, content) for kind, content, _ in window)

from app.db.engine import get_project_db  # noqa: E402
from app.db.models import ChapterModel, NodeModel, ProjectModel  # noqa: E402
from app.services.translation.worker import TranslationWorker  # noqa: E402

PROJECT_ID = "bench"
db = get_project_db(PROJECT_ID)
db.add(ProjectModel(id=PROJECT_ID, title=project_row[0], selected_model=args.model, document_type=project_row[1],
                    translation_mode=project_row[2], qa_level=project_row[3]))
db.add(ChapterModel(id="c0", project_id=PROJECT_ID, title="Chapter", order_index=0))
for index, (node_type, content) in enumerate(picked):
    db.add(NodeModel(id=f"n{index:04d}", project_id=PROJECT_ID, chapter_id="c0", node_type=node_type, content=content,
                     status="PENDING", order_index=index))
if args.terms:
    from app.db.models import GlossaryModel
    for number, item in enumerate(json.loads(Path(args.terms).read_text(encoding="utf-8"))):
        db.add(GlossaryModel(id=f"t{number}", project_id=PROJECT_ID, source_term=item["source_term"], target_term=item["target_term"],
                             preferred_target=item["target_term"], category="AUTO", domain="GENERAL", locked=False, lock_level="SOFT"))
db.commit()
db.close()

if args.backend == "llama":
    from app.services.translation.llama_server_manager import LlamaServerManager
    from app.services.translation.llama_server_provider import LlamaServerProvider
    provider = LlamaServerProvider(default_model=args.model, manager=LlamaServerManager())
    provider.ensure_server()  # nạp model trước khi bấm giờ; máy chủ được để lại cho lần đo sau
    print("llama-server slots:", provider.slots, flush=True)
else:
    from app.services.translation.ollama_provider import OllamaProvider
    provider = OllamaProvider(base_url=args.url or "http://127.0.0.1:11435", default_model=args.model)

calls = []
original_post = provider.session.post
started = time.time()


def recording_post(url, data=None, json=None, **kwargs):
    begin = time.time()
    response = original_post(url, data=data, json=json, **kwargs)
    wall = time.time() - begin
    entry = {"t": round(begin - started, 2), "wall": round(wall, 3), "status": response.status_code}
    try:
        body = response.json()
    except Exception:
        body = {}
    messages = (json or {}).get("messages") or []
    entry.update({
        "system": next((m["content"] for m in messages if m["role"] == "system"), ""),
        "user": "\n\n".join(m["content"] for m in messages if m["role"] != "system"),
        "format": bool((json or {}).get("format")),
        "num_predict": ((json or {}).get("options") or {}).get("num_predict"),
        "temperature": ((json or {}).get("options") or {}).get("temperature"),
        "reply": (body.get("message") or {}).get("content", ""),
        "prompt_tokens": body.get("prompt_eval_count", 0),
        "prompt_s": round(body.get("prompt_eval_duration", 0) / 1e9, 3),
        "output_tokens": body.get("eval_count", 0),
        "output_s": round(body.get("eval_duration", 0) / 1e9, 3),
        "load_s": round(body.get("load_duration", 0) / 1e9, 3),
        "cached_tokens": body.get("cached_tokens"),
        "slot": body.get("slot"),
    })
    calls.append(entry)
    return response


provider.session.post = recording_post
TranslationWorker().translate_project_sync(PROJECT_ID, model_name=args.model, provider=provider)
elapsed = time.time() - started

db = get_project_db(PROJECT_ID)
nodes = []
for node in db.query(NodeModel).order_by(NodeModel.order_index).all():
    nodes.append({"id": node.id, "type": node.node_type, "status": node.status, "source": node.content,
                  "translation": node.translated_content, "draft": (node.node_metadata or {}).get("draft_translation")})
db.close()
sample_chars = sum(len(content) for _, content in picked)
summary = {
    "run": args.run, "set": args.set, "book": args.book, "doc_type": project_row[1], "backend": args.backend, "elapsed_s": round(elapsed, 1), "nodes": len(nodes),
    "sample_chars": sample_chars, "book_chars": book_chars,
    "approved": sum(node["status"] == "TRANSLATED" for node in nodes),
    "needs_review": sum(node["status"] == "NEEDS_REVIEW" for node in nodes),
    "other": sum(node["status"] not in ("TRANSLATED", "NEEDS_REVIEW") for node in nodes),
    "speculation": [getattr(provider, "speculation_hits", 0), getattr(provider, "speculation_starts", 0)],
    "calls": len(calls), "book_hours": round(elapsed / sample_chars * book_chars / 3600, 2),
}
for name, payload in (("calls", calls), ("nodes", nodes), ("summary", summary)):
    (run_dir / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))

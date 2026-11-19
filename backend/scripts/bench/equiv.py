"""Cùng payload dạng Ollama, nhiệt độ 0: lưu câu trả lời của từng máy chủ để so từng ký tự."""
import json, sqlite3, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"E:\tool-dịch\backend")
from app.services.translation.llama_server_provider import LlamaServerProvider
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.provider_base import SEMANTIC_CRITIC_SCHEMA, output_token_cap
from app.services.translation.prompt_builder import PromptBuilder

backend, out = sys.argv[1], sys.argv[2]
db = sqlite3.connect(r"file:E:\tool-dịch\data\projects\a064fe21-1e43-4272-9548-b089cac5a77a\project.db?mode=ro", uri=True)
rows = db.execute("select content, translated_content from nodes where node_type='paragraph' and length(content) between 300 and 1100 order by order_index limit 16 offset 240").fetchall()
system = PromptBuilder.build_system_prompt("SELF_HELP", "BALANCED")
if backend == "llama":
    provider = LlamaServerProvider(base_url="http://127.0.0.1:11436", ban_foreign_scripts=(len(sys.argv) > 3))
else:
    provider = OllamaProvider(base_url="http://127.0.0.1:11435")
provider.remember_prompt_prefix(system)
results = []
for index, (source, translated) in enumerate(rows):
    user = f"CURRENT SOURCE\n{source}\n\nTranslate CURRENT SOURCE completely.\n{PromptBuilder.SINGLE_OUTPUT_CONTRACT}"
    payload = {"model": "qwen2.5:7b", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "stream": False, "keep_alive": "60m",
               "options": provider._options("qwen2.5:7b", 0.0, output_token_cap(source), top_p=0.85, repeat_penalty=1.05, seed=7)}
    body = provider.session.post(f"{provider.base_url}/api/chat", json=payload, timeout=180).json()
    results.append({"kind": "translate", "tokens": body.get("prompt_eval_count"), "out": body.get("eval_count"), "text": body["message"]["content"]})
    if index < 6:
        critic = json.dumps({"source": source, "candidate": translated}, ensure_ascii=False)
        payload = {"model": "qwen2.5:7b", "messages": provider.review_messages("You are a strict semantic fidelity critic. Return JSON.", critic, 450, "qwen2.5:7b"),
                   "format": SEMANTIC_CRITIC_SCHEMA, "stream": False, "keep_alive": "60m", "options": provider._options("qwen2.5:7b", 0.0, 450, seed=7)}
        body = provider.session.post(f"{provider.base_url}/api/chat", json=payload, timeout=180).json()
        results.append({"kind": "critic", "tokens": body.get("prompt_eval_count"), "out": body.get("eval_count"), "text": body["message"]["content"]})
json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(backend, len(results), "câu trả lời")

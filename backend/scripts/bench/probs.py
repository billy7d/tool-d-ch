"""Xác suất token của một máy chủ cho vài prompt cố định, để so tại chỗ hai máy chủ rẽ nhánh."""
import json, sqlite3, sys, requests
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"E:\tool-dịch\backend")
from app.services.translation.llama_server_provider import chatml_prompt
from app.services.translation.prompt_builder import PromptBuilder
backend, out, penalty = sys.argv[1], sys.argv[2], float(sys.argv[3])
db = sqlite3.connect(r"file:E:\tool-dịch\data\projects\a064fe21-1e43-4272-9548-b089cac5a77a\project.db?mode=ro", uri=True)
rows = db.execute("select content from nodes where node_type='paragraph' and length(content) between 300 and 1100 order by order_index limit 16 offset 240").fetchall()
system = PromptBuilder.build_system_prompt("SELF_HELP", "BALANCED")
results = []
for index in (2, 9, 10, 11, 12, 13):
    source = rows[index][0]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"CURRENT SOURCE\n{source}\n\nTranslate CURRENT SOURCE completely.\n{PromptBuilder.SINGLE_OUTPUT_CONTRACT}"}]
    if backend == "llama":
        body = requests.post("http://127.0.0.1:11436/completion", json={"prompt": chatml_prompt(messages), "n_predict": 90, "temperature": 0.0, "top_k": 40, "top_p": 0.85, "min_p": 0.0, "typical_p": 1.0,
                             "repeat_penalty": penalty, "repeat_last_n": 64, "seed": 7, "n_probs": 3, "cache_prompt": True, "id_slot": 0}, timeout=180).json()
        steps = [{"token": s.get("token"), "top": [(t.get("token"), round(t.get("logprob", 0), 3)) for t in (s.get("top_logprobs") or [])]} for s in body.get("completion_probabilities", [])]
        text = body["content"]
    else:
        body = requests.post("http://127.0.0.1:11435/api/chat", json={"model": "qwen2.5:7b", "messages": messages, "stream": False, "logprobs": True, "top_logprobs": 3,
                             "options": {"temperature": 0.0, "top_p": 0.85, "repeat_penalty": penalty, "num_predict": 90, "num_ctx": 4096, "num_gpu": 99, "seed": 7}}, timeout=180).json()
        steps = [{"token": s.get("token"), "top": [(t.get("token"), round(t.get("logprob", 0), 3)) for t in (s.get("top_logprobs") or [])]} for s in (body.get("logprobs") or [])]
        text = body["message"]["content"]
    results.append({"index": index, "text": text, "steps": steps})
json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(backend, penalty, [len(r["steps"]) for r in results])

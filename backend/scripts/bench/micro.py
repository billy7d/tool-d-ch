"""Đo nhanh llama-server: tốc độ đọc prompt, dùng lại cache, hai luồng cùng lúc, grammar cấm bộ chữ."""
import json, sqlite3, sys, threading, time
import requests
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"E:\tool-dịch\backend")
from app.services.translation.llama_server_provider import chatml_prompt, script_ban_grammar
from app.services.translation.prompt_builder import PromptBuilder

URL = "http://127.0.0.1:11436/completion"
db = sqlite3.connect(r"file:E:\tool-dịch\data\projects\a064fe21-1e43-4272-9548-b089cac5a77a\project.db?mode=ro", uri=True)
paras = [r[0] for r in db.execute("select content from nodes where node_type='paragraph' and length(content) between 600 and 900 limit 12 offset 100")]
system = PromptBuilder.build_system_prompt("SELF_HELP", "BALANCED")
filler = "\n\n".join(f"SOURCE PREVIOUS:\n{p}" for p in paras[4:8])


def call(slot, user, n=200, temperature=0.0, grammar=None, cache=True):
    body = {"prompt": chatml_prompt([{"role": "system", "content": system}, {"role": "user", "content": user}]),
            "n_predict": n, "cache_prompt": cache, "id_slot": slot, "temperature": temperature, "top_k": 40, "top_p": 0.85,
            "min_p": 0.0, "repeat_penalty": 1.05, "repeat_last_n": 64, "seed": 1}
    if grammar:
        body["grammar"] = grammar
    begin = time.time()
    data = requests.post(URL, json=body, timeout=300).json()
    t = data["timings"]
    return {"wall": round(time.time() - begin, 2), "prompt_total": data.get("tokens_evaluated"), "prompt_new": t["prompt_n"], "cache": t.get("cache_n"),
            "prompt_tps": round(t["prompt_per_second"]), "gen_n": t["predicted_n"], "gen_tps": round(t["predicted_per_second"], 1), "text": data["content"]}


def show(label, r):
    print(label, {k: v for k, v in r.items() if k != "text"})


user = lambda p: f"IMMEDIATE CONTEXT\n{filler}\n\nCURRENT SOURCE\n{p}\n\nTranslate CURRENT SOURCE completely.\nReturn plain text only."
r1 = call(0, user(paras[0]), cache=False); show("lạnh, không cache     ", r1)
r2 = call(0, user(paras[1])); show("cùng phần đầu, slot 0 ", r2)
r3 = call(1, "Summarize: " + paras[2], n=60); show("prompt khác ở slot 1  ", r3)
r4 = call(0, user(paras[3])); show("quay lại slot 0       ", r4)
r5 = call(0, user(paras[0])); print("trùng lượt lạnh ở nhiệt độ 0:", r5["text"] == r1["text"])
g = script_ban_grammar(system + user(paras[0]))
r6 = call(0, user(paras[0]), grammar=g); print("có grammar trùng không grammar:", r6["text"] == r1["text"], "| gen_tps", r6["gen_tps"])
r7 = call(0, "Viết một câu tiếng Trung Quốc bằng chữ Hán nói 'xin chào thế giới', rồi dịch sang tiếng Việt.", n=60, grammar=g); print("ép viết chữ Hán khi bị cấm:", repr(r7["text"][:120]))

results = {}
def worker(slot, p):
    results[slot] = call(slot, user(p), n=220)
call(0, user(paras[8]), n=1); call(1, user(paras[9]), n=1)  # làm nóng cache cả hai slot
begin = time.time(); a = call(0, user(paras[8]), n=220); b = call(1, user(paras[9]), n=220); seq = time.time() - begin
call(0, user(paras[10]), n=1); call(1, user(paras[11]), n=1)
threads = [threading.Thread(target=worker, args=(0, paras[10])), threading.Thread(target=worker, args=(1, paras[11]))]
begin = time.time(); [t.start() for t in threads]; [t.join() for t in threads]; par = time.time() - begin
print(f"tuần tự 2 lệnh: {seq:.1f}s ({a['gen_n']}+{b['gen_n']} token, {a['gen_tps']} tok/s) | song song: {par:.1f}s ({results[0]['gen_n']}+{results[1]['gen_n']} token, mỗi luồng {results[0]['gen_tps']}/{results[1]['gen_tps']} tok/s)")

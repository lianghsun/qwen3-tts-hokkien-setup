#!/usr/bin/env python3
"""
台語音頻批次生成器（Qwen3-TTS VoiceDesign）

一鍵執行（自動偵測 GPU 數）：
    python synthesize_audio.py --src-dir ./tw-hokkien-seed-text

常用選項：
    python synthesize_audio.py --n-gpus 3 --batch-size 8
    python synthesize_audio.py --max-samples 5   # 試跑
"""

import os, sys, io, json, sqlite3, logging, argparse
import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
HF_TOKEN      = os.getenv("HF_TOKEN", "")
HF_SRC_REPO   = "lianghsun/tw-hokkien-seed-text"
HF_AUDIO_REPO = "lianghsun/tw-hokkien-audio-qwen3"
SAMPLE_RATE   = 12000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 循環使用的聲色描述（台語/閩南語說話者多樣性）
# 描述風格參考官方範例：自然、情感豐富、具體
VOICE_DESIGNS = [
    "一位台灣中年女性，說話帶有濃厚的閩南語腔調，語調溫柔自然，如同在日常閒話家常，聲音親切而有溫度。",
    "一位台灣中年男性，說話帶有道地的閩南語口音，語氣穩重從容，像是長輩在細說往事，聲音渾厚有力。",
    "一位台灣年輕女性，閩南語腔調鮮明，語調輕快活潑，充滿朝氣，說話自然流暢，像在跟朋友聊天。",
    "一位台灣年輕男性，帶有台灣閩南語口音，說話清晰俐落，語氣輕鬆隨和，像在輕描淡寫地分享日常。",
    "一位台灣阿嬤，閩南語腔調道地純正，語速稍緩，語氣慈祥溫和，帶著歲月沉澱的從容感。",
    "一位台灣老先生，說話帶有濃厚閩南語腔，語速緩慢，聲音沙啞厚實，如同老故事說書人般娓娓道來。",
    "一位台灣年輕女性，閩南語口音清晰，聲音甜美明亮，語氣輕快，帶著年輕人特有的活力與熱情。",
    "一位台灣成熟男性，閩南語腔調自然流露，語速適中，說話條理清晰，帶有自信從容的氣息。",
]


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-gpus",        type=int, default=None)
    p.add_argument("--model-id",      default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    p.add_argument("--src-dir",       default=None)
    p.add_argument("--audio-dir",     default=os.path.join(SCRIPT_DIR, "audio_output"))
    p.add_argument("--db-path",       default=os.path.join(SCRIPT_DIR, "synthesis_checkpoint.db"))
    p.add_argument("--upload-every",  type=int, default=200)
    p.add_argument("--max-disk-gb",   type=float, default=20.0)
    p.add_argument("--batch-size",           type=int,   default=8)
    p.add_argument("--max-samples",          type=int,   default=0)
    p.add_argument("--temperature",          type=float, default=0.7,
                   help="主模型 temperature（預設 0.7，越低越穩定）")
    p.add_argument("--top-p",                type=float, default=0.9)
    p.add_argument("--repetition-penalty",   type=float, default=1.1)
    p.add_argument("--subtalker-temperature",type=float, default=0.7,
                   help="聲色風格 temperature（預設 0.7）")
    return p.parse_args()


# ── Database ───────────────────────────────────────────────────────────────────
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS synthesis (
            text_id   INTEGER,
            worker_id INTEGER,
            status    TEXT DEFAULT 'done',
            hf_batch  TEXT,
            error_msg TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (text_id, worker_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS worker_batches (
            worker_id INTEGER,
            batch_num INTEGER,
            hf_path   TEXT,
            PRIMARY KEY (worker_id, batch_num)
        )
    """)
    conn.commit()
    conn.close()


def get_done_ids(db_path, worker_id):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT text_id FROM synthesis WHERE worker_id=? AND status='done'", (worker_id,)
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def mark_done(db_path, text_id, worker_id, hf_batch=""):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO synthesis (text_id, worker_id, status, hf_batch) VALUES (?,?,?,?)",
        (text_id, worker_id, "done", hf_batch)
    )
    conn.commit()
    conn.close()


def mark_error(db_path, text_id, worker_id, error_msg):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO synthesis (text_id, worker_id, status, error_msg) VALUES (?,?,?,?)",
        (text_id, worker_id, "error", error_msg)
    )
    conn.commit()
    conn.close()


def next_batch_num(db_path, worker_id):
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT COALESCE(MAX(batch_num), 0) FROM worker_batches WHERE worker_id=?", (worker_id,)
    ).fetchone()
    conn.close()
    return row[0] + 1


def record_batch(db_path, worker_id, batch_num, hf_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO worker_batches (worker_id, batch_num, hf_path) VALUES (?,?,?)",
        (worker_id, batch_num, hf_path)
    )
    conn.commit()
    conn.close()


# ── Audio Utils ────────────────────────────────────────────────────────────────
def audio_to_wav_bytes(audio_np, sr):
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def dir_size_gb(path):
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / (1024 ** 3)


# ── HuggingFace Upload ─────────────────────────────────────────────────────────
def upload_batch(samples, worker_id, batch_num, audio_dir, db_path, hf_api):
    if not samples:
        return

    hf_meta = json.dumps({
        "info": {
            "features": {
                "audio":        {"_type": "Audio", "sampling_rate": SAMPLE_RATE},
                "text":         {"_type": "Value", "dtype": "string"},
                "duration":     {"_type": "Value", "dtype": "float32"},
                "sample_rate":  {"_type": "Value", "dtype": "int32"},
                "voice_design": {"_type": "Value", "dtype": "string"},
                "domain":       {"_type": "Value", "dtype": "string"},
                "subdomain":    {"_type": "Value", "dtype": "string"},
                "scene":        {"_type": "Value", "dtype": "string"},
                "emotion":      {"_type": "Value", "dtype": "string"},
                "accent":       {"_type": "Value", "dtype": "string"},
                "seed_text_id": {"_type": "Value", "dtype": "int32"},
            }
        }
    })

    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema([
        pa.field("audio",        audio_type),
        pa.field("text",         pa.string()),
        pa.field("duration",     pa.float32()),
        pa.field("sample_rate",  pa.int32()),
        pa.field("voice_design", pa.string()),
        pa.field("domain",       pa.string()),
        pa.field("subdomain",    pa.string()),
        pa.field("scene",        pa.string()),
        pa.field("emotion",      pa.string()),
        pa.field("accent",       pa.string()),
        pa.field("seed_text_id", pa.int32()),
    ], metadata={"huggingface": hf_meta})

    table = pa.table({
        "audio":        [{"bytes": s["audio_bytes"], "path": None} for s in samples],
        "text":         [s["text"]         for s in samples],
        "duration":     [s["duration"]     for s in samples],
        "sample_rate":  [s["sample_rate"]  for s in samples],
        "voice_design": [s["voice_design"] for s in samples],
        "domain":       [s["domain"]       for s in samples],
        "subdomain":    [s["subdomain"]    for s in samples],
        "scene":        [s["scene"]        for s in samples],
        "emotion":      [s["emotion"]      for s in samples],
        "accent":       [s["accent"]       for s in samples],
        "seed_text_id": [s["seed_text_id"] for s in samples],
    }, schema=schema)

    hf_path  = f"data/worker{worker_id}/batch_{batch_num:06d}.parquet"
    tmp_path = os.path.join(audio_dir, f"tmp_w{worker_id}_b{batch_num}.parquet")
    pq.write_table(table, tmp_path)
    hf_api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=hf_path,
        repo_id=HF_AUDIO_REPO,
        repo_type="dataset",
    )
    os.remove(tmp_path)
    record_batch(db_path, worker_id, batch_num, hf_path)
    logger.info("Worker%d uploaded batch %06d (%d samples) → %s",
                worker_id, batch_num, len(samples), hf_path)


# ── Worker ─────────────────────────────────────────────────────────────────────
def worker_fn(worker_id, n_workers, args_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    log = logging.getLogger(f"W{worker_id}")
    log.info("Starting on GPU %d", worker_id)

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args_dict["model_id"],
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    log.info("Model loaded: %s", args_dict["model_id"])

    db_path  = args_dict["db_path"]
    init_db(db_path)
    done_ids = get_done_ids(db_path, worker_id)
    log.info("Already done: %d", len(done_ids))

    from datasets import load_dataset
    src_dir = args_dict.get("src_dir")
    if src_dir:
        dataset = load_dataset("parquet",
                               data_files=str(Path(src_dir) / "data" / "*.parquet"),
                               split="train", streaming=True)
    else:
        dataset = load_dataset(HF_SRC_REPO, split="train", streaming=True,
                               token=args_dict["hf_token"])

    hf_api    = HfApi(token=args_dict["hf_token"])
    audio_dir = args_dict["audio_dir"]
    os.makedirs(audio_dir, exist_ok=True)

    local_samples = []
    batch_num     = next_batch_num(db_path, worker_id)
    batch_size    = args_dict["batch_size"]
    max_samples   = args_dict["max_samples"]
    pending       = []   # [(text_id, text, item)]
    batch_idx     = 0    # 用來循環 VOICE_DESIGNS

    def flush_pending():
        nonlocal batch_num, local_samples, batch_idx
        if not pending:
            return

        texts       = [p[1] for p in pending]
        voice_desc  = VOICE_DESIGNS[batch_idx % len(VOICE_DESIGNS)]
        instructs   = [voice_desc] * len(texts)

        try:
            wavs, sr = model.generate_voice_design(
                text=texts,
                language=["auto"] * len(texts),
                instruct=instructs,
                temperature=args_dict["temperature"],
                top_p=args_dict["top_p"],
                repetition_penalty=args_dict["repetition_penalty"],
                subtalker_temperature=args_dict["subtalker_temperature"],
            )
        except Exception as e:
            log.error("Batch failed (voice=%s): %s", voice_desc[:20], e)
            for text_id, _, _ in pending:
                mark_error(db_path, text_id, worker_id, str(e))
            pending.clear()
            batch_idx += 1
            return

        for i, (text_id, target_text, item) in enumerate(pending):
            try:
                audio_np = wavs[i]
                if isinstance(audio_np, torch.Tensor):
                    audio_np = audio_np.cpu().numpy()
                audio_np = audio_np.squeeze()
                duration = round(len(audio_np) / sr, 3)
                local_samples.append({
                    "audio_bytes":  audio_to_wav_bytes(audio_np, sr),
                    "text":         target_text,
                    "duration":     duration,
                    "sample_rate":  sr,
                    "voice_design": voice_desc,
                    "domain":       item.get("domain", ""),
                    "subdomain":    item.get("subdomain", ""),
                    "scene":        item.get("scene", ""),
                    "emotion":      item.get("emotion", ""),
                    "accent":       item.get("accent", ""),
                    "seed_text_id": text_id,
                })
                done_ids.add(text_id)
                mark_done(db_path, text_id, worker_id, f"batch_{batch_num:06d}")
            except Exception as e:
                log.error("text_id=%d post-process failed: %s", text_id, e)
                mark_error(db_path, text_id, worker_id, str(e))

        pending.clear()
        batch_idx += 1

        should_upload = (
            len(local_samples) >= args_dict["upload_every"] or
            dir_size_gb(audio_dir) >= args_dict["max_disk_gb"]
        )
        if should_upload:
            upload_batch(local_samples, worker_id, batch_num, audio_dir, db_path, hf_api)
            local_samples = []
            batch_num += 1

    for global_idx, item in enumerate(dataset):
        if global_idx % n_workers != worker_id:
            continue
        text_id = int(item.get("id", global_idx))
        if text_id in done_ids:
            continue
        target_text = item.get("text", "").strip()
        if not target_text:
            continue
        if max_samples and len(done_ids) >= max_samples:
            break
        pending.append((text_id, target_text, item))
        if len(pending) >= batch_size:
            flush_pending()

    flush_pending()
    if local_samples:
        upload_batch(local_samples, worker_id, batch_num, audio_dir, db_path, hf_api)

    log.info("Worker %d finished.", worker_id)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args      = parse_args()
    n_workers = args.n_gpus or torch.cuda.device_count()

    if n_workers == 0:
        logger.error("No CUDA GPUs found. Exiting.")
        sys.exit(1)

    logger.info("Starting Qwen3-TTS VoiceDesign synthesis on %d GPU(s)", n_workers)

    hf_api = HfApi(token=HF_TOKEN)
    repo_exists = False
    try:
        hf_api.repo_info(repo_id=HF_AUDIO_REPO, repo_type="dataset")
        repo_exists = True
    except Exception:
        hf_api.create_repo(repo_id=HF_AUDIO_REPO, repo_type="dataset", private=False)
        hf_api.update_repo_settings(repo_id=HF_AUDIO_REPO, repo_type="dataset", gated="manual")
        logger.info("Created HF repo: %s", HF_AUDIO_REPO)

    if not repo_exists:
        dataset_card = """\
---
language:
- nan
- zh
license: cc-by-4.0
task_categories:
- text-to-speech
tags:
- hokkien
- taiwanese
- tts
- qwen3
pretty_name: Taiwanese Hokkien TTS Audio (Qwen3-TTS VoiceDesign)
---

# Taiwanese Hokkien TTS Audio (Qwen3-TTS VoiceDesign)

台語（閩南語）合成語音資料集，由 [Qwen3-TTS-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) 生成。

使用 8 種聲色描述（台灣閩南語腔調）循環生成，搭配 `instruct` 參數指定台灣口音。
"""
        hf_api.upload_file(
            path_or_fileobj=dataset_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=HF_AUDIO_REPO,
            repo_type="dataset",
        )

    os.makedirs(args.audio_dir, exist_ok=True)
    init_db(args.db_path)

    args_dict = {
        "model_id":     args.model_id,
        "src_dir":      args.src_dir,
        "audio_dir":    args.audio_dir,
        "db_path":      args.db_path,
        "upload_every": args.upload_every,
        "max_disk_gb":  args.max_disk_gb,
        "batch_size":           args.batch_size,
        "max_samples":          args.max_samples,
        "temperature":          args.temperature,
        "top_p":                args.top_p,
        "repetition_penalty":   args.repetition_penalty,
        "subtalker_temperature":args.subtalker_temperature,
        "hf_token":             HF_TOKEN,
    }

    mp.set_start_method("spawn", force=True)
    processes = []
    for worker_id in range(n_workers):
        p = mp.Process(target=worker_fn, args=(worker_id, n_workers, args_dict))
        p.start()
        processes.append(p)
        logger.info("Started worker %d (PID %d)", worker_id, p.pid)

    for p in processes:
        p.join()

    logger.info("All workers finished.")


if __name__ == "__main__":
    main()

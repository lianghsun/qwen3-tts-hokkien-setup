#!/usr/bin/env python3
"""
台語音頻批次生成器（Qwen3-TTS）

一鍵執行（自動偵測 GPU 數）：
    python synthesize_audio.py --src-dir ./tw-hokkien-seed-text

常用選項：
    python synthesize_audio.py --n-gpus 2 --batch-size 16
    python synthesize_audio.py --max-samples 5   # 試跑
"""

import os, sys, io, json, sqlite3, logging, argparse, time
import numpy as np
import soundfile as sf
import pandas as pd
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
SAMPLE_RATE   = 12000  # Qwen3-TTS 固定輸出 12000 Hz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Qwen3-TTS CustomVoice 9 個預設說話者（循環使用）
SPEAKERS = [
    "Vivian",    # 明亮活潑女聲（中文）
    "Serena",    # 溫柔知性女聲（中文）
    "Uncle_Fu",  # 低沉醇厚男聲（中文）
    "Dylan",     # 清朗北京男聲（中文）
    "Eric",      # 四川男聲（中文）
    "Ryan",      # 英語男聲
    "Aiden",     # 美式男聲（英語）
    "Ono_Anna",  # 日語女聲
    "Sohee",     # 韓語女聲
]

# 閩南語生成指令
INSTRUCT = "请用闽南语发音。"


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-gpus",       type=int, default=None,
                   help="GPU 數量（預設自動偵測）")
    p.add_argument("--model-id",     default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                   help="HuggingFace model ID")
    p.add_argument("--src-dir",      default=None,
                   help="本地 seed-text 資料集目錄")
    p.add_argument("--audio-dir",    default=os.path.join(SCRIPT_DIR, "audio_output"))
    p.add_argument("--db-path",      default=os.path.join(SCRIPT_DIR, "synthesis_checkpoint.db"))
    p.add_argument("--upload-every", type=int, default=200,
                   help="每幾筆上傳一次 HF")
    p.add_argument("--max-disk-gb",  type=float, default=20.0)
    p.add_argument("--batch-size",   type=int, default=8,
                   help="每次 inference 批次大小（Qwen3-TTS 支援 batch）")
    p.add_argument("--max-samples",  type=int, default=0,
                   help="每個 worker 最多生幾筆就停（0 = 不限，用於試跑）")
    return p.parse_args()


# ── Database ───────────────────────────────────────────────────────────────────
def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS synthesis (
            text_id    INTEGER,
            worker_id  INTEGER,
            status     TEXT DEFAULT 'done',
            hf_batch   TEXT,
            error_msg  TEXT,
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


def get_done_ids(db_path: str, worker_id: int) -> set:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT text_id FROM synthesis WHERE worker_id=? AND status='done'", (worker_id,)
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def mark_done(db_path, text_id, worker_id, hf_batch=""):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO synthesis (text_id, worker_id, status, hf_batch)
        VALUES (?, ?, 'done', ?)
    """, (text_id, worker_id, hf_batch))
    conn.commit()
    conn.close()


def mark_error(db_path, text_id, worker_id, error_msg):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO synthesis (text_id, worker_id, status, error_msg)
        VALUES (?, ?, 'error', ?)
    """, (text_id, worker_id, error_msg))
    conn.commit()
    conn.close()


def next_batch_num(db_path: str, worker_id: int) -> int:
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
def audio_to_wav_bytes(audio_np: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def dir_size_gb(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / (1024 ** 3)


# ── HuggingFace Upload ─────────────────────────────────────────────────────────
def upload_batch(samples: list, worker_id: int, batch_num: int,
                 audio_dir: str, db_path: str, hf_api: HfApi):
    if not samples:
        return

    hf_meta = json.dumps({
        "info": {
            "features": {
                "audio":       {"_type": "Audio", "sampling_rate": SAMPLE_RATE},
                "text":        {"_type": "Value", "dtype": "string"},
                "duration":    {"_type": "Value", "dtype": "float32"},
                "sample_rate": {"_type": "Value", "dtype": "int32"},
                "speaker":     {"_type": "Value", "dtype": "string"},
                "instruct":    {"_type": "Value", "dtype": "string"},
                "domain":      {"_type": "Value", "dtype": "string"},
                "subdomain":   {"_type": "Value", "dtype": "string"},
                "scene":       {"_type": "Value", "dtype": "string"},
                "emotion":     {"_type": "Value", "dtype": "string"},
                "accent":      {"_type": "Value", "dtype": "string"},
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
        pa.field("speaker",      pa.string()),
        pa.field("instruct",     pa.string()),
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
        "speaker":      [s["speaker"]      for s in samples],
        "instruct":     [s["instruct"]     for s in samples],
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
def worker_fn(worker_id: int, n_workers: int, args_dict: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

    log = logging.getLogger(f"W{worker_id}")
    log.info("Starting on GPU %d", worker_id)

    # 載入 Qwen3-TTS 模型
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args_dict["model_id"],
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    log.info("Model loaded: %s", args_dict["model_id"])

    # 斷點
    db_path  = args_dict["db_path"]
    init_db(db_path)
    done_ids = get_done_ids(db_path, worker_id)
    log.info("Already done: %d", len(done_ids))

    # 讀取文本資料集
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
    pending       = []  # [(global_idx, text_id, item), ...]

    def flush_pending():
        nonlocal batch_num, local_samples
        if not pending:
            return

        texts    = [p[2].get("text", "") for p in pending]
        speakers = [SPEAKERS[p[1] % len(SPEAKERS)] for p in pending]
        instructs = [INSTRUCT] * len(pending)

        try:
            wavs, sr = model.generate_custom_voice(
                text=texts,
                language=["Chinese"] * len(texts),
                speaker=speakers,
                instruct=instructs,
            )
        except Exception as e:
            for _, text_id, _ in pending:
                mark_error(db_path, text_id, worker_id, str(e))
            log.error("Batch inference failed: %s", e)
            pending.clear()
            return

        for i, (global_idx, text_id, item) in enumerate(pending):
            try:
                audio_np = wavs[i]
                if isinstance(audio_np, torch.Tensor):
                    audio_np = audio_np.cpu().numpy()
                audio_np = audio_np.squeeze()
                duration = round(len(audio_np) / sr, 3)

                local_samples.append({
                    "audio_bytes":  audio_to_wav_bytes(audio_np, sr),
                    "text":         texts[i],
                    "duration":     duration,
                    "sample_rate":  sr,
                    "speaker":      speakers[i],
                    "instruct":     INSTRUCT,
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

        pending.append((global_idx, text_id, item))

        if len(pending) >= batch_size:
            flush_pending()

    # 收尾
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

    logger.info("Starting Qwen3-TTS synthesis on %d GPU(s)", n_workers)

    # 確保 HF repo 存在
    hf_api = HfApi(token=HF_TOKEN)
    repo_exists = False
    try:
        hf_api.repo_info(repo_id=HF_AUDIO_REPO, repo_type="dataset")
        repo_exists = True
        logger.info("HF repo already exists: %s", HF_AUDIO_REPO)
    except Exception:
        hf_api.create_repo(repo_id=HF_AUDIO_REPO, repo_type="dataset", private=False)
        hf_api.update_repo_settings(repo_id=HF_AUDIO_REPO, repo_type="dataset", gated="manual")
        logger.info("Created HF repo (public, gated/manual): %s", HF_AUDIO_REPO)

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
pretty_name: Taiwanese Hokkien TTS Audio (Qwen3-TTS)
---

# Taiwanese Hokkien TTS Audio (Qwen3-TTS)

台語（閩南語）合成語音資料集，由 [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) 批次生成。

## 資料來源

- **文本**：[lianghsun/tw-hokkien-seed-text](https://huggingface.co/datasets/lianghsun/tw-hokkien-seed-text)
- **生成模型**：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- **說話者**：9 個預設說話者循環（Vivian、Serena、Uncle_Fu、Dylan、Eric、Ryan、Aiden、Ono_Anna、Sohee）
- **指令**：`请用闽南语发音。`

## 欄位說明

| 欄位 | 說明 |
|------|------|
| `audio` | 合成音頻（WAV，12000 Hz） |
| `text` | 台語文本 |
| `duration` | 音頻長度（秒） |
| `sample_rate` | 12000 Hz |
| `speaker` | Qwen3-TTS 說話者名稱 |
| `instruct` | 生成指令 |
| `domain` / `subdomain` / `scene` | 文本領域分類 |
| `seed_text_id` | 對應文本 ID |
"""
        hf_api.upload_file(
            path_or_fileobj=dataset_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=HF_AUDIO_REPO,
            repo_type="dataset",
        )
        logger.info("Uploaded dataset card")

    os.makedirs(args.audio_dir, exist_ok=True)
    init_db(args.db_path)

    args_dict = {
        "model_id":     args.model_id,
        "src_dir":      args.src_dir,
        "audio_dir":    args.audio_dir,
        "db_path":      args.db_path,
        "upload_every": args.upload_every,
        "max_disk_gb":  args.max_disk_gb,
        "batch_size":   args.batch_size,
        "max_samples":  args.max_samples,
        "hf_token":     HF_TOKEN,
    }

    mp.set_start_method("spawn", force=True)
    processes = []
    for worker_id in range(n_workers):
        p = mp.Process(target=worker_fn, args=(worker_id, n_workers, args_dict))
        p.start()
        processes.append(p)
        logger.info("Started worker %d on GPU %d (PID %d)", worker_id, worker_id, p.pid)

    for p in processes:
        p.join()

    logger.info("All workers finished.")


if __name__ == "__main__":
    main()

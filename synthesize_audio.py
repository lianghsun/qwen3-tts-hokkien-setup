#!/usr/bin/env python3
"""
台語音頻批次生成器（Qwen3-TTS Base - voice cloning）

一鍵執行（自動偵測 GPU 數）：
    python synthesize_audio.py --src-dir ./tw-hokkien-seed-text

常用選項：
    python synthesize_audio.py --n-gpus 3 --batch-size 8
    python synthesize_audio.py --max-samples 5   # 試跑
"""

import os, sys, io, json, sqlite3, logging, argparse, csv
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


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-gpus",        type=int, default=None)
    p.add_argument("--model-id",      default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--src-dir",       default=None,
                   help="本地 seed-text 資料集目錄")
    p.add_argument("--tat-dir",       default=os.path.join(SCRIPT_DIR,
                   "tat_open_source/dev"),
                   help="TAT dev set 目錄（含 dev.tsv 與 hok/）")
    p.add_argument("--hanzi-json",    default=os.path.join(SCRIPT_DIR,
                   "conversion_results_tailo_gemini.json"))
    p.add_argument("--hf-seed-repo",  default="OKHand/Clean_Common_Voice_Speech_24.0-TW")
    p.add_argument("--hf-seed-cache", default=os.path.join(SCRIPT_DIR, "hf_seed_cache"))
    p.add_argument("--audio-dir",     default=os.path.join(SCRIPT_DIR, "audio_output"))
    p.add_argument("--db-path",       default=os.path.join(SCRIPT_DIR, "synthesis_checkpoint.db"))
    p.add_argument("--upload-every",  type=int, default=200)
    p.add_argument("--max-disk-gb",   type=float, default=20.0)
    p.add_argument("--batch-size",    type=int, default=8,
                   help="每批幾筆文字（同一批共用同一個種子說話者）")
    p.add_argument("--max-samples",   type=int, default=0,
                   help="試跑用：每個 worker 最多生幾筆（0 = 不限）")
    return p.parse_args()


# ── Seed Speakers ──────────────────────────────────────────────────────────────
def load_seed_speakers(tat_dir: str, hanzi_json: str) -> list:
    tsv_path = os.path.join(tat_dir, "dev.tsv")
    hanzi_map = {}
    if os.path.exists(hanzi_json):
        with open(hanzi_json, encoding="utf-8") as f:
            data = json.load(f)
            hanzi_map = {k: v["translated_hanzi"] for k, v in data.items()}

    seeds = []
    if not os.path.exists(tsv_path):
        logger.warning("TAT tsv not found: %s", tsv_path)
        return seeds

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            utt_id     = row["id"]
            speaker_id = row["hok_speaker"]
            wav_path   = os.path.join(tat_dir, row["hok_audio"])
            hanlo_text = row["hok_text_hanlo_tai"]
            hanzi      = hanzi_map.get(utt_id, hanlo_text)
            if os.path.exists(wav_path):
                seeds.append({
                    "speaker_id": speaker_id,
                    "utt_id":     utt_id,
                    "wav_path":   wav_path,
                    "hanzi":      hanzi,
                })

    logger.info("Loaded %d TAT seed speakers", len(seeds))
    return seeds


def load_hf_seed_speakers(repo_id: str, cache_dir: str, hf_token: str) -> list:
    from datasets import load_dataset
    import datasets as hf_datasets

    os.makedirs(cache_dir, exist_ok=True)
    done_marker = os.path.join(cache_dir, ".done")

    seeds = []
    if os.path.exists(done_marker):
        for wav_file in sorted(Path(cache_dir).glob("*.wav")):
            meta_file  = wav_file.with_suffix(".txt")
            speaker_id = wav_file.stem.split("_")[0]
            hanzi = meta_file.read_text(encoding="utf-8").strip() if meta_file.exists() else ""
            seeds.append({
                "speaker_id": speaker_id,
                "utt_id":     wav_file.stem,
                "wav_path":   str(wav_file),
                "hanzi":      hanzi,
            })
        logger.info("Loaded %d HF seed speakers from cache", len(seeds))
        return seeds

    logger.info("Downloading HF seed speakers from %s …", repo_id)
    ds = load_dataset(repo_id, split="train", token=hf_token or None)
    ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))

    for idx, item in enumerate(ds):
        audio_info = item.get("audio", {})
        sentence   = item.get("sentence", "")
        client_id  = item.get("client_id", f"spk{idx:06d}")
        safe_id    = client_id[:16].replace("/", "_").replace(" ", "_")
        utt_id     = f"{safe_id}_{idx:06d}"
        wav_path   = os.path.join(cache_dir, f"{utt_id}.wav")

        if not os.path.exists(wav_path):
            raw_bytes = audio_info.get("bytes") if isinstance(audio_info, dict) else None
            if raw_bytes:
                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                sf.write(wav_path, arr, sr)
            else:
                continue

        txt_path = os.path.join(cache_dir, f"{utt_id}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(sentence)

        seeds.append({
            "speaker_id": safe_id,
            "utt_id":     utt_id,
            "wav_path":   wav_path,
            "hanzi":      sentence,
        })
        if (idx + 1) % 1000 == 0:
            logger.info("  cached %d HF seeds", idx + 1)

    with open(done_marker, "w") as f:
        f.write(str(len(seeds)))

    logger.info("Loaded %d HF seed speakers from %s", len(seeds), repo_id)
    return seeds


# ── Database ───────────────────────────────────────────────────────────────────
def init_db(db_path: str):
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
                "audio":        {"_type": "Audio", "sampling_rate": SAMPLE_RATE},
                "text":         {"_type": "Value", "dtype": "string"},
                "duration":     {"_type": "Value", "dtype": "float32"},
                "sample_rate":  {"_type": "Value", "dtype": "int32"},
                "speaker_id":   {"_type": "Value", "dtype": "string"},
                "seed_audio_id": {"_type": "Value", "dtype": "string"},
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
        pa.field("speaker_id",   pa.string()),
        pa.field("seed_audio_id", pa.string()),
        pa.field("domain",       pa.string()),
        pa.field("subdomain",    pa.string()),
        pa.field("scene",        pa.string()),
        pa.field("emotion",      pa.string()),
        pa.field("accent",       pa.string()),
        pa.field("seed_text_id", pa.int32()),
    ], metadata={"huggingface": hf_meta})

    table = pa.table({
        "audio":        [{"bytes": s["audio_bytes"], "path": None} for s in samples],
        "text":         [s["text"]          for s in samples],
        "duration":     [s["duration"]      for s in samples],
        "sample_rate":  [s["sample_rate"]   for s in samples],
        "speaker_id":   [s["speaker_id"]    for s in samples],
        "seed_audio_id": [s["seed_audio_id"] for s in samples],
        "domain":       [s["domain"]        for s in samples],
        "subdomain":    [s["subdomain"]     for s in samples],
        "scene":        [s["scene"]         for s in samples],
        "emotion":      [s["emotion"]       for s in samples],
        "accent":       [s["accent"]        for s in samples],
        "seed_text_id": [s["seed_text_id"]  for s in samples],
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

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args_dict["model_id"],
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    log.info("Model loaded: %s", args_dict["model_id"])

    seeds   = args_dict["seed_speakers"]
    n_seeds = len(seeds)

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

    # pending: list of (text_id, target_text, item, seed)
    pending = []
    # 目前 pending 所用的 seed（同一批共用一個種子說話者以啟用 batch）
    pending_seed = None
    pending_batch_idx = 0  # 用來選說話者：每滿一批就換

    def flush_pending():
        nonlocal batch_num, local_samples, pending_seed, pending_batch_idx

        if not pending:
            return

        seed = pending_seed
        texts = [p[1] for p in pending]

        try:
            # 每批共用同一個 voice_clone_prompt → 真正的 batch inference
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=seed["wav_path"],
                x_vector_only_mode=True,  # 不需要 ref_text，支援閩南語音頻
            )
            wavs, sr = model.generate_voice_clone(
                text=texts,
                language=["Chinese"] * len(texts),
                voice_clone_prompt=prompt_items,
            )
        except Exception as e:
            log.error("Batch inference failed (seed=%s): %s", seed["utt_id"], e)
            for text_id, _, _ ,_ in pending:
                mark_error(db_path, text_id, worker_id, str(e))
            pending.clear()
            pending_batch_idx += 1
            pending_seed = None
            return

        for i, (text_id, target_text, item, _) in enumerate(pending):
            try:
                audio_np = wavs[i]
                if isinstance(audio_np, torch.Tensor):
                    audio_np = audio_np.cpu().numpy()
                audio_np = audio_np.squeeze()
                duration = round(len(audio_np) / sr, 3)

                local_samples.append({
                    "audio_bytes":   audio_to_wav_bytes(audio_np, sr),
                    "text":          target_text,
                    "duration":      duration,
                    "sample_rate":   sr,
                    "speaker_id":    seed["speaker_id"],
                    "seed_audio_id": seed["utt_id"],
                    "domain":        item.get("domain", ""),
                    "subdomain":     item.get("subdomain", ""),
                    "scene":         item.get("scene", ""),
                    "emotion":       item.get("emotion", ""),
                    "accent":        item.get("accent", ""),
                    "seed_text_id":  text_id,
                })
                done_ids.add(text_id)
                mark_done(db_path, text_id, worker_id, f"batch_{batch_num:06d}")
            except Exception as e:
                log.error("text_id=%d post-process failed: %s", text_id, e)
                mark_error(db_path, text_id, worker_id, str(e))

        pending.clear()
        pending_batch_idx += 1
        pending_seed = None

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

        # 同一批共用同一個種子說話者
        if pending_seed is None:
            seed_idx     = (worker_id * 1000 + pending_batch_idx) % n_seeds
            pending_seed = seeds[seed_idx]

        pending.append((text_id, target_text, item, pending_seed))

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

    logger.info("Starting Qwen3-TTS synthesis on %d GPU(s)", n_workers)

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

台語（閩南語）合成語音資料集，由 [Qwen3-TTS-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) voice cloning 生成。

## 資料來源

- **文本**：[lianghsun/tw-hokkien-seed-text](https://huggingface.co/datasets/lianghsun/tw-hokkien-seed-text)
- **聲色種子（TAT）**：[lianghsun/tat_open_source](https://huggingface.co/datasets/lianghsun/tat_open_source) dev/hok（722 筆）
- **聲色種子（Common Voice）**：[OKHand/Clean_Common_Voice_Speech_24.0-TW](https://huggingface.co/datasets/OKHand/Clean_Common_Voice_Speech_24.0-TW)（32,506 筆）
- **生成模型**：`Qwen/Qwen3-TTS-12Hz-1.7B-Base`（x-vector voice cloning）

## 欄位說明

| 欄位 | 說明 |
|------|------|
| `audio` | 合成音頻（WAV，12000 Hz） |
| `text` | 台語文本 |
| `duration` | 音頻長度（秒） |
| `speaker_id` | 種子說話者 ID |
| `seed_audio_id` | 種子音頻 ID |
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

    # 載入種子說話者
    seed_speakers = load_seed_speakers(args.tat_dir, args.hanzi_json)
    if args.hf_seed_repo:
        hf_seeds = load_hf_seed_speakers(args.hf_seed_repo, args.hf_seed_cache, HF_TOKEN)
        seed_speakers = seed_speakers + hf_seeds
        logger.info("Total seed speakers: %d", len(seed_speakers))

    if not seed_speakers:
        logger.error("No seed speakers found. Check --tat-dir.")
        sys.exit(1)

    args_dict = {
        "model_id":      args.model_id,
        "src_dir":       args.src_dir,
        "audio_dir":     args.audio_dir,
        "db_path":       args.db_path,
        "upload_every":  args.upload_every,
        "max_disk_gb":   args.max_disk_gb,
        "batch_size":    args.batch_size,
        "max_samples":   args.max_samples,
        "hf_token":      HF_TOKEN,
        "seed_speakers": seed_speakers,
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

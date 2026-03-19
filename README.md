# qwen3-tts-hokkien-setup

使用 [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) 批次合成台語（閩南語）音頻資料集。

合成結果上傳至：[lianghsun/tw-hokkien-audio-qwen3](https://huggingface.co/datasets/lianghsun/tw-hokkien-audio-qwen3)

---

## 與 CosyVoice3 版本的差異

| | CosyVoice3 | Qwen3-TTS |
|---|---|---|
| 說話者 | TAT + Common Voice 種子音檔 | 9 個預設說話者循環 |
| 輸出取樣率 | 22050 Hz | 12000 Hz |
| Batch 支援 | 無（每次 1 筆） | 原生支援 |
| vLLM 支援 | 無 | vLLM-Omni（獨立專案） |

> **注意**：Qwen3-TTS 使用預設說話者，不接受外部 reference audio。
> 若需聲色克隆，請改用 `Qwen3-TTS-12Hz-1.7B-Base` 搭配 vLLM-Omni ICL 模式（尚未實作）。

---

## 環境需求

- Linux（建議 NVIDIA B200 / H100 / A100）
- Python 3.12
- CUDA 12.x

---

## 安裝

```bash
git clone https://github.com/lianghsun/qwen3-tts-hokkien-setup
cd qwen3-tts-hokkien-setup
bash setup.sh
source .venv/bin/activate
```

---

## 執行合成

```bash
# clone 文本資料集
git clone https://huggingface.co/datasets/lianghsun/tw-hokkien-seed-text

# 設定 HuggingFace token
export HF_TOKEN="hf_..."

# 開始合成（自動偵測 GPU 數）
python synthesize_audio.py --src-dir ./tw-hokkien-seed-text
```

### 常用選項

| 參數 | 預設 | 說明 |
|------|------|------|
| `--n-gpus` | 自動偵測 | 使用幾張 GPU |
| `--batch-size` | `8` | 每次 inference 批次大小 |
| `--upload-every` | `200` | 每幾筆上傳一次 HuggingFace |
| `--max-samples` | `0` | 試跑用：每個 worker 最多生幾筆（0 = 不限）|
| `--model-id` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 模型 ID |

### 試跑 5 筆

```bash
python synthesize_audio.py --src-dir ./tw-hokkien-seed-text \
    --n-gpus 1 --max-samples 5 --upload-every 5
```

### 正式跑（B200 × 3）

```bash
python synthesize_audio.py --src-dir ./tw-hokkien-seed-text \
    --n-gpus 3 --batch-size 16 --upload-every 200
```

---

## 斷點續跑

進度存於 `synthesis_checkpoint.db`，中斷後直接重跑即可續接。

```bash
rm synthesis_checkpoint.db  # 清除進度重來
```

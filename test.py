from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
import ffmpeg
import soundfile as sf
import io
import numpy as np
import os

# --- Whisper モデル設定 ---
MODEL_ID     = "kotoba-tech/kotoba-whisper-v2.2"
TARGET_SR    = 16000            # 推奨サンプリングレート
CHUNK_SEC    = 30               # チャンク長（秒）
OVERLAP_SEC  = 1                # 重複させる秒数
BATCH_SIZE   = 4                # 4060Ti 向け

# --- モデルとプロセッサの読み込み ---
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# サンプリングレート固定（必要）
processor.feature_extractor.sampling_rate = TARGET_SR

# --- GPU 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Whisper Pipeline 構築 ---
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device.index if device.type == "cuda" else -1,
    return_timestamps=False,
    generate_kwargs={
        "forced_decoder_ids": forced_decoder_ids,
        "num_beams": 10,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "temperature": 0.0,
    }
)

# --- 書き起こし開始 ---
output_path = "./gairon/transcriptions.txt"
with open(output_path, "w", encoding="utf-8") as fout:
    for i in range(0,1):  # media1.mp4 を対象（必要に応じて拡張）
        fn = f"./gairon/media{i+1}.mp4"
        if not os.path.exists(fn):
            print(f"[!] ファイルが見つかりません: {fn}")
            continue

        print(f"[+] 処理開始: {fn}")

        # --- ffmpeg で音声抽出 ---
        try:
            out, _ = (
                ffmpeg
                .input(fn)
                .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=TARGET_SR)
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"[ffmpeg エラー]:\n{e.stderr.decode()}")
            continue

        audio, sr = sf.read(io.BytesIO(out))
        assert sr == TARGET_SR, f"{fn} のサンプリングレートが {TARGET_SR} ではありません。"

        # --- チャンク分割（オーバーラップあり） ---
        chunk_len    = TARGET_SR * CHUNK_SEC
        overlap_len  = TARGET_SR * OVERLAP_SEC
        start        = 0
        chunks       = []

        while start < len(audio):
            end = min(start + chunk_len, len(audio))
            chunk = audio[start:end]
            if chunk.size > 0:
                chunks.append(chunk)
            start += chunk_len - overlap_len

        print(f"[+] チャンク数: {len(chunks)} - バッチ処理中（batch_size={BATCH_SIZE}）...")
        results = asr(chunks, batch_size=BATCH_SIZE)

        # --- 重複削除付きマージ処理 ---
        merged_texts = []
        prev_text = ""
        for res in results:
            text = res["text"].strip()
            overlap_len = min(len(prev_text), len(text))
            for i in range(overlap_len, 0, -1):
                if prev_text.endswith(text[:i]):
                    text = text[i:]
                    break
            merged_texts.append(text)
            prev_text += text

        fout.write("\n".join(merged_texts) + "\n\n")
        print(f"[✓] 書き起こし完了: {fn}")

print(f"[✓] すべての書き起こしを {output_path} に保存しました。")

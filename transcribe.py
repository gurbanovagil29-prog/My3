#!/usr/bin/env python3
"""Нейродиктофон: аудио -> текст (faster-whisper) -> структурированный документ (Claude, опционально).

Примеры:
    python transcribe.py запись.m4a
    python transcribe.py запись.m4a --structure tz
    python transcribe.py запись.m4a --model medium --out результаты/
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
PAUSE_FOR_PARAGRAPH = 2.0  # сек тишины, после которой начинается новый абзац
MAX_PARAGRAPH_CHARS = 700


def fmt_time(seconds, srt=False):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if srt:
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


def load_audio(src):
    """Декодирует любой формат через ffmpeg в 16 кГц моно float32 — то, что ест Whisper."""
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(src), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def transcribe(audio, model_name, language, device, threads):
    from faster_whisper import WhisperModel

    compute_type = "int8" if device == "cpu" else "float16"
    print(f"Загрузка модели {model_name} ({device}, {compute_type})...", file=sys.stderr)
    model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=threads)

    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # меньше «зацикливаний» на длинных записях
        initial_prompt="Расшифровка аудиозаписи на русском языке, с пунктуацией.",
    )
    duration = info.duration
    print(f"Длительность: {fmt_time(duration)}, язык: {info.language}", file=sys.stderr)

    result, started = [], time.time()
    for seg in segments:
        text = seg.text.strip()
        if text:
            result.append((seg.start, seg.end, text))
        elapsed = time.time() - started
        pct = seg.end / duration * 100 if duration else 0
        eta = elapsed / seg.end * (duration - seg.end) if seg.end else 0
        print(f"\r  {pct:5.1f}%  [{fmt_time(seg.end)} / {fmt_time(duration)}]  осталось ~{fmt_time(eta)}",
              end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return result


def to_paragraphs(segments):
    paragraphs, current, prev_end = [], [], None
    for start, end, text in segments:
        long_pause = prev_end is not None and start - prev_end > PAUSE_FOR_PARAGRAPH
        too_long = sum(len(t) for _, t in current) > MAX_PARAGRAPH_CHARS
        if current and (long_pause or (too_long and current[-1][1].endswith((".", "!", "?")))):
            paragraphs.append(current)
            current = []
        current.append((start, text))
        prev_end = end
    if current:
        paragraphs.append(current)
    return paragraphs


def write_outputs(segments, out_dir, stem):
    paragraphs = to_paragraphs(segments)
    plain = "\n\n".join(" ".join(t for _, t in p) for p in paragraphs)
    timed = "\n\n".join(f"[{fmt_time(p[0][0])}] " + " ".join(t for _, t in p) for p in paragraphs)
    srt = "\n".join(
        f"{i}\n{fmt_time(s, True)} --> {fmt_time(e, True)}\n{t}\n" for i, (s, e, t) in enumerate(segments, 1)
    )
    files = {
        out_dir / f"{stem}.txt": plain + "\n",
        out_dir / f"{stem}.timecodes.txt": timed + "\n",
        out_dir / f"{stem}.srt": srt,
    }
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
        print(f"  -> {path}", file=sys.stderr)
    return plain


def structure(transcript, template, model, out_dir, stem):
    prompt = (PROMPTS_DIR / f"{template}.md").read_text(encoding="utf-8").replace("{transcript}", transcript)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        path = out_dir / f"{stem}.prompt-{template}.md"
        path.write_text(prompt, encoding="utf-8")
        print(f"ANTHROPIC_API_KEY не задан — готовый промпт сохранён, вставьте его в любой ИИ:\n  -> {path}",
              file=sys.stderr)
        return

    import anthropic

    print(f"Структурирование ({template}) через {model}...", file=sys.stderr)
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=model, max_tokens=16000, messages=[{"role": "user", "content": prompt}]
    ) as stream:
        message = stream.get_final_message()
    text = "".join(b.text for b in message.content if b.type == "text")
    path = out_dir / f"{stem}.{template}.md"
    path.write_text(text, encoding="utf-8")
    print(f"  -> {path}", file=sys.stderr)


def main():
    templates = sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
    ap = argparse.ArgumentParser(description="Аудио -> текст -> структурированный документ")
    ap.add_argument("audio", type=Path, help="файл записи (m4a, mp3, wav, ogg, ...)")
    ap.add_argument("--out", type=Path, default=Path("output"), help="папка для результатов (по умолчанию output/)")
    ap.add_argument("--model", default="large-v3-turbo",
                    help="модель Whisper: large-v3-turbo (по умолчанию), large-v3, medium, small")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--structure", choices=templates + ["none"], default="none",
                    help="шаблон структурирования из prompts/ (нужен ANTHROPIC_API_KEY, иначе сохранится промпт)")
    ap.add_argument("--llm-model", default="claude-sonnet-5-5")
    args = ap.parse_args()

    if not args.audio.exists():
        sys.exit(f"Файл не найден: {args.audio}")

    device = args.device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.audio.stem

    print("Конвертация аудио...", file=sys.stderr)
    audio = load_audio(args.audio)
    started = time.time()
    segments = transcribe(audio, args.model, args.language, device, args.threads)
    print(f"Распознано за {fmt_time(time.time() - started)}", file=sys.stderr)

    transcript = write_outputs(segments, args.out, stem)

    if args.structure != "none":
        structure(transcript, args.structure, args.llm_model, args.out, stem)


if __name__ == "__main__":
    main()

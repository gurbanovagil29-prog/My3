#!/usr/bin/env python3
"""Нейродиктофон: аудио -> текст (faster-whisper) + спикеры -> структурированный документ (Claude, опционально).

Примеры:
    python transcribe.py запись.m4a
    python transcribe.py запись.m4a --structure tz
    python transcribe.py запись.m4a --structure none   # только текст, без промпта
    python transcribe.py запись.m4a --speakers 3      # если точно знаете число участников
    python transcribe.py запись.m4a --model medium --out результаты/
    python transcribe.py часть1.m4a часть2.m4a        # несколько частей одной встречи -> один документ
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROMPTS_DIR = Path(__file__).parent / "prompts"
PAUSE_FOR_PARAGRAPH = 2.0  # сек тишины, после которой начинается новый абзац
MAX_PARAGRAPH_CHARS = 700
# Фразы, которые Whisper «додумывает» на тишине и шуме (выучены на субтитрах к видео)
HALLUCINATIONS = re.compile(
    r"^(продолжение следует|субтитры (сделал|создавал|подготовил)|редактор субтитров|корректор|"
    r"спасибо за просмотр|подписывайтесь на канал|ставьте лайки)",
    re.IGNORECASE,
)


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
        word_timestamps=True,  # нужно, чтобы резать фразы по предложениям при разметке спикеров
    )
    duration = info.duration
    print(f"Длительность: {fmt_time(duration)}, язык: {info.language}", file=sys.stderr)

    result, started = [], time.time()
    for seg in segments:
        text = seg.text.strip()
        if text and not HALLUCINATIONS.match(text):
            result.append(seg)
        elapsed = time.time() - started
        pct = seg.end / duration * 100 if duration else 0
        eta = elapsed / seg.end * (duration - seg.end) if seg.end else 0
        print(f"\r  {pct:5.1f}%  [{fmt_time(seg.end)} / {fmt_time(duration)}]  осталось ~{fmt_time(eta)}",
              end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return result


def speaker_prefix(speaker):
    return f"Спикер {speaker}: " if speaker else ""


def transcribe_cached(path, audio, args, device):
    """Распознаёт файл, сохраняя результат в output/.cache — повторный запуск (например, с новой частью) не пересчитывает."""
    cache = args.out / ".cache" / f"{path.stem}.{args.model}.{args.language}.json"
    if cache.exists():
        print(f"{path.name}: распознавание взято из кэша", file=sys.stderr)
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        started = time.time()
        segments = transcribe(audio, args.model, args.language, device, args.threads)
        print(f"Распознано за {fmt_time(time.time() - started)}", file=sys.stderr)
        data = [
            {"start": seg.start, "end": seg.end, "text": seg.text,
             "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (seg.words or [])]}
            for seg in segments
        ]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return [
        SimpleNamespace(start=d["start"], end=d["end"], text=d["text"],
                        words=[SimpleNamespace(**w) for w in d["words"]])
        for d in data
    ]


def shift(segments, offset):
    """Сдвигает таймкоды части, чтобы они продолжали предыдущую."""
    for seg in segments:
        seg.start += offset
        seg.end += offset
        for w in seg.words:
            w.start += offset
            w.end += offset
    return segments


def to_paragraphs(utterances):
    """Абзац заканчивается при смене спикера, долгой паузе или когда он стал слишком длинным."""
    paragraphs, current, prev_end = [], [], None
    for start, end, text, speaker in utterances:
        new_speaker = current and speaker != current[0][2]
        long_pause = prev_end is not None and start - prev_end > PAUSE_FOR_PARAGRAPH
        too_long = sum(len(t) for _, t, _ in current) > MAX_PARAGRAPH_CHARS
        if current and (new_speaker or long_pause or (too_long and current[-1][1].endswith((".", "!", "?")))):
            paragraphs.append(current)
            current = []
        current.append((start, text, speaker))
        prev_end = end
    if current:
        paragraphs.append(current)
    return paragraphs


def write_outputs(utterances, out_dir, stem):
    paragraphs = to_paragraphs(utterances)
    body = [speaker_prefix(p[0][2]) + " ".join(t for _, t, _ in p) for p in paragraphs]
    plain = "\n\n".join(body)
    timed = "\n\n".join(f"[{fmt_time(p[0][0])}] {b}" for p, b in zip(paragraphs, body))
    srt = "\n".join(
        f"{i}\n{fmt_time(s, True)} --> {fmt_time(e, True)}\n{f'[Спикер {spk}] ' if spk else ''}{t}\n"
        for i, (s, e, t, spk) in enumerate(utterances, 1)
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


def structure(transcript, utterances, template, model, out_dir, stem):
    # {transcript} — сплошной текст, {transcript_timed} — каждая фраза со своим таймкодом и спикером
    timed = "\n".join(f"[{fmt_time(s)}] {speaker_prefix(spk)}{t}" for s, _, t, spk in utterances)
    prompt = (
        (PROMPTS_DIR / f"{template}.md").read_text(encoding="utf-8")
        .replace("{transcript_timed}", timed)
        .replace("{transcript}", transcript)
    )

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
    ap.add_argument("audio", type=Path, nargs="+",
                    help="файл записи (m4a, mp3, wav, ogg, ...); несколько частей одной встречи — по порядку")
    ap.add_argument("--name", help="имя результатов (по умолчанию — имя первого файла)")
    ap.add_argument("--out", type=Path, default=Path("output"), help="папка для результатов (по умолчанию output/)")
    ap.add_argument("--model", default="large-v3-turbo",
                    help="модель Whisper: large-v3-turbo (по умолчанию), large-v3, medium, small")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--structure", choices=templates + ["none"], default="qa",
                    help="шаблон из prompts/, по умолчанию qa — вопрос/ответ с таймкодами (нужен ANTHROPIC_API_KEY, иначе сохранится промпт)")
    ap.add_argument("--speakers", type=int, default=None,
                    help="число участников, если известно (иначе определяется автоматически)")
    ap.add_argument("--no-speakers", action="store_true", help="не размечать спикеров")
    ap.add_argument("--llm-model", default="claude-sonnet-5-5")
    args = ap.parse_args()

    for path in args.audio:
        if not path.exists():
            sys.exit(f"Файл не найден: {path}")

    device = args.device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.name or args.audio[0].stem

    import numpy as np

    parts, segments, offset = [], [], 0.0
    for i, path in enumerate(args.audio, 1):
        print(f"[{i}/{len(args.audio)}] {path.name}: конвертация аудио...", file=sys.stderr)
        audio = load_audio(path)
        segments += shift(transcribe_cached(path, audio, args, device), offset)
        if len(args.audio) > 1:
            print(f"  часть {i} начинается с {fmt_time(offset)}", file=sys.stderr)
        parts.append(audio)
        offset += len(audio) / 16000
    audio = np.concatenate(parts)  # спикеры размечаются по всем частям сразу — номера сквозные

    if args.no_speakers:
        utterances = [(seg.start, seg.end, seg.text.strip(), None) for seg in segments]
    else:
        from diarize import diarize
        utterances = diarize(audio, segments, args.threads, args.speakers)

    transcript = write_outputs(utterances, args.out, stem)

    if args.structure != "none":
        structure(transcript, utterances, args.structure, args.llm_model, args.out, stem)


if __name__ == "__main__":
    main()

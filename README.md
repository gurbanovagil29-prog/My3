# Нейродиктофон

Превращает запись с диктофона (30–40+ минут) в текст, а при желании — в структурированное ТЗ или конспект.

```
аудио → ffmpeg (16 кГц моно) → faster-whisper (распознавание) → .txt / .timecodes.txt / .srt
                                                               └→ Claude по шаблону → .tz.md / .summary.md
```

Распознавание работает **локально и бесплатно** (модель Whisper large-v3-turbo), запись никуда не отправляется.

## Установка

Нужны Python 3.9+ и ffmpeg.

```bash
# ffmpeg: macOS — brew install ffmpeg; Ubuntu — sudo apt install ffmpeg; Windows — winget install ffmpeg
pip install -r requirements.txt
```

При первом запуске модель (~1.6 ГБ) скачается автоматически.

## Использование

```bash
# Только текст
python transcribe.py запись.m4a

# Текст + ТЗ (нужен ключ Claude API)
export ANTHROPIC_API_KEY=sk-ant-...
python transcribe.py запись.m4a --structure tz

# Текст + конспект встречи
python transcribe.py запись.m4a --structure summary
```

Без `ANTHROPIC_API_KEY` флаг `--structure` сохранит **готовый промпт** (`*.prompt-tz.md`) — его можно целиком вставить в любой чат с ИИ.

### Что получится в `output/`

| Файл | Что внутри |
|---|---|
| `запись.txt` | Чистый текст, разбитый на абзацы по паузам — для копирования в ИИ |
| `запись.timecodes.txt` | То же с таймкодами `[12:34]` у каждого абзаца |
| `запись.srt` | Субтитры |
| `запись.tz.md` / `запись.summary.md` | Структурированный документ (с `--structure`) |

### Опции

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--model` | `large-v3-turbo` | `large-v3` — чуть точнее и в ~2 раза медленнее; `medium`/`small` — быстрее, но хуже |
| `--device` | `auto` | `cuda` для видеокарты NVIDIA, иначе `cpu` |
| `--language` | `ru` | Язык записи |
| `--out` | `output` | Папка результатов |
| `--structure` | `none` | Шаблон из `prompts/`: `tz`, `summary` |

## Скорость

- Видеокарта NVIDIA: 40 минут записи ≈ 2–4 минуты.
- Процессор (4 ядра): примерно 0.5–1× длительности записи.

## Свои шаблоны

Положите файл `prompts/<имя>.md` с плейсхолдером `{transcript}` — он сразу станет доступен как `--structure <имя>`.

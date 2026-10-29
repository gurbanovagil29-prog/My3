"""Разметка спикеров: голосовой «отпечаток» каждой реплики (TitaNet через sherpa-onnx) + кластеризация."""
import sys
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/nemo_en_titanet_small.onnx")
MODEL_PATH = Path.home() / ".cache" / "neuro-dictaphone" / "nemo_en_titanet_small.onnx"
SAMPLE_RATE = 16000
MIN_CLUSTER_UTTERANCE = 1.5  # сек: по более коротким репликам голос определяется ненадёжно
MIN_SPEAKER_SHARE = 0.02     # доля записи, меньше которой кластер считается шумом, а не спикером
MIN_SPEAKER_SECONDS = 10


def ensure_model():
    if not MODEL_PATH.exists():
        print("Скачивание модели распознавания голосов (~40 МБ)...", file=sys.stderr)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(MODEL_PATH)
    return MODEL_PATH


def split_sentences(segments):
    """Режет сегменты Whisper на предложения по словам: так вопрос и ответ не склеиваются в одну реплику."""
    utterances = []
    for seg in segments:
        words = seg.words or []
        if not words:
            utterances.append([seg.start, seg.end, seg.text.strip()])
            continue
        current = []
        for w in words:
            current.append(w)
            if w.word.strip().endswith((".", "?", "!", "…")):
                utterances.append([current[0].start, current[-1].end, "".join(x.word for x in current).strip()])
                current = []
        if current:
            utterances.append([current[0].start, current[-1].end, "".join(x.word for x in current).strip()])
    return [u for u in utterances if u[2]]


def embed(audio, utterances, threads):
    import sherpa_onnx

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(ensure_model()), num_threads=threads)
    )
    vectors = []
    for start, end, _ in utterances:
        chunk = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE // 4:  # слишком коротко для модели — расширяем окно до 0.5 с
            mid = int((start + end) / 2 * SAMPLE_RATE)
            chunk = audio[max(0, mid - SAMPLE_RATE // 4):mid + SAMPLE_RATE // 4]
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, chunk)
        stream.input_finished()
        v = np.array(extractor.compute(stream))
        vectors.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(vectors)


def average_linkage(vectors, threshold, num_speakers=None):
    """Агломеративная кластеризация по косинусному сходству (average linkage, обновление Ланса–Уильямса)."""
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -np.inf)
    clusters = [[i] for i in range(len(vectors))]
    alive = np.ones(len(vectors), dtype=bool)
    while alive.sum() > 1:
        masked = np.where(alive[:, None] & alive[None, :], sim, -np.inf)
        i, j = np.unravel_index(masked.argmax(), masked.shape)
        if num_speakers is None and masked[i, j] < threshold:
            break
        if num_speakers is not None and alive.sum() <= num_speakers:
            break
        ni, nj = len(clusters[i]), len(clusters[j])
        sim[i, :] = sim[:, i] = (ni * sim[i, :] + nj * sim[j, :]) / (ni + nj)
        sim[i, i] = -np.inf
        clusters[i] += clusters[j]
        alive[j] = False
    return [clusters[k] for k in np.flatnonzero(alive)]


def diarize(audio, segments, threads=4, num_speakers=None, threshold=0.5):
    """Возвращает реплики [(start, end, text, speaker_number)], спикеры пронумерованы по первому появлению."""
    print("Разметка спикеров...", file=sys.stderr)
    utterances = split_sentences(segments)
    vectors = embed(audio, utterances, threads)
    durations = np.array([e - s for s, e, _ in utterances])

    reliable = np.flatnonzero(durations >= MIN_CLUSTER_UTTERANCE)
    if len(reliable) < 2:
        return [(s, e, t, 1) for s, e, t in utterances]
    clusters = [reliable[c] for c in average_linkage(vectors[reliable], threshold, num_speakers)]

    total = durations.sum()
    min_seconds = max(MIN_SPEAKER_SECONDS, MIN_SPEAKER_SHARE * total)
    speakers = [c for c in clusters if durations[c].sum() >= min_seconds] or [max(clusters, key=len)]
    centroids = np.array([vectors[c].mean(axis=0) for c in speakers])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    # каждую реплику (включая короткие и «шумовые» кластеры) относим к ближайшему настоящему спикеру
    labels = (vectors @ centroids.T).argmax(axis=1)

    order = {}
    for lab in labels:
        order.setdefault(lab, len(order) + 1)
    print(f"  найдено спикеров: {len(order)}", file=sys.stderr)
    return [(s, e, t, order[lab]) for (s, e, t), lab in zip(utterances, labels)]

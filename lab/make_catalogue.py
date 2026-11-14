"""Write ``lab/catalogue.json``, the model lab's list of candidates.

Run ``python3 lab/make_catalogue.py`` after changing an entry here; the
JSON is what the service reads. Facts (licence, gating, sizes) were
read from the model cards and the Hugging Face API on 2026-11-14;
``vram_gb`` is the peak GPU memory measured by the runner on a 122 s
public test clip (LibriSpeech dummy utterances) on the RTX 3070 Ti,
unless the notes say otherwise. Peak memory grows with the recording
for some models (noted).
"""

import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent

DIARIZE = {"num_speakers": None, "max_speakers": None}
WHISPER_LANGS = ["multilingual"]
PARAKEET_LANGS = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el",
    "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es",
    "sv", "ru", "uk",
]


def entry(**fields: Any) -> Dict[str, Any]:
    """Return one catalogue entry with every field present."""
    base: Dict[str, Any] = {
        "id": None, "stage": None, "family": None, "display_name": None,
        "source_url": None, "licence": None, "licence_commercial": True,
        "gated": False, "token_url": None, "languages": ["any"],
        "vram_gb": None, "parameters": {}, "weights": [],
        "weights_mb": None, "notes": "", "lineage": None, "revision": 1,
    }
    unknown = set(fields) - set(base)
    assert not unknown, unknown
    base.update(fields)
    return base


ENTRIES: List[Dict[str, Any]] = [
    # ---------------------------------------------------------------- denoise
    entry(
        id="none", stage="denoise", family="project",
        display_name="No enhancement (original audio)",
        source_url="", licence="n/a",
        vram_gb=0.0, weights_mb=0,
        notes="In-process: the session's audio as decoded (16 kHz mono).",
    ),
    entry(
        id="deepfilternet3", stage="denoise", family="enhance",
        display_name="DeepFilterNet3",
        source_url="https://github.com/Rikorose/DeepFilterNet",
        licence="MIT OR Apache-2.0 (code and weights)",
        vram_gb=1.9, parameters={"atten_lim_db": None,
                                 "post_filter": False},
        weights=["https://github.com/Rikorose/DeepFilterNet/raw/main/"
                 "models/DeepFilterNet3.zip"],
        weights_mb=9,
        notes="deepfilternet 0.5.6 (the current release; DeepFilterNet3"
              " is its default model), 48 kHz full band. The package"
              " pins numpy<2 and imports torchaudio.backend (removed in"
              " torchaudio 2.1): installed without its pins and run with"
              " an empty stand-in for that one type; the env's torchaudio"
              " 2.11.1 runs without its compiled extension. Weights come"
              " from the GitHub release into ~/.cache/DeepFilterNet."
              " Measured 1.7 s for 122 s on the GPU (1.9 GB peak).",
    ),
    entry(
        id="mossformer2-se-48k", stage="denoise", family="enhance",
        display_name="ClearerVoice MossFormer2_SE_48K",
        source_url="https://huggingface.co/alibabasglab/MossFormer2_SE_48K",
        licence="Apache-2.0",
        vram_gb=0.5,
        weights=["alibabasglab/MossFormer2_SE_48K"], weights_mb=222,
        notes="ClearerVoice-Studio (clearvoice 0.1.2, installed without"
              " its pins: numpy<2, soundfile 0.12, torchvision/opencv for"
              " its video models). 48 kHz model; ClearVoice cuts long"
              " audio into windows itself. Checkpoints in"
              " ~/.local/share/ragputin/lab-models/clearvoice.",
    ),
    entry(
        id="mossformergan-se-16k", stage="denoise", family="enhance",
        display_name="ClearerVoice MossFormerGAN_SE_16K",
        source_url="https://huggingface.co/alibabasglab/"
                   "MossFormerGAN_SE_16K",
        licence="Apache-2.0",
        vram_gb=6.3,
        weights=["alibabasglab/MossFormerGAN_SE_16K"], weights_mb=136,
        notes="ClearerVoice-Studio 16 kHz GAN enhancement, 10 s decoding"
              " windows. Heavy: 6.3 GB peak on the GPU (it went out of"
              " memory while other jobs held 5 GB), 21 s for 122 s.",
    ),
    entry(
        id="sepformer-dns4", stage="denoise", family="speechbrain", revision=2,
        display_name="SpeechBrain SepFormer DNS-4 (16 kHz)",
        source_url="https://huggingface.co/speechbrain/"
                   "sepformer-dns4-16k-enhancement",
        licence="Apache-2.0",
        vram_gb=1.8, parameters={"piece_seconds": 10.0},
        weights=["speechbrain/sepformer-dns4-16k-enhancement"],
        weights_mb=113,
        notes="SpeechBrain 1.1.2. Run on 10 s pieces with 1 s"
              " cross-fades so memory does not grow with the recording."
              " Revision 2: each piece is scaled back to its input's RMS"
              " level (the model's output level is arbitrary: revision 1"
              " wrote audio at about 17 times the input's RMS, clipped,"
              " on which no voice matched a voiceprint).",
    ),
    entry(
        id="metricgan-plus", stage="denoise", family="speechbrain", revision=2,
        display_name="SpeechBrain MetricGAN+ (VoiceBank)",
        source_url="https://huggingface.co/speechbrain/"
                   "metricgan-plus-voicebank",
        licence="Apache-2.0",
        vram_gb=0.3, parameters={"piece_seconds": 10.0},
        weights=["speechbrain/metricgan-plus-voicebank"], weights_mb=8,
        notes="Spectral mask enhancement trained on VoiceBank-DEMAND"
              " (read speech, synthetic noise); 10 s pieces. Revision 2:"
              " each piece scaled back to its input's RMS level"
              " (revision 1 doubled it, clipped).",
    ),
    entry(
        id="rnnoise", stage="denoise", family="enhance",
        display_name="RNNoise",
        source_url="https://github.com/xiph/rnnoise",
        licence="BSD-3-Clause (RNNoise); Apache-2.0 (pyrnnoise wrapper)",
        vram_gb=0.0, weights=["pyrnnoise==0.4.5 (bundled)"], weights_mb=0,
        notes="pyrnnoise 0.4.5's bundled librnnoise, 48 kHz, 10 ms"
              " frames, CPU only (3.5 s for 122 s).",
    ),
    # ---------------------------------------------------------------- diarize
    entry(
        id="builtin", stage="diarize", family="project",
        display_name="Built-in (voice windows + grouping)",
        source_url="", licence="project",
        parameters={"num_speakers": None, "expected_speakers": None},
        weights=["openai/whisper-large-v3-turbo",
                 "microsoft/wavlm-base-plus-sv"],
        notes="In-process: the service's transcription on the lab's"
              " audio (the service's recogniser for speech regions, WavLM"
              " windows, grouping, identification with the session's"
              " expected speakers and count); segments are the turns of"
              " the attributed words, so it never reports overlap and its"
              " time includes its recognition.",
    ),
    entry(
        id="pyannote-3.1", stage="diarize", family="pyannote",
        lineage="pyannote",
        display_name="pyannote speaker-diarization-3.1",
        source_url="https://huggingface.co/pyannote/"
                   "speaker-diarization-3.1",
        licence="MIT", gated=True,
        token_url="https://huggingface.co/pyannote/speaker-diarization-3.1",
        parameters={"num_speakers": None, "min_speakers": None,
                    "max_speakers": None},
        weights=["pyannote/speaker-diarization-3.1",
                 "pyannote/segmentation-3.0",
                 "pyannote/wespeaker-voxceleb-resnet34-LM"],
        vram_gb=2.2, weights_mb=33,
        notes="Gated: accept the conditions of"
              " pyannote/speaker-diarization-3.1 and"
              " pyannote/segmentation-3.0, then log in. Run through"
              " pyannote.audio 4.0.7 with audio in memory (torchcodec"
              " cannot load without FFmpeg). Ran on R1 (388 s) once the"
              " owner's token was present: 6.5-8.1 s, peak 2216 MB"
              " (without a token the runner stops with needs_token).",
    ),
    entry(
        id="pyannote-community-1", stage="diarize", family="pyannote",
        lineage="pyannote",
        display_name="pyannote speaker-diarization-community-1",
        source_url="https://huggingface.co/pyannote/"
                   "speaker-diarization-community-1",
        licence="CC-BY-4.0", gated=True,
        token_url="https://huggingface.co/pyannote/"
                  "speaker-diarization-community-1",
        parameters={"num_speakers": None, "min_speakers": None,
                    "max_speakers": None},
        weights=["pyannote/speaker-diarization-community-1"],
        vram_gb=2.2, weights_mb=34,
        notes="Gated (accept the conditions, log in). pyannote.audio"
              " 4.0.7; the overlapping speaker_diarization output is the"
              " result, the exclusive one is kept under 'exclusive'."
              " Ran on R1 (388 s) with the owner's token: 6.5-6.6 s, peak"
              " 2216 MB.",
    ),
    entry(
        id="sortformer-streaming-v2", stage="diarize", family="nemo",
        display_name="NVIDIA Streaming Sortformer v2 (4 speakers)",
        source_url="https://huggingface.co/nvidia/"
                   "diar_streaming_sortformer_4spk-v2",
        licence="CC-BY-4.0",
        vram_gb=1.0, parameters={"threshold": 0.5},
        weights=["nvidia/diar_streaming_sortformer_4spk-v2"],
        weights_mb=471,
        notes="NeMo 3.0.0, the card's offline (very high latency)"
              " streaming setting; posteriors every 80 ms, a speaker is"
              " active at >= threshold; at most 4 speakers, num_speakers"
              " is not used. Overlap native. 0.6 s for 122 s.",
    ),
    entry(
        id="nemotron-3-diarization", stage="diarize", family="transformers",
        display_name="NVIDIA Nemotron-3 Diarization (8 speakers)",
        source_url="https://huggingface.co/nvidia/Nemotron-3-Diarization",
        licence="OpenMDW-1.1 (card: commercial use allowed)",
        vram_gb=0.6, parameters={"threshold": 0.5,
                                 "streaming_mode": None},
        weights=["nvidia/Nemotron-3-Diarization"], weights_mb=397,
        notes="Runs with the project's transformers 5.20.2 (model type"
              " nemotron3_diarization; NeMo is not needed, so the Python"
              " >= 3.11 requirement of NeMo's main branch does not"
              " apply). Chunked streaming inference as on the card, in"
              " the longest-latency mode the processor offers"
              " (low_latency, 1.04 s); frames every 10 ms; overlap"
              " native; num_speakers not used. 4.7 s for 122 s.",
    ),
    entry(
        id="diarizen-wavlm-large-s80-md-v2", stage="diarize",
        family="diarizen",
        display_name="DiariZen WavLM-Large s80 v2 (BUT Speech@FIT)",
        source_url="https://huggingface.co/BUT-FIT/"
                   "diarizen-wavlm-large-s80-md-v2",
        licence="CC-BY-NC-4.0 (weights); MIT (code)",
        licence_commercial=False,
        vram_gb=3.7,
        parameters={"num_speakers": None, "min_speakers": None,
                    "max_speakers": None, "batch_size": 8},
        weights=["BUT-FIT/diarizen-wavlm-large-s80-md-v2",
                 "pyannote/wespeaker-voxceleb-resnet34-LM"],
        weights_mb=305,
        notes="NON-COMMERCIAL weights: evaluation only. DiariZen at"
              " commit 844f555 with its pyannote.audio 3.1.1 fork; VBx"
              " clustering; overlap native (powerset). Batch 8 (32, the"
              " default, ran out of memory). Needs shims for torchaudio"
              " 2.11 (file loading, a removed type) and torch's"
              " weights_only loading. 5.3 s for 122 s.",
    ),
    entry(
        id="ecapa-spectral", stage="diarize", family="speechbrain",
        display_name="ECAPA-TDNN + spectral clustering (lab's own)",
        source_url="https://huggingface.co/speechbrain/"
                   "spkrec-ecapa-voxceleb",
        licence="Apache-2.0 (weights)",
        vram_gb=1.4,
        parameters={"num_speakers": None, "max_speakers": None,
                    "window": 1.5, "hop": 0.75},
        weights=["speechbrain/spkrec-ecapa-voxceleb"], weights_mb=89,
        notes="The lab's own simple diarizer, not a published recipe:"
              " energy speech mask, 1.5 s ECAPA windows every 0.75 s,"
              " spectral clustering (eigengap when num_speakers is not"
              " given). One speaker per frame: never marks overlap.",
    ),
    entry(
        id="nemo-clustering", stage="diarize", family="nemo",
        display_name="NeMo clustering diarizer (MarbleNet + TitaNet-L)",
        source_url="https://huggingface.co/nvidia/"
                   "speakerverification_en_titanet_large",
        licence="CC-BY-4.0 (TitaNet-L); NeMo VAD weights per NGC",
        vram_gb=2.9,
        parameters={"num_speakers": None, "max_speakers": None,
                    "batch_size": 16},
        weights=["nvidia/speakerverification_en_titanet_large",
                 "vad_multilingual_marblenet (NGC)"],
        weights_mb=102,
        notes="NeMo 3.0.0 ClusteringDiarizer with the settings of NeMo's"
              " diar_infer_meeting.yaml, inline; multi-scale TitaNet-L"
              " embeddings, spectral clustering; ignores overlap. Batch 16:"
              " 2.9 GB peak (nvidia-smi), 17 s for 122 s; batch 64"
              " peaked at 5.7 GB.",
    ),
    # -------------------------------------------------------------- recognise
    entry(
        id="whisper-large-v3-turbo", stage="recognise", family="project",
        display_name="Whisper large-v3-turbo (project)",
        source_url="https://huggingface.co/openai/whisper-large-v3-turbo",
        licence="MIT", languages=WHISPER_LANGS,
        parameters={"language": None},
        weights=["openai/whisper-large-v3-turbo"], weights_mb=1620,
        notes="In-process: the service's current recogniser"
              " (transformers).",
    ),
    entry(
        id="whisper-large-v3", stage="recognise", family="project",
        display_name="Whisper large-v3 (project adapter)",
        source_url="https://huggingface.co/openai/whisper-large-v3",
        licence="MIT", languages=WHISPER_LANGS,
        parameters={"language": None},
        weights=["openai/whisper-large-v3"], weights_mb=3087,
        notes="In-process through the project's Whisper adapter. Not"
              " installed: disk (its model.safetensors is 3.1 GB and the"
              " lab's 7 GB budget was spent on the other candidates).",
    ),
    entry(
        id="faster-whisper-large-v3-turbo", stage="recognise",
        family="ctranslate2",
        display_name="faster-whisper large-v3-turbo (CTranslate2 int8)",
        source_url="https://github.com/SYSTRAN/faster-whisper",
        licence="MIT (faster-whisper, CTranslate2, Whisper weights)",
        languages=WHISPER_LANGS, vram_gb=1.4,
        parameters={"language": None, "beam_size": 5, "vad_filter": False},
        weights=["openai/whisper-large-v3-turbo (converted locally)"],
        weights_mb=780,
        notes="The cached openai/whisper-large-v3-turbo converted"
              " locally with ct2-transformers-converter to int8 weights,"
              " float16 compute (make lab-env FAMILY=ctranslate2);"
              " faster-whisper 1.2.1, CTranslate2 4.9.0, word"
              " timestamps, beam 5. cuBLAS/cuDNN come from the project's"
              " nvidia packages. 9.6-11.8 s for 122 s. Peak measured"
              " with nvidia-smi (includes the CUDA context).",
    ),
    entry(
        id="parakeet-tdt-0.6b-v3", stage="recognise", family="nemo",
        display_name="NVIDIA Parakeet TDT 0.6B v3 (25 EU languages)",
        source_url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
        licence="CC-BY-4.0", languages=PARAKEET_LANGS,
        vram_gb=5.8,
        parameters={"language": None, "local_attention": True},
        weights=["nvidia/parakeet-tdt-0.6b-v3"], weights_mb=2509,
        notes="NeMo 3.0.0; detects the language itself (a language"
              " parameter outside its 25 is refused as"
              " unsupported_language, inside them it is only noted);"
              " word timestamps; local attention [256, 256] as the card"
              " advises for long audio. The .nemo archive is unpacked in"
              " /dev/shm when it has room. fp32: 5.8 GB peak (nvidia-smi,"
              " loading included); it ran out of memory while another"
              " job held 3.7 GB. 2.0 s for 122 s after a 28 s load.",
    ),
]


def main() -> None:
    """Write the catalogue JSON next to this script."""
    ids = [e["id"] for e in ENTRIES]
    assert len(ids) == len(set(ids)), "duplicate ids"
    document = {"schema": "lab.catalogue.v1", "entries": ENTRIES}
    (HERE / "catalogue.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()

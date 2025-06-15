import whisper  # Provided by openai-whisper
from g2p_en import G2p
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import torch
import os
import json
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import logging
from pydub import AudioSegment
import soundfile as sf
from kokoro import KPipeline
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Update CORS to allow your Firebase-hosted frontend
CORS(app, resources={r"/*": {"origins": ["https://<your-firebase-app>.web.app", "*"]}})

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
AUDIO_ERRORS_FOLDER = 'audio_errors'
TTS_OUTPUT_FOLDER = 'tts_output'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'wav'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['AUDIO_ERRORS_FOLDER'] = AUDIO_ERRORS_FOLDER
app.config['TTS_OUTPUT_FOLDER'] = TTS_OUTPUT_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(AUDIO_ERRORS_FOLDER, exist_ok=True)
os.makedirs(TTS_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Model initialization
logger.info("Loading models...")
try:
    model_whisper = whisper.load_model("base")
    processor_wav2vec = Wav2Vec2Processor.from_pretrained("excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k")
    model_wav2vec = Wav2Vec2ForCTC.from_pretrained("excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k")
    kokoro_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}", exc_info=True)
    raise

# Helper functions (unchanged)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = librosa.util.normalize(audio)
        return audio, sr
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise

def transcribe_audio(audio_path):
    audio, sr = preprocess_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    result = model_whisper.transcribe(audio, language='en')
    return result['text']

def transcription_to_phonemes(transcription):
    g2p = G2p()
    gt_phonemes = g2p(transcription)
    return ' '.join(gt_phonemes)

def normalize_gt_phonemes(gt_phonemes):
    gt_phonemes_normalized = ''.join([c for c in gt_phonemes if not c.isdigit()])
    gt_phonemes_normalized = gt_phonemes_normalized.replace("'", "").replace("-", "").replace(",", "").replace(".", "").replace("?", "").replace("!", "")
    gt_phonemes_normalized = gt_phonemes_normalized.replace("  ", " ")
    gt_phonemes_normalized = gt_phonemes_normalized.split()
    return [phoneme.strip() for phoneme in gt_phonemes_normalized if phoneme.strip()]

def audio_to_arpabet_and_timestamps(audio_path, model_wav2vec, processor_wav2vec):
    audio, sr = preprocess_audio(audio_path)
    input_values = processor_wav2vec(audio, return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        logits = model_wav2vec(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)[0]
    audio_duration = input_values.shape[1] / sr
    num_frames = logits.shape[1]
    frame_duration = audio_duration / num_frames
    raw_timit_tokens = []
    prev_id = None
    for i, token_id in enumerate(predicted_ids):
        if token_id == prev_id:
            continue
        token_str = processor_wav2vec.tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        if token_str == processor_wav2vec.tokenizer.pad_token or token_str == "|":
            continue
        start_time = i * frame_duration
        raw_timit_tokens.append((token_str.strip().lower(), round(start_time, 3)))
        prev_id = token_id
    timit_with_durations = []
    for i in range(len(raw_timit_tokens)):
        start = raw_timit_tokens[i][1]
        end = raw_timit_tokens[i + 1][1] if i + 1 < len(raw_timit_tokens) else round(audio_duration, 3)
        duration = round(end - start, 3)
        timit_with_durations.append((raw_timit_tokens[i][0], start, end, duration))
    timit_phoneme_seq = [p[0] for p in timit_with_durations]
    mapping = create_timit_to_arpabet_mapping()
    arpabet_phoneme_seq = convert_with_closure_handling(timit_phoneme_seq, mapping=mapping)
    arpabet_with_durations = []
    i = j = 0
    while i < len(timit_phoneme_seq) and j < len(arpabet_phoneme_seq):
        current = timit_phoneme_seq[i]
        if current.endswith('cl') and i + 1 < len(timit_phoneme_seq):
            next_phoneme = timit_phoneme_seq[i + 1]
            closure_to_stop = {
                'bcl': ['b', 'p'], 'dcl': ['d', 't'], 'gcl': ['g', 'k'],
                'pcl': ['p', 'b'], 'tcl': ['t', 'd'], 'kcl': ['k', 'g']
            }
            if current in closure_to_stop and next_phoneme in closure_to_stop[current]:
                i += 1
        phoneme, start, end, duration = timit_with_durations[i]
        mapped = mapping.get(phoneme, '')
        if mapped:
            arpabet_with_durations.append((arpabet_phoneme_seq[j], start, end, duration))
            j += 1
        i += 1
    return arpabet_phoneme_seq, arpabet_with_durations, audio_duration

def create_timit_to_arpabet_mapping():
    consonant_mapping = {
        'b': 'B', 'd': 'D', 'g': 'G', 'p': 'P', 't': 'T', 'k': 'K',
        'bcl': 'B', 'dcl': 'D', 'gcl': 'G', 'pcl': 'P', 'tcl': 'T', 'kcl': 'K',
        'f': 'F', 'v': 'V', 'th': 'TH', 'dh': 'DH', 's': 'S', 'z': 'Z',
        'sh': 'SH', 'zh': 'ZH', 'hh': 'HH', 'h': 'HH',
        'ch': 'CH', 'jh': 'JH',
        'm': 'M', 'n': 'N', 'ng': 'NG', 'em': 'M', 'en': 'N', 'eng': 'NG',
        'l': 'L', 'r': 'R', 'el': 'L', 'er': 'R',
        'w': 'W', 'y': 'Y',
        'q': '', 'dx': 'DX', 'hv': 'HH',
    }
    vowel_mapping = {
        'iy': 'IY', 'ih': 'IH', 'eh': 'EH', 'ey': 'EY', 'ae': 'AE', 'aa': 'AA',
        'aw': 'AW', 'ay': 'AY', 'ah': 'AH', 'ao': 'AO', 'oy': 'OY', 'ow': 'OW',
        'uh': 'UH', 'uw': 'UW', 'ux': 'UW', 'er': 'ER', 'axr': 'ER', 'ax': 'AH', 'ix': 'IH', 'ax-h': 'AH',
    }
    silence_mapping = {
        'sil': '', 'sp': '', 'spn': '', 'pau': '', 'h#': '', '#h': '', 'epi': '',
    }
    return {**consonant_mapping, **vowel_mapping, **silence_mapping}

def convert_with_closure_handling(timit_phonemes, mapping=None):
    if mapping is None:
        mapping = create_timit_to_arpabet_mapping()
    if isinstance(timit_phonemes, str):
        timit_phonemes = timit_phonemes.split()
    arpabet_phonemes = []
    i = 0
    while i < len(timit_phonemes):
        current = timit_phonemes[i].strip().lower()
        if current.endswith('cl') and i + 1 < len(timit_phonemes):
            next_phoneme = timit_phonemes[i + 1].strip().lower()
            closure_to_stop = {
                'bcl': ['b', 'p'], 'dcl': ['d', 't'], 'gcl': ['g', 'k'],
                'pcl': ['p', 'b'], 'tcl': ['t', 'd'], 'kcl': ['k', 'g']
            }
            if current in closure_to_stop and next_phoneme in closure_to_stop[current]:
                converted = mapping.get(next_phoneme, '')
                if converted:
                    arpabet_phonemes.append(converted)
                i += 2
                continue
        converted = mapping.get(current, '')
        if converted:
            arpabet_phonemes.append(converted)
        elif current not in ['sil', 'sp', 'spn', 'pau', 'h#', '#h', 'epi', 'q']:
            logger.warning(f"Unknown TIMIT phoneme '{timit_phonemes[i]}' - skipping")
        i += 1
    return arpabet_phonemes

def parse_wav2vec_phonemes(phonemes_with_timestamps: List[Tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(phonemes_with_timestamps, columns=['phoneme', 'start', 'end', 'duration'])

def parse_gt_phoneme(phonemes: List[str], audio_duration: float) -> List[dict]:
    num_phonemes = len(phonemes)
    if num_phonemes == 0:
        return []
    segment_duration = audio_duration / num_phonemes
    return [
        {"phoneme": phoneme, "start": round(i * segment_duration, 3), "end": round((i + 1) * segment_duration, 3)}
        for i, phoneme in enumerate(phonemes)
    ]

def sliding_window_comparison(reference, prediction, window_size=5):
    errors = []
    S = D = I = C = 0
    matched_preds = set()
    for i, ref_label in enumerate(reference):
        best_j = -1
        best_match = None
        search_start = max(0, i - window_size)
        search_end = min(len(prediction), i + window_size)
        for j in range(search_start, search_end):
            if j in matched_preds:
                continue
            pred_label, pred_start, pred_end = prediction[j]
            if ref_label == pred_label:
                best_j = j
                best_match = (pred_label, pred_start, pred_end)
                break
            if best_match is None:
                best_j = j
                best_match = (pred_label, pred_start, pred_end)
        if best_match:
            matched_preds.add(best_j)
            pred_label, pred_start, pred_end = best_match
            if pred_label == ref_label:
                C += 1
            else:
                S += 1
                errors.append({
                    "timestamp_start": pred_start,
                    "timestamp_end": pred_end,
                    "predicted_phoneme": pred_label,
                    "expected_phoneme": ref_label
                })
        else:
            D += 1
            errors.append({
                "timestamp_start": None,
                "timestamp_end": None,
                "predicted_phoneme": None,
                "expected_phoneme": ref_label
            })
    for j, pred in enumerate(prediction):
        if j not in matched_preds:
            I += 1
            errors.append({
                "timestamp_start": pred[1],
                "timestamp_end": pred[2],
                "predicted_phoneme": pred[0],
                "expected_phoneme": None
            })
    total = len(reference)
    PER = (S + D + I) / total if total > 0 else 0
    return {
        "errors": errors,
        "metrics": {
            "phoneme_error_rate": round(PER * 100, 2),
            "total_phonemes": total,
            "correct": C,
            "substitutions": S,
            "insertions": I,
            "deletions": D,
            "accuracy": round((C / total * 100) if total > 0 else 0, 2)
        }
    }

def generate_audio_snippet(audio_path: str, start: float, end: float, file_id: str, index: int) -> str:
    try:
        logger.info(f"Generating snippet for {file_id}_error_{index}.wav, start={start}s, end={end}s")
        audio = AudioSegment.from_wav(audio_path)
        start_ms = max(0, (start - 0.05) * 1000)  # 50ms padding
        end_ms = min(len(audio), (end + 0.05) * 1000)  # 50ms padding
        if start_ms >= end_ms:
            logger.warning(f"Invalid snippet range: start={start_ms}ms, end={end_ms}ms")
            return ""
        snippet = audio[start_ms:end_ms]
        snippet_path = os.path.join(app.config['AUDIO_ERRORS_FOLDER'], f"{file_id}_error_{index}.wav")
        snippet.export(snippet_path, format="wav", codec="pcm_s16le")
        logger.info(f"Snippet saved: {snippet_path}")
        if os.path.exists(snippet_path):
            return f"/audio_errors/{file_id}_error_{index}.wav"
        logger.error(f"Snippet file not found after export: {snippet_path}")
        return ""
    except Exception as e:
        logger.error(f"Error generating audio snippet: {str(e)}", exc_info=True)
        return ""

def combine_consecutive_errors(errors, audio_path, file_id):
    if not errors:
        return []
    
    combined_errors = []
    current_group = [errors[0]]
    current_start = errors[0]["timestamp_start"]
    current_end = errors[0]["timestamp_end"]
    current_phonemes = [errors[0]["predicted_phoneme"]]
    current_expected = [errors[0]["expected_phoneme"]]

    for error in errors[1:]:
        if (error["timestamp_start"] is not None and 
            error["timestamp_end"] is not None and 
            abs(error["timestamp_start"] - current_end) <= 0.001):
            current_group.append(error)
            current_end = error["timestamp_end"]
            current_phonemes.append(error["predicted_phoneme"])
            current_expected.append(error["expected_phoneme"])
        else:
            if current_group:
                audio_url = None
                if all(e["timestamp_start"] is not None and e["timestamp_end"] is not None for e in current_group):
                    try:
                        audio = AudioSegment.from_wav(audio_path)
                        start_ms = max(0, (current_start - 0.05) * 1000)  # 50ms padding
                        end_ms = min(len(audio), (current_end + 0.05) * 1000)  # 50ms padding
                        if start_ms < end_ms:
                            snippet = audio[start_ms:end_ms]
                            snippet_path = os.path.join(app.config['AUDIO_ERRORS_FOLDER'], f"{file_id}_error_{len(combined_errors)}.wav")
                            snippet.export(snippet_path, format="wav", codec="pcm_s16le")
                            if os.path.exists(snippet_path):
                                audio_url = f"/audio_errors/{file_id}_error_{len(combined_errors)}.wav"
                                logger.info(f"Combined snippet saved: {snippet_path}")
                            else:
                                logger.error(f"Combined snippet not found: {snippet_path}")
                    except Exception as e:
                        logger.error(f"Error combining audio snippets: {str(e)}", exc_info=True)
                
                # Filter out None values and convert to strings
                valid_phonemes = [str(p) for p in current_phonemes if p is not None]
                valid_expected = [str(e) for e in current_expected if e is not None]
                
                if not valid_phonemes:
                    logger.warning(f"Skipping error group with no valid phonemes: {current_group}")
                    continue
                
                combined_errors.append({
                    "phoneme": ", ".join(valid_phonemes),
                    "start": current_start,
                    "end": current_end,
                    "audioUser": audio_url,
                    "expected_phoneme": ", ".join(valid_expected)
                })
            
            current_group = [error]
            current_start = error["timestamp_start"]
            current_end = error["timestamp_end"]
            current_phonemes = [error["predicted_phoneme"]]
            current_expected = [error["expected_phoneme"]]
    
    # Handle the last group
    if current_group:
        audio_url = None
        if all(e["timestamp_start"] is not None and e["timestamp_end"] is not None for e in current_group):
            try:
                audio = AudioSegment.from_wav(audio_path)
                start_ms = max(0, (current_start - 0.05) * 1000)  # 50ms padding
                end_ms = min(len(audio), (current_end + 0.05) * 1000)  # 50ms padding
                if start_ms < end_ms:
                    snippet = audio[start_ms:end_ms]
                    snippet_path = os.path.join(app.config['AUDIO_ERRORS_FOLDER'], f"{file_id}_error_{len(combined_errors)}.wav")
                    snippet.export(snippet_path, format="wav", codec="pcm_s16le")
                    if os.path.exists(snippet_path):
                        audio_url = f"/audio_errors/{file_id}_error_{len(combined_errors)}.wav"
                        logger.info(f"Combined snippet saved: {snippet_path}")
                    else:
                        logger.error(f"Combined snippet not found: {snippet_path}")
            except Exception as e:
                logger.error(f"Error combining audio snippets: {str(e)}", exc_info=True)
        
        # Filter out None values and convert to strings
        valid_phonemes = [str(p) for p in current_phonemes if p is not None]
        valid_expected = [str(e) for e in current_expected if e is not None]
        
        if not valid_phonemes:
            logger.warning(f"Skipping error group with no valid phonemes: {current_group}")
        else:
            combined_errors.append({
                "phoneme": ", ".join(valid_phonemes),
                "start": current_start,
                "end": current_end,
                "audioUser": audio_url,
                "expected_phoneme": ", ".join(valid_expected)
            })
    
    return combined_errors

def generate_tts_audio(text: str, voice_name: str = 'af_bella', lang_code: str = 'a', speed: float = 0.7) -> str:
    try:
        logger.info(f"Generating TTS audio for text: {text[:50]}..., voice={voice_name}, speed={speed}")
        if not text.strip():
            logger.error("Empty text provided for TTS")
            return ""
        logger.debug("Calling kokoro_pipeline")
        generator = kokoro_pipeline(text, voice=voice_name, speed=speed)
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is None or len(audio) == 0:
                logger.error("Kokoro generated empty audio")
                return ""
            file_id = str(uuid.uuid4())
            output_path = os.path.join(app.config['TTS_OUTPUT_FOLDER'], f"{file_id}.wav")
            logger.debug(f"Writing audio to: {output_path}")
            sf.write(output_path, audio, 24000)
            logger.info(f"TTS audio saved: {output_path}")
            if not os.path.exists(output_path):
                logger.error(f"TTS file not found after export: {output_path}")
                return ""
            return f"/tts_audio/{file_id}.wav"
        logger.error("No audio generated by Kororo")
        return ""
    except Exception as e:
        logger.error(f"Error generating TTS audio: {str(e)}", exc_info=True)
        return ""

def cleanup_old_files(folder, max_age_seconds=86400):
    try:
        for file in Path(folder).glob('*.wav'):
            if time.time() - file.stat().st_mtime > max_age_seconds:
                file.unlink()
                logger.info(f"Deleted old file: {file}")
    except Exception as e:
        logger.error(f"Error cleaning up files in {folder}: {str(e)}")

# Endpoints
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only WAV files are allowed"}), 400
    if file.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "File size exceeds 50MB limit"}), 400
    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
    try:
        file.save(file_path)
        logger.info(f"Processing transcription for file_id: {file_id}")
        transcription = transcribe_audio(file_path)
        logger.info(f"Transcription completed for file_id: {file_id}")
        return jsonify({"text": transcription, "file_id": file_id})
    except Exception as e:
        logger.error(f"Error in /transcribe: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up audio file: {file_path}")

@app.route('/process', methods=['POST'])
def process():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only WAV files are allowed"}), 400
    if file.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "File size exceeds 50MB limit"}), 400
    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
    try:
        file.save(file_path)
        logger.info(f"Starting processing for file_id: {file_id}")
        # Check for custom transcription in form data
        transcription = request.form.get('transcription', None)
        if transcription:
            logger.info(f"Using provided transcription: {transcription[:50]}...")
        else:
            transcription = transcribe_audio(file_path)
            logger.info(f"Generated transcription: {transcription[:50]}...")
        gt_phonemes = transcription_to_phonemes(transcription)
        normalized_gt_phonemes = normalize_gt_phonemes(gt_phonemes)
        arpabet_phonemes, arpabet_phonemes_with_timestamps, audio_duration = audio_to_arpabet_and_timestamps(file_path, model_wav2vec, processor_wav2vec)
        wav2vec_df = parse_wav2vec_phonemes(arpabet_phonemes_with_timestamps)
        gt_phonemes_with_timestamps = parse_gt_phoneme(normalized_gt_phonemes, audio_duration)
        prediction = [(phoneme, start, end) for phoneme, start, end, duration in arpabet_phonemes_with_timestamps]
        comparison_result = sliding_window_comparison(normalized_gt_phonemes, prediction, window_size=5)
        errors_with_audio = combine_consecutive_errors(comparison_result['errors'], file_path, file_id)
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{file_id}.json")
        result_data = {
            "transcription": transcription,
            "phonemesModel": wav2vec_df[['phoneme', 'start', 'end']].to_dict(orient='records'),
            "phonemesGroundTruth": gt_phonemes_with_timestamps,
            "accuracy": comparison_result['metrics']['accuracy'],
            "per": comparison_result['metrics']['phoneme_error_rate'],
            "errors": errors_with_audio
        }
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        logger.info(f"Processing completed for file_id: {file_id}")
        cleanup_old_files(AUDIO_ERRORS_FOLDER)
        cleanup_old_files(TTS_OUTPUT_FOLDER)
        return jsonify({"file_id": file_id, "message": "Processing complete"})
    except Exception as e:
        logger.error(f"Error in /process: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up audio file: {file_path}")

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    if not data or 'text' not in data:
        logger.error("TTS request missing text")
        return jsonify({"error": "Text is required"}), 400
    text = data['text']
    voice_name = data.get('voice_name', 'af_bella')
    speed = data.get('speed', 1.2)
    try:
        logger.info(f"Received TTS request for text: {text[:50]}...")
        audio_url = generate_tts_audio(text, voice_name=voice_name, speed=speed)
        if not audio_url:
            logger.error("Failed to generate TTS audio")
            return jsonify({"error": "Failed to generate TTS audio"}), 500
        logger.info(f"TTS audio generated: {audio_url}")
        return jsonify({"audio_url": audio_url})
    except Exception as e:
        logger.error(f"Error in /tts: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/tts_audio/<filename>', methods=['GET'])
def serve_tts_audio(filename):
    file_path = os.path.join(app.config['TTS_OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        logger.info(f"Serving TTS audio file: {file_path}")
        return send_file(file_path, mimetype='audio/wav')
    logger.error(f"TTS audio file not found: {file_path}")
    return jsonify({"error": "Audio file not found"}), 404

@app.route('/data', methods=['GET'])
def get_data():
    file_id = request.args.get('file_id')
    if not file_id:
        logger.error("Missing file_id in /data request")
        return jsonify({"error": "file_id query parameter is required"}), 400
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{file_id}.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Served results for file_id: {file_id}")
        return jsonify(data)
    logger.warning(f"Results not found for file_id: {file_id}")
    return jsonify({"error": "Results not found"}), 404

@app.route('/audio_errors/<filename>', methods=['GET'])
def serve_audio_error(filename):
    file_path = os.path.join(app.config['AUDIO_ERRORS_FOLDER'], filename)
    if os.path.exists(file_path):
        logger.info(f"Serving audio file: {file_path}")
        return send_file(file_path, mimetype='audio/wav')
    logger.error(f"Audio file not found: {file_path}")
    return jsonify({"error": "Audio file not found"}), 404

@app.route('/static/<filename>', methods=['GET'])
def serve_static(filename):
    file_path = os.path.join(app.config['STATIC_FOLDER'], filename)
    if os.path.exists(file_path):
        logger.info(f"Serving static file: {file_path}")
        return send_file(file_path, mimetype='audio/wav')
    logger.error(f"Static file not found: {file_path}")
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    # Use environment variables for host and port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
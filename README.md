This application was done as part of TMS4043 Artificial Intelligence course.

**Phoneme Recognition and Pronunciation Checker (PRPC) Backend**

This project presents the development of an AI-powered pronunciation checker application that evaluates spoken English pronunciation by comparing user audio input phoneme with standard phoneme sequences from text. The application integrates pretrained models, Whisper for speech-to-text transcription, Wav2Vec for audio to phoneme recognition, and Kokoro for corrective audio feedback. Built using React, Vite, TypeScript, and Python, the application addresses the real-world need for accessible pronunciation training tools

**Application Flowchart**

<img width="1622" height="475" alt="image" src="https://github.com/user-attachments/assets/9c2ef06f-3ad3-4645-8e49-c6b382ffbaaf" />

The application was structured with a Flask backend and a React frontend. Where the services of the backend are exposed as API endpoints.

Following are the backend services provided,

| **Service**                  | **Tool/Model Used**                 | **Key Functions**                            |
|-----------------------------|-------------------------------------|----------------------------------------------|
| **Preprocessing**           | Librosa, pydub                      | Resampling, normalization, conversion        |
| **Transcription**           | Whisper                             | Converts speech to text                      |
| **Grapheme-to-Phoneme (G2P)** | g2p_en + custom normalization      | Generate ground-truth phoneme                |
| **Audio-to-Phoneme Extraction** | Wav2Vec2.0 + custom mapper       | Extracts phonemes from speech                |
| **Phoneme Comparison**      | Sliding window algorithm            | Compares and highlights errors               |
| **Text-to-Speech (TTS)**    | Kokoro TTS                          | Generate correct pronunciation audio         |

The preprocessing service prepares user-uploaded audio for analysis. Using Librosa, the audio is resampled to 16 kHz and normalized for consistent amplitude, while pydub ensures format compatibility by converting files into mono-channel WAV format. This ensures that the audio aligns with the input requirements of the downstream models. 

Next, the transcription service leverages the pretrained Whisper base model to convert speech into text. The transcribed output is then cleaned by converting it to lowercase and removing unnecessary punctuation to maintain consistency for phoneme conversion. 

The Grapheme-to-Phoneme (G2P) service utilizes the g2p_en package to convert the transcription text into ARPAbet phoneme sequences (Ribeiro et al., 2023). A custom normalization function is applied to remove stress markers and other artifacts, ensuring alignment with the output format of the phoneme extraction model. 

For phoneme prediction from audio, the Wav2Vec phoneme extraction uses a pretrained model (excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit4k) to generate phonemes directly from speech (Excalibur12, 2024). The output, initially in TIMIT phoneme format, is mapped to ARPAbet using a custom converter. The result is a sequence of phonemes with associated timestamps. 

The phoneme comparison aligns and compares the phoneme outputs from G2P and Wav2Vec using a custom sliding window algorithm. The comparison identifies mispronunciations, insertions, and deletions, and outputs a structured JSON object containing the phoneme match results and their respective time locations. 

To help users correct their pronunciation, the Text-to-Speech (TTS) service uses the Kokoro TTS model to generate high-quality reference audio (Hexgrad, 2025). This service can synthesize entire words or specific phoneme sequences with configurable voice and speed parameters to enhance learning clarity. 

**Model Selection & Rationale,**

**Whisper (OpenAI)**

The Whisper model by OpenAI is used for transcribing spoken audio into text. Its inclusion is justified by its strong performance in transcribing speech from speakers with diverse English accents, as well as its resilience in noisy environments. These capabilities are essential for ensuring that users’ recordings are accurately transcribed, even in less-than-ideal conditions. As reported by Radford et al. (2022), Whisper achieves state-of-the-art results in multilingual and multitask speech recognition, making it a reliable foundation for generating accurate phoneme ground truths in this application (Amorese et al., 2023). 

**Wav2Vec 2.0 (Facebook AI)**

Wav2Vec 2.0, specifically the version fine-tuned on the TIMIT dataset (excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k), is used to extract phonemes directly from audio. This model was selected for its high precision in phoneme recognition tasks, particularly at the segmental level (Oh et al., 2021). Baevski et al. (2020) highlight Wav2Vec 2.0’s ability to learn high-quality speech representations from raw audio, making it well-suited for detecting subtle pronunciation errors. Its fine-tuning on the TIMIT dataset ensures it performs reliably on phoneme-level distinctions, which is crucial for providing accurate feedback in this application. 

**Kokoro TTS**

The Kokoro-82M model is integrated for generating reference audio that demonstrates the correct pronunciation of words or phonemes. The decision to use Kokoro is supported by its reputation for producing high-quality, natural-sounding speech. The model has been well received by the Hugging Face community for its clarity and expressiveness, which are important qualities in an educational context (Hexgrad, 2025). Providing users with clear reference audio enhances their ability to understand and mimic correct pronunciations, contributing significantly to the learning process.  

**Phoneme Comparison**

The phoneme comparison detects mispronunciations by aligning reference or ground truth phonemes (from G2P) with predicted phonemes (from Wav2Vec) using a sliding window alignment. It identifies errors such as substitutions (S), insertions (I), and deletions (D), and generates an error report with metrics like Phoneme Error Rate (PER) and accuracy for user feedback. The diagram in Figure 6 visually represents this alignment process across six steps, offering a clear illustration of how the algorithm matches phonemes within a defined window.

<img width="755" height="662" alt="image" src="https://github.com/user-attachments/assets/bbd7e58f-506a-4b99-98d2-00419f4cd33a" />

The diagram depicts a sequence of reference phonemes (A, B, C, D, E, F, G, H) and a corresponding sequence of predicted phonemes (A, B, X, D, E, F, H). The window size in the example provided was set to be ±3 positions. Coloured boxes highlight the current reference phoneme (green for a match, orange for a mismatch or error), and the process is explained as follows: 

Step 1: Reference A and B (Indices 0 and 1) 

The window starts at index 0, aligning reference phoneme A with predicted phoneme A, then shifts to index 1 to match B with predicted B. Green highlights indicate correct matches (C += 1 for each), and the indices (0 and 1) are marked as matched 

Step 2: Reference C (Index 2)  

For reference phoneme C, the window spans predicted phonemes around index 2. No exact match is found (C is absent in the predicted sequence), and the algorithm selects the closest unmatched phoneme, X. The orange box indicates a substitution error (S += 1), and an error entry is recorded with the predicted phoneme (X) and expected phoneme (C). 

Step 3: Reference D, E, and F (Indices 3, 4, and 5)  

The window moves to index 3, aligning D with predicted D, then to index 4 for E with predicted E, and index 5 for F with predicted F. Green highlights signify correct matches (C += 1 for each), with indices (3, 4, and 5) marked as used. 

Step 4: Reference G (Index 6)  

The window aligns reference phoneme G with predicted phoneme H at index 6. Orange highlight indicates a substitution error (S += 1), recorded in errors with predicted=H, expected=G. 

Step 5: Post-Processing (Unmatched Predicted Phonemes)  

After processing all reference phonemes, the algorithm checks for unmatched predicted phonemes (none in this case, as the sequence ends at H). No insertions (I += 1) are recorded, but this step would apply if extra predicted phonemes existed. 

**Installation**

Install the dependencies from the _requirements.txt_ and make sure to also install Kokoro TTS, which **may** require you to run the program in Linux or WSL if on Windows.

Update the following line :

CORS(app, resources={r"/*": {"origins": ["https://ai-phoneme-checker.web.app", "*"]}})

According to your frontend URL or remove if hosting locally.

# PolyComs
**Track:** 4.4 AI + Languages
**Team:** [Your Name]

## The problem
Many translation tools are built around English and widely spoken European languages, leaving speakers of less-served languages—and users with visual, cognitive, or hearing barriers—behind. They also present uncertain machine outputs as fact without showing confidence, alternatives, or safety warnings.

## What I built
PolyComs is an accessible, machine-learning-powered translation web app built in Python and Google Colab. Users can type (including via an ARIA on-screen keyboard), upload audio, or speak into a microphone; PolyComs identifies the language, translates it with an honesty/confidence meter and labeled alternatives (polite, casual, differing grammar), simplifies complex text into Easy-Read or step-by-step chunks while preserving medical/driving warnings, and reads translations aloud sentence-by-sentence. Accessibility controls include Light/Dark mode, colorblindness and high-contrast palettes, dyslexia-friendly fonts, adjustable spacing, a reading ruler, slower speech playback, and Safe Language Mode.

## How it works
* **Data:** Trained and tested Language Identification on Tatoeba (`sentences.csv`), dropping missing rows (`dropna()`), sampling up to 2,000 sentences per language, and splitting 75% train / 25% held-out test (`stratify=lang`). Translation quality was benchmarked on parallel FLORES+-aligned sentences across real-world domains (medication, driving, school notices, safety warnings, idioms).
* **Models:**
  * **Language ID:** `TfidfVectorizer` (`FeatureUnion` of `char_wb` 2–4 n-grams and `word` 1–2 n-grams) + `LogisticRegression` (`class_weight="balanced"`), benchmarked against `langdetect`.
  * **Translation:** `facebook/nllb-200-distilled-600M` (loaded in `float16` on a T4 GPU) with beam search for confidence and alternative phrasing.
  * **Speech:** `openai/whisper-small` for multilingual Speech-to-Text (evaluated with `jiwer` WER), `facebook/mms-tts-yor` for Yoruba TTS, and `edge-tts` / `gTTS` for the remaining 23 languages.
  * **Backend/UI:** FastAPI, Uvicorn, Cloudflare Tunnel (`cloudflared`), `sacrebleu` (`chrF`), plus custom Easy-Read, key-info extraction, and Safe Language Mode layers.
  * ** OCR from EasyOCR to detect languages from images, and pictures. This is now both in the "Translate & Explain" and "Language ID" sections, for ease of use.

## Results
* **Language Identification (4.4.1):** On held-out Tatoeba test data, PolyComs achieved **95.7% overall accuracy**, reaching **100%** on Swahili (`swh`, 500 test samples) and outperforming `langdetect` on less-served languages like Yoruba (`yor`, where `langdetect` fails because Yoruba is unsupported). Its lowest score was Indonesian (`ind`) at **78.2%**, largely due to lexical overlap with look-alike Malay (`zsm`).
* **Translation Quality (`chrF`, 4.4.2):** Widely served Latin-script European languages scored highest—French (**79.69**), Spanish (**78.52**), and German (**71.69**)—followed by Cyrillic languages Russian (**54.22**) and Ukrainian (**48.11**). Non-Latin CJK languages scored lower—Mandarin (**29.63**), Japanese (**25.19**), and Korean (**22.83**)—and Yoruba scored lowest at **10.52**, including a critical medical error where *"tablets"* was mistranslated as *"injections"*.
* ** Language ID in Japanese appears to believe English, likely due to not understanding Romaji writing of the language. For example, "arigato", and "arigatogozaimasu" (thank you, and thank you very much), are detected as Cantonese, with 13% confidence, which is obviously incorrect. This may expand to other Language ID issues with typing in scripts other than their native writing methods and scripts.
* ** Without full training upon loads, ID accuracy drops significantly, to ~74%.
* **Takeaway:** While `chrF` penalizes valid differences in grammar or politeness (especially in Japanese/Korean), low scores on high-stakes medical text show why confidence meters and human-verification warnings are essential.
* ** PLEASE NOTE - WHILST THIS TOOL MAY ATTEMPT TO TRANSLATE MEDICAL OR IMPORTANT INSTRUCTIONS, THIS IS A PROTOTYPE, AND SHOULD NOT BE RELIED ON FOR IMPORTANT INFORMATION, SUCH AS MEDICATION DOSAGES, FINANCIAL INFORMATION, OR LEGAL INFORMATION!!!

## What I'd do next
1. Investigate why Yoruba (`yor_Latn`) scored `10.52` `chrF` despite using the Latin script—likely due to scarce training data and tone-mark sensitivity—and evaluate larger or domain-adapted models for Yoruba, Cyrillic, Japanese, Korean, and Mandarin.
2. Expand real-speaker validation and test targeted fine-tuning or glossary constraints for medical terms so *"tablets"* is never swapped for *"injections"*.

## Sources
* Tatoeba Sentences Dataset: https://tatoeba.org/en/downloads
* FLORES+ Benchmark Dataset: https://huggingface.co/datasets/openlanguagedata/flores_plus
* NLLB-200 (`facebook/nllb-200-distilled-600M`): https://huggingface.co/facebook/nllb-200-distilled-600M
* Whisper (`openai/whisper-small`): https://huggingface.co/openai/whisper-small
* Yoruba TTS (`facebook/mms-tts-yor`): https://huggingface.co/facebook/mms-tts-yor
* Edge-TTS: https://github.com/rany2/edge-tts
* Libraries: `scikit-learn`, `pandas`, `sacrebleu`, `jiwer`, `langdetect`, `gTTS`, `fastapi`, `uvicorn`
* AI Tools Used: Google Gemini (coding support and rubric alignment) and OpenAI ChatGPT (understanding `chrF` metrics, write-up structure, and troubleshooting).
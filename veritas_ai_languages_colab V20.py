"""
Veritas AI International High School Hackathon 2026 — Track 4.4: AI + Languages
==============================================================================
All-in-One Google Colab Backend + Embedded Local HTML/CSS/JS Frontend

Models & Datasets Used:
- 4.4.1 Language Identification: TF-IDF (char n-gram 1..3) + LogisticRegression
        trained on Tatoeba sentences, benchmarked against `langdetect`.
- 4.4.2 Translation & Quality: `facebook/nllb-200-distilled-600M` (Hugging Face)
        evaluated with `sacrebleu` (chrF) on FLORES+ / parallel test sets.
- 4.4.3 Speech-to-Text: `openai/whisper-small` + `jiwer` (WER) for audio input.

How to run in Google Colab:
1. Select Runtime -> Change runtime type -> T4 GPU (recommended, CPU also works).
2. Paste this entire script into a Colab cell and run it.
3. Click the Colab-served link (or view the embedded iframe) to use the app!
   It also saves `veritas_languages_app.html` locally in your Colab workspace.
"""

import os
import sys
import subprocess
import threading
import time
import io
import json
import re
import tempfile
from typing import List, Dict, Optional

# ==============================================================================
# 1. AUTO-INSTALL DEPENDENCIES (Safe for Google Colab & Local Python)
# ==============================================================================
REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "python-multipart",
    "transformers",
    "sentencepiece",
    "accelerate",
    "sacrebleu",
    "scikit-learn",
    "pandas",
    "langdetect",
    "openai-whisper",
    "jiwer",
    "edge-tts",
    "gTTS",
    "scipy",
    "easyocr",
    "pillow",
]

def ensure_dependencies():
    missing = []
    import_map = {
        "python-multipart": "multipart",
        "scikit-learn": "sklearn",
        "openai-whisper": "whisper",
        "pillow": "PIL",
    }
    for pkg in REQUIRED_PACKAGES:
        mod = import_map.get(pkg, pkg.replace("-", "_"))
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    import shutil
    try:
        from typing_extensions import sentinel  # noqa: F401
    except ImportError:
        missing.append("typing-extensions>=4.12.2")
    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg  # type: ignore[import-not-found]
        except ImportError:
            missing.append("imageio-ffmpeg")
    if missing:
        print(f"[Setup] Installing/updating packages: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + missing)
        print("[Setup] Dependencies installed.")

if __name__ == "__main__":
    ensure_dependencies()


import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
import edge_tts
from gtts import gTTS
from scipy.io import wavfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from langdetect import detect, detect_langs, LangDetectException
import sacrebleu
import whisper
import easyocr  # type: ignore[import-not-found]
from PIL import Image
from jiwer import wer as compute_wer

# ==============================================================================
# 2. LANGUAGE METADATA & CONFUSABLE PAIRS
# ==============================================================================
LANGUAGES = {
    "eng_Latn": {"name": "English", "iso2": "en", "iso3": "eng", "tier": "Widely Served", "rtl": False},
    "fra_Latn": {"name": "French", "iso2": "fr", "iso3": "fra", "tier": "Widely Served", "rtl": False},
    "spa_Latn": {"name": "Spanish", "iso2": "es", "iso3": "spa", "tier": "Widely Served", "rtl": False},
    "deu_Latn": {"name": "German", "iso2": "de", "iso3": "deu", "tier": "Widely Served", "rtl": False},
    "swe_Latn": {"name": "Swedish", "iso2": "sv", "iso3": "swe", "tier": "Widely Served", "rtl": False},
    "fin_Latn": {"name": "Finnish", "iso2": "fi", "iso3": "fin", "tier": "Moderate", "rtl": False},
    "nob_Latn": {"name": "Norwegian (Bokmål)", "iso2": "no", "iso3": "nob", "tier": "Moderate", "rtl": False},
    "por_Latn": {"name": "Portuguese", "iso2": "pt", "iso3": "por", "tier": "Widely Served", "rtl": False},
    "rus_Cyrl": {"name": "Russian", "iso2": "ru", "iso3": "rus", "tier": "Widely Served", "rtl": False},
    "ukr_Cyrl": {"name": "Ukrainian", "iso2": "uk", "iso3": "ukr", "tier": "Moderate", "rtl": False},
    "jpn_Jpan": {"name": "Japanese", "iso2": "ja", "iso3": "jpn", "tier": "Widely Served", "rtl": False},
    "kor_Hang": {"name": "Korean", "iso2": "ko", "iso3": "kor", "tier": "Widely Served", "rtl": False},
    "zho_Hans": {"name": "Chinese (Mandarin)", "iso2": "zh", "iso3": "cmn", "tier": "Widely Served", "rtl": False},
    "yue_Hant": {"name": "Chinese (Cantonese)", "iso2": "yue", "iso3": "yue", "tier": "Less Served", "rtl": False},
    "vie_Latn": {"name": "Vietnamese", "iso2": "vi", "iso3": "vie", "tier": "Moderate", "rtl": False},
    "tgl_Latn": {"name": "Tagalog", "iso2": "tl", "iso3": "tgl", "tier": "Less Served", "rtl": False},
    "ind_Latn": {"name": "Indonesian", "iso2": "id", "iso3": "ind", "tier": "Moderate", "rtl": False},
    "zsm_Latn": {"name": "Malay", "iso2": "ms", "iso3": "zsm", "tier": "Less Served", "rtl": False},
    "swh_Latn": {"name": "Swahili", "iso2": "sw", "iso3": "swh", "tier": "Less Served", "rtl": False},
    "yor_Latn": {"name": "Yoruba", "iso2": "yo", "iso3": "yor", "tier": "Less Served", "rtl": False},
    "mri_Latn": {"name": "Māori (Polynesian)", "iso2": "mi", "iso3": "mri", "tier": "Less Served", "rtl": False},
    "smo_Latn": {"name": "Samoan (Polynesian)", "iso2": "sm", "iso3": "smo", "tier": "Less Served", "rtl": False},
    "arb_Arab": {"name": "Arabic", "iso2": "ar", "iso3": "arb", "tier": "Moderate", "rtl": True},
    "hin_Deva": {"name": "Hindi", "iso2": "hi", "iso3": "hin", "tier": "Moderate", "rtl": False},
}

CONFUSABLE_PAIRS = {
    "ind_Latn": ("zsm_Latn", "Indonesian and Malay share high lexical overlap; check words like 'kereta' (car in Malay vs. train/carriage in Indonesian) or 'bisa' (can vs. poison)."),
    "zsm_Latn": ("ind_Latn", "Malay and Indonesian are closely related; verify regional vocabulary if the input is short."),
    "spa_Latn": ("por_Latn", "Spanish and Portuguese share many cognates; short phrases can be ambiguous."),
    "por_Latn": ("spa_Latn", "Portuguese and Spanish share many cognates; check diacritics (ã, õ, ç vs. ñ)."),
    "swe_Latn": ("nob_Latn", "Swedish and Norwegian share many cognates and false friends (e.g., 'rolig' means funny in Swedish but calm in Norwegian)."),
    "nob_Latn": ("swe_Latn", "Norwegian and Swedish are closely related; check false friends and letters (æ/ø vs. ä/ö)."),
}

# ==============================================================================
# 3. 4.4.1 LANGUAGE IDENTIFICATION (Tatoeba-style dataset + TF-IDF + LogReg)
# ==============================================================================
# Built-in curated Tatoeba-aligned starter corpus (used immediately so the app
# works out-of-the-box; you can also point TATOEBA_TSV_PATH to a downloaded
# Tatoeba `sentences.csv` for large-scale training).
TATOEBA_STARTER_DATA = [
    # English (eng)
    ("eng", "I need to take this medicine twice a day after eating."),
    ("eng", "Could you please tell me how to get to the train station?"),
    ("eng", "She speaks three languages fluently and is learning a fourth."),
    ("eng", "It is raining cats and dogs outside right now."),
    ("eng", "Do not drive if this medication makes you feel dizzy."),
    ("eng", "Where is the nearest hospital or pharmacy?"),
    ("eng", "That test was a piece of cake for everyone in class."),
    ("eng", "Hey, what's up? Wanna grab lunch later?"),
    ("eng", "Please submit the completed form before Friday afternoon."),
    ("eng", "Water boils at one hundred degrees Celsius."),
    ("eng", "I am looking for a quiet place to study."),
    ("eng", "Can you repeat that more slowly, please?"),
    # French (fra)
    ("fra", "Je dois prendre ce médicament deux fois par jour après le repas."),
    ("fra", "Pourriez-vous me dire comment aller à la gare, s'il vous plaît ?"),
    ("fra", "Elle parle couramment trois langues et en apprend une quatrième."),
    ("fra", "Il pleut des cordes dehors en ce moment."),
    ("fra", "Ne conduisez pas si ce médicament vous donne des vertiges."),
    ("fra", "Où se trouve l'hôpital ou la pharmacie la plus proche ?"),
    ("fra", "Cet examen était du gâteau pour toute la classe."),
    ("fra", "Salut, ça va ? On mange ensemble tout à l'heure ?"),
    ("fra", "Veuillez soumettre le formulaire rempli avant vendredi après-midi."),
    ("fra", "L'eau bout à cent degrés Celsius."),
    ("fra", "Je cherche un endroit calme pour étudier."),
    ("fra", "Pouvez-vous répéter cela plus lentement, s'il vous plaît ?"),
    # Spanish (spa)
    ("spa", "Tengo que tomar esta medicina dos veces al día después de comer."),
    ("spa", "¿Podría decirme cómo llegar a la estación de tren, por favor?"),
    ("spa", "Ella habla tres idiomas con fluidez y está aprendiendo un cuarto."),
    ("spa", "Está lloviendo a cántaros afuera en este momento."),
    ("spa", "No conduzca si este medicamento le causa mareos."),
    ("spa", "¿Dónde está el hospital o la farmacia más cercana?"),
    ("spa", "Ese examen fue pan comido para todos en la clase."),
    ("spa", "Hola, ¿qué tal? ¿Vamos a almorzar más tarde?"),
    ("spa", "Por favor entregue el formulario completo antes del viernes por la tarde."),
    ("spa", "El agua hierve a cien grados Celsius."),
    ("spa", "Estoy buscando un lugar tranquilo para estudiar."),
    ("spa", "¿Puede repetir eso más despacio, por favor?"),
    # Swahili (swh)
    ("swh", "Ninahitaji kunywa dawa hii mara mbili kwa siku baada ya kula."),
    ("swh", "Tafadhali unaweza kuniambia njia ya kufika kituo cha treni?"),
    ("swh", "Anazungumza lugha tatu kwa ufasaha na anajifunza ya nne."),
    ("swh", "Mvua kubwa inanyesha sana nje hivi sasa."),
    ("swh", "Usiendeshe gari ikiwa dawa hii inakufanya ujisikie kizunguzungu."),
    ("swh", "Hospitali au duka la dawa lililo karibu liko wapi?"),
    ("swh", "Mtihani huo ulikuwa rahisi sana kwa kila mtu darasani."),
    ("swh", "Mambo vipi? Twende tukale chakula cha mchana baadaye?"),
    ("swh", "Tafadhali wasilisha fomu iliyojazwa kabla ya Ijumaa alasiri."),
    ("swh", "Maji huchemka katika nyuzi joto mia moja za Selsiasi."),
    ("swh", "Ninatafuta mahali patulivu pa kujisomea."),
    ("swh", "Unaweza kurudia hilo polepole zaidi, tafadhali?"),
    # Yoruba (yor)
    ("yor", "Mo nilo lati mu oogun yii lẹẹmeji lojumọ lẹhin ti mo ba jẹun."),
    ("yor", "Jọwọ ṣe o le sọ fun mi bi mo ṣe le de ibudo ọkọ oju irin?"),
    ("yor", "O n sọ ede mẹta daradara o si n kọ ẹkọ kẹrin."),
    ("yor", "Ojo n rọ gidigidi ni ita lọwọlọwọ yii."),
    ("yor", "Maṣe wa ọkọ ti oogun yii ba n mu ori rẹ yi."),
    ("yor", "Nibo ni ile-iwosan tabi ile-itaja oogun ti o sunmọ wa?"),
    ("yor", "Idanwo naa rọrun pupọ fun gbogbo eniyan ni kilasi."),
    ("yor", "Bawo ni? Ṣe a le jẹun papọ nigbamii?"),
    ("yor", "Jọwọ fi fọọmu ti a ti kun silẹ ṣaaju ọjọ Jimọ ọsan."),
    ("yor", "Omi n ho ni iwọn ogọrun Sẹlsiọsi."),
    ("yor", "Mo n wa ibi idakẹjẹ lati kawe."),
    ("yor", "Ṣe o le tun sọ iyẹn ni pẹlẹbẹ, jọwọ?"),
    # Indonesian (ind)
    ("ind", "Saya harus minum obat ini dua kali sehari setelah makan."),
    ("ind", "Bisakah Anda memberi tahu saya cara menuju stasiun kereta?"),
    ("ind", "Dia berbicara tiga bahasa dengan lancar dan sedang belajar bahasa keempat."),
    ("ind", "Di luar sedang hujan sangat deras sekarang."),
    ("ind", "Jangan mengemudi jika obat ini membuat Anda merasa pusing."),
    ("ind", "Di mana rumah sakit atau apotek terdekat?"),
    ("ind", "Ujian itu sangat mudah bagi semua orang di kelas."),
    ("ind", "Hai, apa kabar? Mau makan siang bareng nanti?"),
    ("ind", "Harap serahkan formulir yang sudah diisi sebelum Jumat sore."),
    ("ind", "Air mendidih pada suhu seratus derajat Celsius."),
    ("ind", "Saya sedang mencari tempat yang tenang untuk belajar."),
    ("ind", "Bisakah Anda mengulanginya lebih pelan, tolong?"),
    # Malay (zsm)
    ("zsm", "Saya perlu makan ubat ini dua kali sehari selepas makan."),
    ("zsm", "Bolehkah anda beritahu saya cara ke stesen kereta api?"),
    ("zsm", "Dia bertutur tiga bahasa dengan fasih dan sedang belajar bahasa keempat."),
    ("zsm", "Hujan turun dengan sangat lebat di luar sekarang."),
    ("zsm", "Jangan memandu jika ubat ini membuatkan anda berasa pening."),
    ("zsm", "Di manakah hospital atau farmasi yang paling dekat?"),
    ("zsm", "Peperiksaan itu sangat senang bagi semua pelajar di dalam kelas."),
    ("zsm", "Hai, apa khabar? Jom makan tengah hari nanti?"),
    ("zsm", "Sila hantar borang yang telah lengkap sebelum petang Jumaat."),
    ("zsm", "Air mendidih pada suhu seratus darjah Celsius."),
    ("zsm", "Saya sedang mencari tempat yang sunyi untuk mengulang kaji."),
    ("zsm", "Boleh anda ulang semula dengan lebih perlahan?"),
    # Japanese (jpn)
    ("jpn", "この薬は食後に一日二回飲む必要があります。"),
    ("jpn", "駅への行き方を教えていただけますか？"),
    ("jpn", "彼女は三つの言語を流暢に話し、四つ目を学んでいます。"),
    ("jpn", "今、外では土砂降りの雨が降っています。"),
    ("jpn", "この薬を飲んでめまいがする場合は運転しないでください。"),
    ("jpn", "一番近い病院や薬局はどこですか？"),
    ("jpn", "そのテストはクラスの全員にとって朝飯前でした。"),
    ("jpn", "ねえ、元気？あとで一緒にお昼ご飯食べない？"),
    ("jpn", "金曜日の午後までに記入済みの書類を提出してください。"),
    ("jpn", "水は摂氏百度で沸騰します。"),
    ("jpn", "勉強するための静かな場所を探しています。"),
    ("jpn", "もう少しゆっくり繰り返していただけますか？"),
    # Extra Yoruba (without diacritics for real-world coverage)
    ("yor", "Mo nilo lati mu oogun yii ni emeji lojumo leyin ti mo ba jeun."),
    ("yor", "Jowo se o le so fun mi bi mo se le de ibudo oko oju irin?"),
    ("yor", "E ku aro, bawo ni ara yin loni?"),
    ("yor", "Oko mi ko si nibe, e jowo e ran mi lowo."),
    # German (deu)
    ("deu", "Ich muss dieses Medikament zweimal täglich nach dem Essen einnehmen."),
    ("deu", "Könnten Sie mir bitte sagen, wie ich zum Bahnhof komme?"),
    ("deu", "Fahren Sie nicht Auto, wenn Sie sich schwindelig fühlen."),
    ("deu", "Wo ist das nächste Krankenhaus oder die nächste Apotheke?"),
    # Russian (rus)
    ("rus", "Мне нужно принимать это лекарство два раза в день после еды."),
    ("rus", "Не подскажете, как пройти к железнодорожному вокзалу?"),
    ("rus", "Не садитесь за руль, если лекарство вызывает головокружение."),
    ("rus", "Где находится ближайшая больница или аптека?"),
    # Ukrainian (ukr)
    ("ukr", "Мені потрібно приймати ці ліки двічі на день після їжі."),
    ("ukr", "Підкажіть, будь ласка, як дістатися до залізничного вокзалу?"),
    ("ukr", "Не сідайте за кермо, якщо ці ліки викликають запаморочення."),
    ("ukr", "Де знаходиться найближча лікарня чи аптека?"),
    # Korean (kor)
    ("kor", "이 약은 식후 하루에 두 번 복용해야 합니다."),
    ("kor", "기차역으로 가는 길을 알려주시겠어요?"),
    ("kor", "이 약을 먹고 어지러우면 운전하지 마세요."),
    ("kor", "가장 가까운 병원이나 약국이 어디인가요?"),
    # Mandarin (cmn)
    ("cmn", "这种药我需要每天饭后服用两次。"),
    ("cmn", "请问去火车站怎么走？"),
    ("cmn", "如果这种药让您感到头晕，请不要开车。"),
    ("cmn", "最近的医院或药店在哪里？"),
    # Cantonese (yue)
    ("yue", "呢隻藥我要每日飯後食兩次。"),
    ("yue", "唔該，請問去火車站點行呀？"),
    ("yue", "如果食完呢隻藥覺得頭暈，就唔好揸車喇。"),
    ("yue", "最近嘅醫院或者藥房喺邊度呀？"),
    # Māori (mri)
    ("mri", "Me kai e rua ngā pire i ia rā i muri i te kai."),
    ("mri", "Kōrero mai me pēhea te haere ki te teihana tereina."),
    ("mri", "Kaua e taraiwa mēnā ka pōrori tō māhunga i tēnei rongoā."),
    ("mri", "Kei hea te hōhipera, te whare rongoā rānei e tata ana?"),
    # Swedish (swe)
    ("swe", "Jag måste ta den här medicinen två gånger om dagen efter maten."),
    ("swe", "Kan du berätta hur man kommer till tågstationen, tack?"),
    ("swe", "Kör inte bil om det här läkemedlet gör dig yr."),
    ("swe", "Var ligger närmaste sjukhus eller apotek?"),
    # Finnish (fin)
    ("fin", "Minun täytyy ottaa tätä lääkettä kahdesti päivässä ruokailun jälkeen."),
    ("fin", "Voisitko kertoa minulle, miten pääsen rautatieasemalle?"),
    ("fin", "Älä aja autoa, jos tämä lääke aiheuttaa huimausta."),
    ("fin", "Missä on lähin sairaala tai apteekki?"),
    # Norwegian (nob)
    ("nob", "Jeg må ta denne medisinen to ganger om dagen etter måltid."),
    ("nob", "Kan du fortelle meg veien til togstasjonen, vær så snill?"),
    ("nob", "Ikke kjør bil hvis denne medisinen gjør deg svimmel."),
    ("nob", "Hvor er nærmeste sykehus eller apotek?"),
]

class LanguageIDModel:
    def __init__(self):
        # Combine word-boundary char n-grams + word n-grams to separate look-alike languages (e.g. ind vs zsm)
        self.vectorizer = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ])
        self.clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")
        self.labels = []
        self.metrics = {}
        if not os.path.exists("sentences.csv"):
            self._download_tatoeba()
        self.train_and_evaluate("sentences.csv" if os.path.exists("sentences.csv") else None)

    def _download_tatoeba(self):
        import urllib.request
        import tarfile
        url = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
        archive = "sentences.tar.bz2"
        try:
            if not os.path.exists(archive):
                print("[Tatoeba] Downloading sentences.tar.bz2 (~150 MB)...")
                urllib.request.urlretrieve(url, archive)
            print("[Tatoeba] Extracting sentences.csv...")
            with tarfile.open(archive, "r:bz2") as tar:
                tar.extractall(".")
            print("[Tatoeba] sentences.csv ready.")
        except Exception as e:
            print(f"[Tatoeba] Download skipped ({e}); using built-in starter corpus.")

    def train_and_evaluate(self, tsv_path: Optional[str] = None):
        if tsv_path and os.path.exists(tsv_path):
            df = pd.read_csv(tsv_path, sep="\t", names=["id", "lang", "text"], on_bad_lines="skip")
            target_langs = ["eng", "fra", "spa", "deu", "swe", "fin", "nob", "rus", "ukr", "kor", "cmn", "yue", "swh", "yor", "ind", "zsm", "jpn", "mri"]
            df = df[df["lang"].isin(target_langs)].dropna()
            df = pd.concat([g.sample(min(len(g), 2000), random_state=42) for _, g in df.groupby("lang")], ignore_index=True)
        else:
            df = pd.DataFrame(TATOEBA_STARTER_DATA, columns=["lang", "text"])

        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["lang"], test_size=0.25, random_state=42, stratify=df["lang"]
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.clf.fit(X_train_vec, y_train)
        self.labels = list(self.clf.classes_)

        # Evaluate our TF-IDF + LogReg model
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.clf.predict(X_test_vec)
        overall_acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=self.labels).tolist()

        # Benchmark against langdetect
        iso3_to_iso2 = {"eng": "en", "fra": "fr", "spa": "es", "deu": "de", "swe": "sv", "fin": "fi", "nob": "no", "rus": "ru", "ukr": "uk", "kor": "ko", "cmn": "zh-cn", "yue": "zh-tw", "swh": "sw", "yor": "yo", "ind": "id", "zsm": "ms", "jpn": "ja", "mri": "mi"}
        ld_preds = []
        for txt in X_test:
            try:
                ld_iso2 = detect(txt)
                inv = {v: k for k, v in iso3_to_iso2.items()}
                ld_preds.append(inv.get(ld_iso2, "other"))
            except LangDetectException:
                ld_preds.append("other")

        per_language = []
        y_test_list = list(y_test)
        for lang_code in self.labels:
            idxs = [i for i, y in enumerate(y_test_list) if y == lang_code]
            our_acc = sum(1 for i in idxs if y_pred[i] == lang_code) / max(len(idxs), 1)
            ld_acc = sum(1 for i in idxs if ld_preds[i] == lang_code) / max(len(idxs), 1)
            per_language.append({
                "code": lang_code,
                "n": len(idxs),
                "tfidf_accuracy": round(our_acc * 100, 1),
                "langdetect_accuracy": round(ld_acc * 100, 1),
            })

        self.metrics = {
            "overall_tfidf_accuracy": round(overall_acc * 100, 1),
            "labels": self.labels,
            "confusion_matrix": cm,
            "per_language": per_language,
            "dataset_size": len(df),
            "dataset_source": "Tatoeba TSV" if tsv_path else "Tatoeba-aligned starter corpus (load full Tatoeba sentences.csv in Colab for 2,000/lang)",
        }

    def predict(self, text: str) -> Dict:
        vec = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(vec)[0]
        ranked = sorted(zip(self.labels, probs), key=lambda x: x[1], reverse=True)
        top_code, top_prob = ranked[0]

        # Also run langdetect for comparison
        try:
            ld_res = [{"lang": str(r.lang), "prob": round(float(r.prob) * 100, 1)} for r in detect_langs(text)[:3]]
        except LangDetectException:
            ld_res = [{"lang": "unknown", "prob": 0.0}]

        # Check confusable languages (e.g. ind vs zsm)
        iso3_to_nllb = {"eng": "eng_Latn", "fra": "fra_Latn", "spa": "spa_Latn", "deu": "deu_Latn", "swe": "swe_Latn", "fin": "fin_Latn", "nob": "nob_Latn", "rus": "rus_Cyrl", "ukr": "ukr_Cyrl", "kor": "kor_Hang", "cmn": "zho_Hans", "yue": "yue_Hant", "swh": "swh_Latn", "yor": "yor_Latn", "ind": "ind_Latn", "zsm": "zsm_Latn", "jpn": "jpn_Jpan", "mri": "mri_Latn", "smo": "smo_Latn", "vie": "vie_Latn", "tgl": "tgl_Latn"}
        nllb_code = iso3_to_nllb.get(top_code, "eng_Latn")
        confusion_note = None
        if len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) < 0.25:
            confusion_note = f"Close match between {ranked[0][0]} ({ranked[0][1]*100:.1f}%) and {ranked[1][0]} ({ranked[1][1]*100:.1f}%)."
        if nllb_code in CONFUSABLE_PAIRS:
            partner, note = CONFUSABLE_PAIRS[nllb_code]
            confusion_note = (confusion_note + " " if confusion_note else "") + note

        return {
            "predicted_iso3": top_code,
            "predicted_nllb": nllb_code,
            "confidence": round(float(top_prob) * 100, 1),
            "top_candidates": [{"code": c, "confidence": round(float(p) * 100, 1)} for c, p in ranked[:3]],
            "langdetect_comparison": ld_res,
            "confusion_note": confusion_note,
        }


# ==============================================================================
# 4. 4.4.2 TRANSLATION ENGINE (facebook/nllb-200-distilled-600M + FLORES+ chrF)
# ==============================================================================
def classify_alternative_nuance(primary: str, alt: str, tgt_lang: str) -> str:
    p_low, a_low = primary.lower(), alt.lower()
    if tgt_lang == "jpn_Jpan":
        keigo = ["ございます", "いらっしゃ", "いたし", "申し", "くださいます", "召し上が", "ご覧", "伺い"]
        polite = ["です", "ます", "ません", "でした", "ました", "ください", "でしょう", "お願い", "すみません", "ありがとう"]
        casual = ["だ", "だよ", "だね", "かな", "じゃ", "どうも", "やあ", "バイバイ", "サンキュ"]
        p_score = sum(2 for k in keigo if k in primary) + sum(1 for k in polite if k in primary) - sum(1 for k in casual if k in primary)
        a_score = sum(2 for k in keigo if k in alt) + sum(1 for k in polite if k in alt) - sum(1 for k in casual if k in alt)
        if any(k in alt for k in keigo) and not any(k in primary for k in keigo):
            return "Honorific (Keigo)"
        if a_score > p_score:
            return "More Polite"
        if a_score < p_score:
            return "More Casual"
    elif tgt_lang == "fra_Latn":
        if ("vous" in a_low or "veuillez" in a_low) and "vous" not in p_low:
            return "More Polite (vous)"
        if ("tu " in a_low or "salut" in a_low) and "tu " not in p_low:
            return "More Casual (tu)"
    elif tgt_lang == "spa_Latn":
        if ("usted" in a_low or "por favor" in a_low) and "usted" not in p_low:
            return "More Polite (usted)"
        if ("tú" in a_low or "hola" in a_low) and "tú" not in p_low:
            return "More Casual (tú)"
    elif tgt_lang in ("ind_Latn", "zsm_Latn"):
        if any(w in a_low for w in ["anda", "mohon", "silakan", "sila"]) and not any(w in p_low for w in ["anda", "mohon", "silakan", "sila"]):
            return "More Polite"
        if any(w in a_low for w in ["kamu", "aku", "jom", "bareng"]) and not any(w in p_low for w in ["kamu", "aku", "jom", "bareng"]):
            return "More Casual"

    # Structural / grammar / punctuation fallback
    if re.sub(r"[?!.。？！\s]", "", primary) == re.sub(r"[?!.。？！\s]", "", alt):
        return "Different Punctuation / Tone"
    if abs(len(alt.split()) - len(primary.split())) >= 2 or abs(len(alt) - len(primary)) >= 6:
        return "Differing Grammar / Phrasing"
    return "Alternative Wording"

class NLLBTranslator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[NLLB-200] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()
        print("[NLLB-200] Model ready.")

    def translate(self, text: str, src_lang: str, tgt_lang: str, num_beams: int = 4, num_alternatives: int = 3) -> Dict:
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=512,
                num_beams=max(num_beams, num_alternatives),
                num_return_sequences=num_alternatives,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=False,
            )

        decoded = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        # Convert sequence log-likelihood score to an interpretable 0-100 confidence proxy
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            raw_scores = outputs.sequences_scores.cpu().numpy().tolist()
            probs = [round(float(np.exp(s)) * 100, 1) for s in raw_scores]
            probs = [min(99.5, max(5.0, p)) for p in probs]
        else:
            probs = [85.0 - i * 8.0 for i in range(len(decoded))]

        return {
            "primary_translation": decoded[0],
            "confidence": probs[0],
            "alternatives": [{"text": d, "confidence": p, "label": classify_alternative_nuance(decoded[0], d, tgt_lang)} for d, p in zip(decoded[1:], probs[1:])],
            "token_count_in": int(inputs["input_ids"].shape[1]),
            "token_count_out": int(outputs.sequences.shape[1]),
        }


# FLORES+ style parallel benchmark sentences (Widely served vs. Less widely served)
FLORES_BENCHMARK_SAMPLES = [
    {
        "eng_Latn": "Take two tablets by mouth every eight hours with a full glass of water.",
        "fra_Latn": "Prenez deux comprimés par voie orale toutes les huit heures avec un grand verre d'eau.",
        "deu_Latn": "Nehmen Sie alle acht Stunden zwei Tabletten mit einem vollen Glas Wasser ein.",
        "swe_Latn": "Ta två tabletter genom munnen var åttonde timme med ett fullt glas vatten.",
        "fin_Latn": "Ota kaksi tablettia suun kautta kahdeksan tunnin välein täyden vesilasillisen kanssa.",
        "nob_Latn": "Ta to tabletter gjennom munnen hver åttende time med et fullt glass vann.",
        "spa_Latn": "Tome dos tabletas por vía oral cada ocho horas con un vaso lleno de agua.",
        "rus_Cyrl": "Принимайте по две таблетки внутрь каждые восемь часов, запивая полным стаканом воды.",
        "ukr_Cyrl": "Приймайте по дві таблетки перорально кожні вісім годин, запиваючи повною склянкою води.",
        "jpn_Jpan": "8時間ごとにコップ1杯の水で2錠を服用してください。",
        "kor_Hang": "8시간마다 물 한 잔과 함께 두 알을 복용하세요.",
        "zho_Hans": "每八小时用一整杯水口服两片。",
        "swh_Latn": "Meza vidonge viwili kwa mdomo kila baada ya saa nane na glasi kamili ya maji.",
        "yor_Latn": "Mu tabulẹti meji ni ẹnu ni gbogbo wakati mẹjọ pẹlu gilasi omi kikun.",
        "domain": "Medication",
    },
    {
        "eng_Latn": "Turn left at the next traffic light and continue straight for two kilometers.",
        "fra_Latn": "Tournez à gauche au prochain feu et continuez tout droit pendant deux kilomètres.",
        "deu_Latn": "Biegen Sie an der nächsten Ampel links ab und fahren Sie zwei Kilometer geradeaus.",
        "swe_Latn": "Sväng vänster vid nästa trafikljus och fortsätt rakt fram i två kilometer.",
        "fin_Latn": "Käänny seuraavista liikennevaloista vasemmalle ja jatka suoraan kaksi kilometriä.",
        "nob_Latn": "Sving til venstre ved neste trafikklys og fortsett rett frem i to kilometer.",
        "spa_Latn": "Gire a la izquierda en el próximo semáforo y continúe recto durante dos kilómetros.",
        "rus_Cyrl": "Поверните налево на следующем светофоре и продолжайте движение прямо два километра.",
        "ukr_Cyrl": "Поверніть ліворуч на наступному світлофорі та їдьте прямо два кілометри.",
        "jpn_Jpan": "次の信号を左折し、2キロ直進してください。",
        "kor_Hang": "다음 신호등에서 좌회전한 후 2킬로미터 직진하세요.",
        "zho_Hans": "在下一个红绿灯左转，然后直行两公里。",
        "swh_Latn": "Geuka kushoto kwenye taa inayofuata ya barabarani na uendelee moja kwa moja kwa kilomita mbili.",
        "yor_Latn": "Yipada si osi ni ina opopona ti o tẹle ki o tẹsiwaju taara fun kilomita meji.",
        "domain": "Driving / Directions",
    },
    {
        "eng_Latn": "Please bring a valid photo identification card to the school office before noon.",
        "fra_Latn": "Veuillez apporter une pièce d'identité avec photo valide au secrétariat de l'école avant midi.",
        "deu_Latn": "Bitte bringen Sie vor Mittag einen gültigen Lichtbildausweis zum Schulsekretariat.",
        "swe_Latn": "Ta med en giltig fotolegitimation till skolans expedition före klockan tolv.",
        "fin_Latn": "Tuo voimassa oleva kuvallinen henkilötodistus koulun toimistoon ennen puoltapäivää.",
        "nob_Latn": "Vennligst ta med gyldig legitimasjon med bilde til skolens kontor før klokken tolv.",
        "spa_Latn": "Por favor traiga una identificación con foto válida a la oficina de la escuela antes del mediodía.",
        "rus_Cyrl": "Пожалуйста, принесите действительное удостоверение личности с фотографией в школьный офис до полудня.",
        "ukr_Cyrl": "Будь ласка, принесіть дійсне посвідчення особи з фотографією до шкільного офісу до полудня.",
        "jpn_Jpan": "正午までに有効な顔写真付き身分証明書を学校の事務室へお持ちください。",
        "kor_Hang": "정오 전까지 유효한 사진이 부착된 신분증을 학교 행정실로 가져오세요.",
        "zho_Hans": "请在中午之前携带有效的带照片身份证件到学校办公室。",
        "swh_Latn": "Tafadhali leta kitambulisho halali chenye picha katika ofisi ya shule kabla ya adhuhuri.",
        "yor_Latn": "Jọwọ mu kaadi idanimọ ti o ni fọto ti o wulo wa si ọfiisi ile-iwe ṣaaju ọsan.",
        "domain": "School Notice",
    },
    {
        "eng_Latn": "Do not drive or operate heavy machinery if this medicine makes you feel sleepy.",
        "fra_Latn": "Ne conduisez pas et n'utilisez pas de machines lourdes si ce médicament vous rend somnolent.",
        "deu_Latn": "Fahren Sie kein Auto und bedienen Sie keine schweren Maschinen, wenn dieses Medikament Sie schläfrig macht.",
        "swe_Latn": "Kör inte bil och använd inte tunga maskiner om den här medicinen gör dig sömnig.",
        "fin_Latn": "Älä aja autoa tai käytä raskaita koneita, jos tämä lääke tekee sinut uneliaaksi.",
        "nob_Latn": "Ikke kjør bil eller bruk tunge maskiner hvis denne medisinen gjør deg søvnig.",
        "spa_Latn": "No conduzca ni opere maquinaria pesada si este medicamento le produce sueño.",
        "rus_Cyrl": "Не садитесь за руль и не работайте со сложными механизмами, если это лекарство вызывает сонливость.",
        "ukr_Cyrl": "Не керуйте автомобілем та важкою технікою, якщо ці ліки викликають сонливість.",
        "jpn_Jpan": "この薬で眠気を感じた場合は、運転や重機の操作をしないでください。",
        "kor_Hang": "이 약을 복용하고 졸음이 오면 운전이나 중장비 조작을 하지 마세요.",
        "zho_Hans": "如果这种药让您感到困倦，请勿驾驶或操作重型机械。",
        "swh_Latn": "Usiendeshe gari wala kuendesha mitambo mizito ikiwa dawa hii inakufanya uhisi usingizi.",
        "yor_Latn": "Maṣe wa ọkọ tabi ṣiṣẹ ẹrọ ti o wuwo ti oogun yii ba n mu oorun kun ọ.",
        "domain": "Safety Warning",
    },
    {
        "eng_Latn": "Breaking a leg before a performance is actually a wish for good luck in theater.",
        "fra_Latn": "Souhaiter à quelqu'un de se casser une jambe avant un spectacle est en fait un vœu de bonne chance au théâtre.",
        "deu_Latn": "Jemandem vor einer Aufführung Hals- und Beinbruch zu wünschen, ist im Theater eigentlich ein Glückwunsch.",
        "swe_Latn": "Att önska någon ett brutet ben före en föreställning är faktiskt en lyckönskning inom teatern.",
        "fin_Latn": "Jalan katkeamisen toivottaminen ennen esitystä on teatterissa oikeastaan onnentoivotus.",
        "nob_Latn": "Å ønske noen å brekke et ben før en forestilling er faktisk et lykkeønske i teateret.",
        "spa_Latn": "Desearle a alguien que se rompa una pierna antes de una función es en realidad un deseo de buena suerte en el teatro.",
        "rus_Cyrl": "Пожелать сломать ногу перед выступлением — это на самом деле театральное пожелание удачи.",
        "ukr_Cyrl": "Побажати зламати ногу перед виступом — це насправді театральне побажання удачі.",
        "jpn_Jpan": "公演前に「足を折れ」と言うのは、実は演劇界では幸運を祈る言葉です。",
        "kor_Hang": "공연 전에 다리가 부러지라고 말하는 것은 사실 연극계에서 행운을 비는 표현입니다.",
        "zho_Hans": "在演出前祝人“断条腿”在戏剧界其实是祝你好运的意思。",
        "swh_Latn": "Kumtakia mtu kuvunjika mguu kabla ya onyesho kwa kweli ni kumtakia bahati nzuri katika ukumbi wa michezo.",
        "yor_Latn": "Fifẹ ki ẹnikan fọ ẹsẹ ṣaaju ere jẹ ifẹ oriire ni itage.",
        "domain": "Idiom",
    },
]

# ==============================================================================
# 5. SAFETY, IDIOM, REGISTER & EXPLANATION ANALYZER
# ==============================================================================
SAFETY_RULES = [
    ("Medication / Health", ["medicine", "medication", "tablet", "pill", "dose", "dosage", "mg", "ml", "allergy", "allergic", "symptom", "hospital", "pharmacy", "doctor", "prescription", "dizzy", "overdose", "ubat", "obat", "dawa", "médicament", "medicina", "薬"],
     "HIGH RISK (Medical): Dosage, frequency, or allergy mistranslations can cause harm. Always confirm with a pharmacist or clinician."),
    ("Driving / Navigation", ["drive", "driving", "turn left", "turn right", "highway", "exit", "speed", "traffic", "brake", "kilometer", "km", "miles", "intersection", "conduisez", "conduzca", "運転"],
     "HIGH RISK (Driving/Safety): Never rely solely on real-time MT/TTS while operating a vehicle."),
    ("Legal / Emergency", ["emergency", "police", "ambulance", "fire", "contract", "sign here", "consent", "visa", "court", "law", "evacuate", "danger", "poison", "warning"],
     "CAUTION (Legal/Emergency): Verify official forms and emergency instructions with a qualified human speaker."),
]

IDIOMS_DB = {
    "raining cats and dogs": "English idiom meaning 'raining very heavily' (literal translation will confuse listeners).",
    "piece of cake": "English idiom meaning 'very easy'.",
    "break a leg": "Theater idiom meaning 'good luck' (literal translation sounds like physical harm!).",
    "il pleut des cordes": "French idiom meaning 'raining heavily' (literally 'raining ropes').",
    "pan comido": "Spanish idiom meaning 'very easy' (literally 'eaten bread').",
    "朝飯前": "Japanese idiom (asameshimae) meaning 'a piece of cake / effortless' (literally 'before breakfast').",
}

SLANG_MARKERS = ["wanna", "gonna", "what's up", "hey", "cool", "bro", "lol", "omg", "jom", "bareng", "salut", "ça va", "qué tal", "ねえ", "mambo vipi"]
FORMAL_MARKERS = ["please", "could you", "kindly", "submit", "completed form", "veuillez", "pourriez-vous", "estimado", "por favor", "tafadhali", "sila", "harapkan", "いただけますか", "ください"]

def analyze_pragmatics_and_safety(text: str, translation: str, src_lang: str, tgt_lang: str) -> Dict:
    lower = text.lower()
    safety_flags = []
    for cat, kws, msg in SAFETY_RULES:
        matched = [k for k in kws if k in lower]
        if matched:
            safety_flags.append({"category": cat, "matched_terms": matched, "warning": msg})

    idiom_notes = []
    for idiom, meaning in IDIOMS_DB.items():
        if idiom in lower:
            idiom_notes.append({"idiom": idiom, "explanation": meaning})

    slang_hits = [m for m in SLANG_MARKERS if m in lower]
    formal_hits = [m for m in FORMAL_MARKERS if m in lower]
    if slang_hits and not formal_hits:
        register = "Casual / Slang"
    elif formal_hits and not slang_hits:
        register = "Formal / Polite"
    elif slang_hits and formal_hits:
        register = "Mixed Register"
    else:
        register = "Neutral / Standard"

    # Grammatical & Structural Explanation
    src_name = LANGUAGES.get(src_lang, {}).get("name", src_lang)
    tgt_name = LANGUAGES.get(tgt_lang, {}).get("name", tgt_lang)
    grammar_notes = [
        f"Translated from {src_name} ({src_lang}) to {tgt_name} ({tgt_lang}) using NLLB-200 encoder-decoder attention with beam search.",
        f"Detected source register: {register}.",
    ]
    if tgt_lang == "jpn_Jpan":
        grammar_notes.append("Target language (Japanese) uses Subject-Object-Verb (SOV) order and politeness levels (desu/masu vs. plain).")
    elif tgt_lang == "arb_Arab":
        grammar_notes.append("Target language (Arabic) is written Right-to-Left (RTL) and uses Verb-Subject-Object or Subject-Verb-Object depending on emphasis.")
    elif tgt_lang in ("swh_Latn", "yor_Latn"):
        grammar_notes.append(f"Target language ({tgt_name}) is a less-widely-served language in standard commercial benchmarks; verify idiomatic phrasing with a native speaker.")

    return {
        "register": register,
        "safety_flags": safety_flags,
        "idiom_notes": idiom_notes,
        "grammar_notes": grammar_notes,
    }


# ==============================================================================
# 5B. COGNITIVE ACCESSIBILITY (EASY-READ) & SAFE LANGUAGE MODE
# ==============================================================================
SIMPLIFY_REPLACEMENTS = [
    (r"\bthe implementation of\b", "doing"),
    (r"\baforementioned\b", "this"),
    (r"\bis contingent upon\b", "depends on"),
    (r"\bcontingent upon\b", "depends on"),
    (r"\bthe availability of\b", "having"),
    (r"\brequisite\b", "needed"),
    (r"\bcommence\b", "start"),
    (r"\butilize\b", "use"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequently\b", "then"),
    (r"\bin the event that\b", "if"),
    (r"\badminister\b", "take or give"),
    (r"\bdiscontinue\b", "stop"),
    (r"\bconsume\b", "eat or drink"),
    (r"\bapproximately\b", "about"),
    (r"\bassistance\b", "help"),
    (r"\bmedication\b", "medicine"),
    (r"\bphysician\b", "doctor"),
    (r"\bvertigo\b", "dizziness"),
]

DIFFICULT_WORDS_GLOSSARY = {
    "contingent": "Depends on something else happening first (e.g., 'Outdoor recess is contingent upon good weather').",
    "aforementioned": "Already mentioned earlier in the text.",
    "requisite": "Required or needed for a task.",
    "dosage": "The exact amount of medicine to take at one time.",
    "vertigo": "A spinning or dizzy feeling.",
    "allergy": "A body reaction to something like food or medicine.",
    "prescription": "A doctor's written order for medicine.",
    "flammable": "Catches fire very easily.",
}

PROFANITY_LIST = {
    "fuck": ("f***", "very bad", "strong profanity"),
    "fucking": ("f******", "extremely", "strong profanity"),
    "shit": ("s***", "terrible", "profanity"),
    "bitch": ("b****", "unkind person", "insult"),
    "asshole": ("a******", "rude person", "insult"),
    "bastard": ("b******", "unpleasant person", "insult"),
    "damn": ("d***", "very", "mild profanity"),
    "crap": ("c***", "nonsense", "mild profanity"),
    "idiot": ("i****", "unhelpful", "insult"),
    "stupid": ("s*****", "unwise", "insult"),
    "putain": ("p*****", "oh no", "French profanity"),
    "merde": ("m****", "terrible", "French profanity"),
    "mierda": ("m*****", "terrible", "Spanish profanity"),
    "くそ": ("く*", "terrible", "Japanese profanity"),
    "ばか": ("ば*", "silly", "Japanese insult"),
}

def simplify_text_easy_read(text: str, reading_level: str) -> Dict:
    simplified = text
    for pat, rep in SIMPLIFY_REPLACEMENTS:
        simplified = re.sub(pat, rep, simplified, flags=re.IGNORECASE)
    # Clean up double spaces
    simplified = re.sub(r"\s+", " ", simplified).strip()

    # Split into smaller sentences / step-by-step if requested
    raw_chunks = [c.strip() for c in re.split(r"(?<=[.!?。！？])\s+|,\s+(?=(?:then|and then|before|after|do not|if)\b)", simplified) if c.strip()]
    if reading_level == "steps" and len(raw_chunks) > 1:
        simplified_out = "\n".join(f"Step {i+1}: {c}" for i, c in enumerate(raw_chunks))
    elif reading_level == "simple":
        simplified_out = " ".join(raw_chunks)
    else:
        simplified_out = text

    # Extract key information (numbers, conditions, warnings) so critical info is never lost
    key_info = []
    nums = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|ml|tablets?|pills?|hours?|times?|km|degrees?)?\b", text, flags=re.IGNORECASE)
    if nums:
        key_info.append("Numbers / Quantities: " + ", ".join(dict.fromkeys(n.strip() for n in nums if n.strip())))
    conds = re.findall(r"\b(?:if|unless|before|after|do not|never|only|until|every)\b[^,.!?]*", text, flags=re.IGNORECASE)
    if conds:
        key_info.append("Critical Conditions: " + " | ".join(c.strip() for c in conds[:3]))

    # Glossary matches
    lower = text.lower()
    glossary = [{"word": w, "definition": d} for w, d in DIFFICULT_WORDS_GLOSSARY.items() if w in lower]

    return {
        "simplified_source": simplified_out,
        "chunks": raw_chunks,
        "key_info": key_info,
        "glossary": glossary,
    }

def apply_safe_language_mode(text: str, mode: str) -> Dict:
    found = []
    out = text
    for bad, (censored, mild, label) in PROFANITY_LIST.items():
        if re.search(rf"\b{re.escape(bad)}\b", out, flags=re.IGNORECASE) or (not bad.isascii() and bad in out):
            found.append(f"{bad} ({label})")
            if mode == "censor":
                out = re.sub(rf"\b{re.escape(bad)}\b", censored, out, flags=re.IGNORECASE) if bad.isascii() else out.replace(bad, censored)
            elif mode in ("replace", "child_friendly"):
                out = re.sub(rf"\b{re.escape(bad)}\b", mild, out, flags=re.IGNORECASE) if bad.isascii() else out.replace(bad, mild)
    return {
        "filtered_text": out,
        "profanity_detected": found,
        "mode_used": mode,
    }

# ==============================================================================
# 6. INITIALIZE MODELS & FASTAPI APP
# ==============================================================================
import gc
lid_model = LanguageIDModel()
translator = None
whisper_model = None
_translator_lock = threading.Lock()
_whisper_lock = threading.Lock()

def get_translator():
    global translator
    if translator is None:
        with _translator_lock:
            if translator is None:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                translator = NLLBTranslator()
    return translator

def get_whisper():
    global whisper_model
    if whisper_model is None:
        with _whisper_lock:
            if whisper_model is None:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[Whisper] Loading openai/whisper-small on {device}...")
                whisper_model = whisper.load_model("small", device=device)
                print("[Whisper] Ready.")
    return whisper_model

app = FastAPI(title="Veritas AI 4.4: PolyComs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "auto"
    tgt_lang: str = "swh_Latn"
    tone_hint: str = "auto"
    reading_level: str = "standard"
    safe_mode: str = "preserve"

class LidRequest(BaseModel):
    text: str

class ChrfCheckRequest(BaseModel):
    hypothesis: str
    reference: str

@app.get("/", response_class=HTMLResponse)
def serve_home():
    return HTMLResponse(content=EMBEDDED_HTML)

@app.get("/api/status")
def api_status():
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "nllb_loaded": translator is not None,
        "whisper_loaded": whisper_model is not None,
        "lid_ready": True,
        "languages": LANGUAGES,
    }

@app.post("/api/translate")
def api_translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "Input text is empty."}, status_code=400)

    lid_info = lid_model.predict(text)
    src_lang = req.src_lang
    auto_detected = False
    if src_lang == "auto":
        src_lang = lid_info["predicted_nllb"]
        auto_detected = True

    cog = simplify_text_easy_read(text, req.reading_level)
    text_to_translate = cog["simplified_source"] if req.reading_level in ("simple", "steps") else text

    safe_in = apply_safe_language_mode(text_to_translate, req.safe_mode)
    trans_res = get_translator().translate(safe_in["filtered_text"], src_lang, req.tgt_lang)

    safe_out = apply_safe_language_mode(trans_res["primary_translation"], req.safe_mode)
    trans_res["primary_translation"] = safe_out["filtered_text"]
    for alt in trans_res["alternatives"]:
        alt["text"] = apply_safe_language_mode(alt["text"], req.safe_mode)["filtered_text"]

    analysis = analyze_pragmatics_and_safety(text, trans_res["primary_translation"], src_lang, req.tgt_lang)
    analysis["cognitive"] = cog
    analysis["safe_language"] = {
        "mode": req.safe_mode,
        "profanity_detected": list(dict.fromkeys(safe_in["profanity_detected"] + safe_out["profanity_detected"])),
    }
    if req.safe_mode == "child_friendly" and analysis["safe_language"]["profanity_detected"]:
        analysis["grammar_notes"].append("Child-Friendly Mode: Strong language was softened without hiding the speaker's core intent.")

    return {
        "source_text": text,
        "src_lang": src_lang,
        "tgt_lang": req.tgt_lang,
        "auto_detected": auto_detected,
        "lid": lid_info,
        "translation": trans_res,
        "analysis": analysis,
        "target_rtl": LANGUAGES.get(req.tgt_lang, {}).get("rtl", False),
    }

@app.post("/api/lid/predict")
def api_lid_predict(req: LidRequest):
    if not req.text.strip():
        return JSONResponse({"error": "Text is empty."}, status_code=400)
    return lid_model.predict(req.text.strip())

@app.get("/api/lid/metrics")
def api_lid_metrics():
    return lid_model.metrics

@app.get("/api/quality/benchmark")
def api_quality_benchmark(langs: str = "fra_Latn,swh_Latn,yor_Latn"):
    tr = get_translator()
    requested = [l.strip() for l in langs.split(",") if l.strip()]
    available = [k for k in FLORES_BENCHMARK_SAMPLES[0].keys() if k not in ("eng_Latn", "domain")]
    targets = [l for l in requested if l in available] or ["fra_Latn", "swh_Latn", "yor_Latn"]
    results_by_lang = {}
    examples = []

    for tgt in targets:
        hyps, refs = [], []
        for idx, item in enumerate(FLORES_BENCHMARK_SAMPLES):
            res = tr.translate(item["eng_Latn"], "eng_Latn", tgt, num_beams=2, num_alternatives=1)
            hyp = res["primary_translation"]
            ref = item[tgt]
            hyps.append(hyp)
            refs.append(ref)
            sent_chrf = sacrebleu.sentence_chrf(hyp, [ref]).score
            examples.append({
                "domain": item["domain"],
                "source_en": item["eng_Latn"],
                "target_lang": LANGUAGES[tgt]["name"],
                "target_code": tgt,
                "model_output": hyp,
                "reference": ref,
                "chrf": round(sent_chrf, 2),
            })
        corpus_chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        results_by_lang[tgt] = {
            "name": LANGUAGES[tgt]["name"],
            "tier": LANGUAGES[tgt]["tier"],
            "corpus_chrf": round(corpus_chrf, 2),
        }

    # Sort examples lowest chrF first so worst errors are highlighted at the top
    examples.sort(key=lambda x: x["chrf"])
    return {"summary": results_by_lang, "examples": examples}

@app.post("/api/quality/chrf")
def api_quality_chrf(req: ChrfCheckRequest):
    score = sacrebleu.sentence_chrf(req.hypothesis.strip(), [req.reference.strip()]).score
    return {"chrf": round(score, 2)}

@app.post("/api/stt")
async def api_stt(audio: UploadFile = File(...), reference_text: str = Form(""), spoken_lang: str = Form("auto")):
    wm = get_whisper()
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        import shutil
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            import imageio_ffmpeg  # type: ignore[import-not-found]
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_bin, "-nostdin", "-threads", "0", "-i", tmp_path, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-"]
        pcm = subprocess.run(cmd, capture_output=True, check=True).stdout
        audio_np = np.frombuffer(pcm, np.int16).flatten().astype(np.float32) / 32768.0
        transcribe_kwargs = {"task": "transcribe"}
        if spoken_lang and spoken_lang != "auto":
            transcribe_kwargs["language"] = spoken_lang
        result = wm.transcribe(audio_np, **transcribe_kwargs)
        transcript = result.get("text", "").strip()
        detected_iso2 = result.get("language", "unknown")
        wer_val = None
        if reference_text.strip():
            norm_ref = re.sub(r"[^\w\s]", "", reference_text.lower()).strip()
            norm_hyp = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
            if norm_ref:
                wer_val = round(compute_wer(norm_ref, norm_hyp) * 100, 1)
        return {
            "transcript": transcript,
            "whisper_language": detected_iso2,
            "lid": lid_model.predict(transcript) if transcript else None,
            "wer_percent": wer_val,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


EDGE_TTS_VOICES = {
    "eng_Latn": "en-US-JennyNeural",
    "fra_Latn": "fr-FR-DeniseNeural",
    "spa_Latn": "es-ES-ElviraNeural",
    "deu_Latn": "de-DE-KatjaNeural",
    "swe_Latn": "sv-SE-SofieNeural",
    "fin_Latn": "fi-FI-NooraNeural",
    "nob_Latn": "nb-NO-PernilleNeural",
    "por_Latn": "pt-PT-RaquelNeural",
    "rus_Cyrl": "ru-RU-SvetlanaNeural",
    "ukr_Cyrl": "uk-UA-PolinaNeural",
    "jpn_Jpan": "ja-JP-NanamiNeural",
    "kor_Hang": "ko-KR-SunHiNeural",
    "zho_Hans": "zh-CN-XiaoxiaoNeural",
    "yue_Hant": "zh-HK-HiuGaaiNeural",
    "vie_Latn": "vi-VN-HoaiMyNeural",
    "tgl_Latn": "fil-PH-BlessicaNeural",
    "ind_Latn": "id-ID-GadisNeural",
    "zsm_Latn": "ms-MY-YasminNeural",
    "swh_Latn": "sw-KE-ZuriNeural",
    "arb_Arab": "ar-SA-ZariyahNeural",
    "hin_Deva": "hi-IN-SwaraNeural",
}

mms_tts_yor = None

class TtsRequest(BaseModel):
    text: str
    lang: str = "eng_Latn"

GTTS_LANGS = {
    "eng_Latn": "en", "fra_Latn": "fr", "spa_Latn": "es", "deu_Latn": "de", "swe_Latn": "sv", "fin_Latn": "fi", "nob_Latn": "no", "por_Latn": "pt",
    "rus_Cyrl": "ru", "ukr_Cyrl": "uk", "jpn_Jpan": "ja", "kor_Hang": "ko",
    "zho_Hans": "zh-CN", "yue_Hant": "zh-TW", "vie_Latn": "vi", "tgl_Latn": "tl",
    "ind_Latn": "id", "zsm_Latn": "ms", "swh_Latn": "sw",
    "arb_Arab": "ar", "hin_Deva": "hi",
}


_ocr_readers = {}
_ocr_lock = threading.Lock()

OCR_LANG_MAP = {
    "auto": ["en"],
    "eng_Latn": ["en"],
    "fra_Latn": ["fr", "en"],
    "spa_Latn": ["es", "en"],
    "deu_Latn": ["de", "en"],
    "swe_Latn": ["sv", "en"],
    "fin_Latn": ["fi", "en"],
    "nob_Latn": ["no", "en"],
    "por_Latn": ["pt", "en"],
    "ind_Latn": ["id", "en"],
    "zsm_Latn": ["ms", "en"],
    "swh_Latn": ["sw", "en"],
    "vie_Latn": ["vi", "en"],
    "tgl_Latn": ["tl", "en"],
    "jpn_Jpan": ["ja", "en"],
    "kor_Hang": ["ko", "en"],
    "zho_Hans": ["ch_sim", "en"],
    "yue_Hant": ["ch_tra", "en"],
    "rus_Cyrl": ["ru", "en"],
    "ukr_Cyrl": ["uk", "en"],
    "arb_Arab": ["ar", "en"],
    "hin_Deva": ["hi", "en"],
}

def get_ocr_reader(lang_key: str):
    langs = tuple(OCR_LANG_MAP.get(lang_key, ["en"]))
    if langs not in _ocr_readers:
        with _ocr_lock:
            if langs not in _ocr_readers:
                use_gpu = torch.cuda.is_available()
                print(f"[EasyOCR] Loading reader for {list(langs)} (gpu={use_gpu})...")
                _ocr_readers[langs] = easyocr.Reader(list(langs), gpu=use_gpu, verbose=False)
                print("[EasyOCR] Ready.")
    return _ocr_readers[langs]

@app.post("/api/ocr")
async def api_ocr(image: UploadFile = File(...), ocr_lang: str = Form("auto")):
    raw = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_np = np.array(pil_img)
        reader = get_ocr_reader(ocr_lang)
        results = reader.readtext(img_np, detail=1)
        lines = [r[1].strip() for r in results if r[1].strip()]
        text = " ".join(lines).strip()
        confs = [float(r[2]) for r in results if len(r) >= 3]
        avg_conf = round((sum(confs) / len(confs)) * 100, 1) if confs else 0.0
        return {
            "text": text,
            "ocr_confidence": avg_conf,
            "lid": lid_model.predict(text) if text else None,
        }
    except Exception as e:
        return JSONResponse({"error": f"OCR failed: {str(e)}"}, status_code=500)

@app.post("/api/tts")
async def api_tts(req: TtsRequest):
    global mms_tts_yor
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "No text to speak."}, status_code=400)

    if req.lang == "yor_Latn":
        if mms_tts_yor is None:
            device = 0 if torch.cuda.is_available() else -1
            print("[MMS-TTS] Loading facebook/mms-tts-yor...")
            mms_tts_yor = hf_pipeline("text-to-speech", model="facebook/mms-tts-yor", device=device)
        out = mms_tts_yor(text)
        audio_arr = np.asarray(out["audio"]).squeeze()
        audio_int16 = np.int16(np.clip(audio_arr, -1.0, 1.0) * 32767)
        buf = io.BytesIO()
        wavfile.write(buf, int(out["sampling_rate"]), audio_int16)
        return Response(content=buf.getvalue(), media_type="audio/wav")

    # Try Edge neural TTS first, then fall back to gTTS (supports Japanese, Arabic, Hindi, Swahili, etc.)
    try:
        voice = EDGE_TTS_VOICES.get(req.lang, "en-US-JennyNeural")
        comm = edge_tts.Communicate(text, voice)
        mp3_buf = io.BytesIO()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                mp3_buf.write(chunk["data"])
        if mp3_buf.tell() > 0:
            return Response(content=mp3_buf.getvalue(), media_type="audio/mpeg")
    except Exception as edge_err:
        print(f"[TTS] edge-tts failed ({edge_err}), falling back to gTTS...")

    gtts_code = GTTS_LANGS.get(req.lang, "en")
    mp3_buf = io.BytesIO()
    gTTS(text=text, lang=gtts_code).write_to_fp(mp3_buf)
    return Response(content=mp3_buf.getvalue(), media_type="audio/mpeg")

# ==============================================================================
# 7. EMBEDDED HTML / CSS / JS FRONTEND (Plum Purple, Gold & Off-White)
# ==============================================================================
EMBEDDED_HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light" data-cb="none" data-motion="full" data-font="standard" data-width="full">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>PolyComs — Veritas AI 4.4</title>
<style>
  :root {
    --bg: #FAF7F2;
    --surface: #FFFFFF;
    --surface-2: #F2ECE4;
    --text: #1F1224;
    --text-muted: #5C4963;
    --primary: #4A154B;
    --primary-hover: #611F69;
    --primary-text: #FFFFFF;
    --gold: #C89B2B;
    --gold-bg: #FDF6E2;
    --gold-text: #594106;
    --border: #DFD5E3;
    --danger-bg: #FDECEC;
    --danger-text: #8A1C1C;
    --ok-bg: #E8F5E9;
    --ok-text: #1B5E20;
    --font-size: 16px;
    --font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    --letter-spacing: 0px;
    --word-spacing: 0px;
    --line-height: 1.55;
    --reading-max: 1200px;
    --anim: 0.2s ease;
  }
  [data-theme="dark"] {
    --bg: #130A18;
    --surface: #1E1126;
    --surface-2: #2B1836;
    --text: #FAF7F2;
    --text-muted: #CBBBD1;
    --primary: #9D4EDD;
    --primary-hover: #B16CE8;
    --primary-text: #130A18;
    --gold: #F3C64F;
    --gold-bg: #332709;
    --gold-text: #F9DC8B;
    --border: #3E274D;
    --danger-bg: #3D161A;
    --danger-text: #FFB4B4;
    --ok-bg: #14331A;
    --ok-text: #A5D6A7;
  }
  [data-cb="deuteranopia"] {
    --primary: #2B3A67;
    --gold: #E69F00;
    --ok-bg: #E1F0FF;
    --ok-text: #004488;
    --danger-bg: #FFF0DB;
    --danger-text: #994F00;
  }
  [data-cb="tritanopia"] {
    --primary: #6A1B4D;
    --gold: #009E73;
    --ok-bg: #E0F5EE;
    --ok-text: #005A40;
  }
  [data-cb="high-contrast"] {
    --bg: #FFFFFF;
    --surface: #FFFFFF;
    --text: #000000;
    --border: #000000;
    --primary: #000000;
    --primary-text: #FFFFFF;
    --gold: #B8860B;
  }
  [data-theme="dark"][data-cb="high-contrast"] {
    --bg: #000000;
    --surface: #0A0A0A;
    --text: #FFFFFF;
    --border: #FFFFFF;
    --primary: #FFD700;
    --primary-text: #000000;
  }
  [data-font="dyslexic"] {
    --font-family: 'Atkinson Hyperlegible', 'Comic Sans MS', Verdana, sans-serif;
  }
  [data-width="narrow"] main { max-width: 68ch; }
  [data-motion="reduced"] * {
    transition: none !important;
    animation: none !important;
  }
  * { box-sizing: border-box; }
  :focus-visible {
    outline: 3px solid var(--gold) !important;
    outline-offset: 2px;
  }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-family);
    font-size: var(--font-size);
    letter-spacing: var(--letter-spacing);
    word-spacing: var(--word-spacing);
    line-height: var(--line-height);
    display: flex;
    min-height: 100vh;
    transition: background var(--anim), color var(--anim);
  }
  #reading-ruler {
    display: none;
    position: fixed;
    left: 0; right: 0;
    height: var(--ruler-height, 36px);
    background: var(--ruler-bg, rgba(212, 175, 55, 0.22));
    border-top: 2px solid var(--ruler-border, var(--gold));
    border-bottom: 2px solid var(--ruler-border, var(--gold));
    pointer-events: none;
    z-index: 9999;
    transform: translateY(-50%);
  }
  #screen-tint {
    display: none;
    position: fixed;
    inset: 0;
    background: var(--tint-color, transparent);
    pointer-events: none;
    z-index: 9998;
  }
  aside {
    width: 240px;
    background: var(--surface);
    border-right: 2px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 16px 12px;
    transition: width var(--anim);
    flex-shrink: 0;
  }
  aside.collapsed { width: 74px; }
  aside.collapsed .nav-label, aside.collapsed .brand-text { display: none; }
  .brand {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px 18px; border-bottom: 1px solid var(--border); margin-bottom: 12px;
  }
  .brand-badge {
    width: 36px; height: 36px; border-radius: 10px;
    background: var(--primary); color: var(--gold);
    display: grid; place-items: center; font-weight: 800; font-size: 18px; flex-shrink: 0;
    border: 2px solid var(--gold);
  }
  .brand-text strong { display: block; font-size: 1.05em; color: var(--text); }
  .brand-text small { color: var(--text-muted); font-size: 0.75em; }
  .nav-btn {
    display: flex; align-items: center; gap: 12px; width: 100%;
    padding: 11px 12px; margin-bottom: 6px; border: 1px solid transparent;
    border-radius: 10px; background: transparent; color: var(--text);
    cursor: pointer; font-size: 0.95em; font-weight: 600; text-align: left;
    transition: background var(--anim), border-color var(--anim);
  }
  .nav-btn:hover { background: var(--surface-2); }
  .nav-btn.active {
    background: var(--primary); color: var(--primary-text);
    border-color: var(--gold);
  }
  .nav-icon { width: 22px; height: 22px; flex-shrink: 0; }
  .collapse-btn {
    margin-top: auto; background: var(--surface-2); border: 1px solid var(--border);
    color: var(--text); border-radius: 8px; padding: 8px; cursor: pointer; font-weight: 600;
  }
  main { flex: 1; padding: 22px 28px; max-width: var(--reading-max); margin: 0 auto; width: 100%; }
  header.topbar {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 12px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 2px solid var(--border);
  }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px;
    border-radius: 999px; font-size: 0.82em; font-weight: 700;
    background: var(--gold-bg); color: var(--gold-text); border: 1px solid var(--gold);
  }
  .card {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: 14px; padding: 18px; margin-bottom: 18px;
  }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 820px) { .grid-2 { grid-template-columns: 1fr; } }
  textarea, select, input[type="text"] {
    width: 100%; padding: 12px; border-radius: 10px;
    border: 1.5px solid var(--border); background: var(--bg); color: var(--text);
    font-family: inherit; font-size: 1em; letter-spacing: inherit; line-height: inherit;
  }
  textarea { min-height: 140px; resize: vertical; }
  .btn {
    display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px;
    border-radius: 10px; border: 1.5px solid transparent; cursor: pointer;
    font-weight: 700; font-size: 0.92em; font-family: inherit;
    transition: transform var(--anim), background var(--anim);
  }
  .btn-primary { background: var(--primary); color: var(--primary-text); border-color: var(--gold); }
  .btn-primary:hover { background: var(--primary-hover); }
  .btn-secondary { background: var(--surface-2); color: var(--text); border-color: var(--border); }
  .btn-icon { padding: 8px 12px; }
  .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .alert {
    padding: 12px 15px; border-radius: 10px; margin-bottom: 14px; font-weight: 600;
    border-left: 5px solid var(--gold); background: var(--gold-bg); color: var(--gold-text);
  }
  .alert-danger {
    border-left-color: var(--danger-text); background: var(--danger-bg); color: var(--danger-text);
  }
  .alert-ok {
    border-left-color: var(--ok-text); background: var(--ok-bg); color: var(--ok-text);
  }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 0.92em; }
  th, td { padding: 10px; border-bottom: 1px solid var(--border); text-align: left; }
  th { background: var(--surface-2); font-weight: 700; }
  .meter { height: 10px; background: var(--surface-2); border-radius: 999px; overflow: hidden; margin-top: 6px; }
  .meter-fill { height: 100%; background: var(--gold); transition: width var(--anim); }
  .alt-chip {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 12px; margin-top: 6px; cursor: pointer;
  }
  .sent-chip {
    display: inline-block; padding: 5px 9px; margin: 3px; border-radius: 8px;
    background: var(--surface-2); border: 1px solid var(--border); cursor: pointer;
  }
  .sent-chip.speaking {
    background: var(--gold-bg); border: 2px solid var(--gold); color: var(--gold-text); font-weight: 700;
  }
  .disclaimer {
    font-size: 0.82em; color: var(--text-muted); border-top: 1px solid var(--border);
    padding-top: 12px; margin-top: 18px;
  }
</style>
</head>
<body>
<div id="screen-tint" aria-hidden="true"></div>
<div id="reading-ruler" aria-hidden="true"></div>

<aside id="sidebar" aria-label="Primary Navigation">
  <div class="brand">
    <div class="brand-badge" aria-hidden="true">文</div>
    <div class="brand-text">
      <strong>PolyComs</strong>
      <small>Veritas AI · Track 4.4</small>
    </div>
  </div>

  <div role="tablist" aria-orientation="vertical">
    <button role="tab" aria-selected="true" class="nav-btn active" onclick="switchTab('translate', this)" title="Translate & Explain">
      <span class="nav-icon" aria-hidden="true">🌐</span><span class="nav-label" data-i18n="nav_translate">Translate & Explain</span>
    </button>
    <button role="tab" aria-selected="false" class="nav-btn" onclick="switchTab('lid', this)" title="Language ID (4.4.1)">
      <span class="nav-icon" aria-hidden="true">🔍</span><span class="nav-label" data-i18n="nav_lid">Language ID</span>
    </button>
    <button role="tab" aria-selected="false" class="nav-btn" onclick="switchTab('quality', this)" title="Quality & chrF (4.4.2)">
      <span class="nav-icon" aria-hidden="true">📊</span><span class="nav-label" data-i18n="nav_quality">Quality (chrF)</span>
    </button>
    <button role="tab" aria-selected="false" class="nav-btn" onclick="switchTab('safety', this)" title="Safety Evaluation">
      <span class="nav-icon" aria-hidden="true">🛡️</span><span class="nav-label" data-i18n="nav_safety">Safety Check</span>
    </button>
    <button role="tab" aria-selected="false" class="nav-btn" onclick="switchTab('accessibility', this)" title="Accessibility">
      <span class="nav-icon" aria-hidden="true">♿</span><span class="nav-label" data-i18n="nav_a11y">Accessibility</span>
    </button>
  </div>

  <button class="collapse-btn" onclick="document.getElementById('sidebar').classList.toggle('collapsed')" aria-label="Collapse or expand sidebar" title="Toggle Sidebar">⇄</button>
</aside>

<main>
  <header class="topbar">
    <div class="row">
      <span class="pill" id="gpu-pill" aria-live="polite">⚙️ Checking Colab Backend...</span>
      <span class="pill">🤖 NLLB-200 + Tatoeba + Whisper</span>
    </div>
    <div class="row">
      <label for="ui-lang" title="Interface Language">🌍</label>
      <select id="ui-lang" aria-label="Interface language" style="width:auto;padding:6px 10px;" onchange="setUiLang(this.value)">
        <option value="en">English (UI)</option>
        <option value="es">Español (UI)</option>
        <option value="fr">Français (UI)</option>
        <option value="sw">Kiswahili (UI)</option>
        <option value="ja">日本語 (UI)</option>
      </select>
      <button class="btn btn-secondary btn-icon" onclick="toggleTheme()" aria-label="Toggle light and dark mode" title="Light / Dark Mode">🌓</button>
    </div>
  </header>

  <!-- TAB 1: TRANSLATE & EXPLAIN -->
  <section id="tab-translate" class="tab-panel active" aria-label="Translate and Explain">
    <div id="safety-banner" class="alert alert-danger" style="display:none;" role="alert" aria-live="assertive"></div>
    <div id="profanity-banner" class="alert" style="display:none;" role="alert" aria-live="polite"></div>
    <div id="confusion-banner" class="alert" style="display:none;" aria-live="polite"></div>

    <div class="card">
      <div class="row" style="justify-content:space-between; margin-bottom:10px;">
        <div class="row" style="flex:1;">
          <select id="src-lang" aria-label="Source language" style="max-width:215px;" onchange="renderKeyboard()">
            <option value="auto">✨ Auto-Detect (Tatoeba)</option>
            <option value="eng_Latn">English</option>
            <option value="fra_Latn">French / Français</option>
            <option value="spa_Latn">Spanish / Español</option>
            <option value="deu_Latn">German / Deutsch</option>
            <option value="swe_Latn">Swedish / Svenska</option>
            <option value="fin_Latn">Finnish / Suomi</option>
            <option value="nob_Latn">Norwegian / Norsk</option>
            <option value="por_Latn">Portuguese / Português</option>
            <option value="rus_Cyrl">Russian / Русский</option>
            <option value="ukr_Cyrl">Ukrainian / Українська</option>
            <option value="jpn_Jpan">Japanese / 日本語</option>
            <option value="kor_Hang">Korean / 한국어</option>
            <option value="zho_Hans">Chinese (Mandarin) / 普通话</option>
            <option value="yue_Hant">Chinese (Cantonese) / 廣東話</option>
            <option value="vie_Latn">Vietnamese / Tiếng Việt</option>
            <option value="tgl_Latn">Tagalog / Wikang Tagalog</option>
            <option value="ind_Latn">Indonesian / Bahasa Indonesia</option>
            <option value="zsm_Latn">Malay / Bahasa Melayu</option>
            <option value="swh_Latn">Swahili / Kiswahili</option>
            <option value="yor_Latn">Yoruba / Yorùbá</option>
            <option value="mri_Latn">Māori / Te Reo Māori</option>
            <option value="smo_Latn">Samoan / Gagana Sāmoa</option>
            <option value="arb_Arab">Arabic / العربية</option>
            <option value="hin_Deva">Hindi / हिन्दी</option>
          </select>
          <span style="font-weight:800;" aria-hidden="true">→</span>
          <select id="tgt-lang" aria-label="Target language" style="max-width:215px;">
            <option value="eng_Latn">English</option>
            <option value="fra_Latn">French / Français</option>
            <option value="spa_Latn">Spanish / Español</option>
            <option value="deu_Latn">German / Deutsch</option>
            <option value="swe_Latn">Swedish / Svenska</option>
            <option value="fin_Latn">Finnish / Suomi</option>
            <option value="nob_Latn">Norwegian / Norsk</option>
            <option value="por_Latn">Portuguese / Português</option>
            <option value="rus_Cyrl">Russian / Русский</option>
            <option value="ukr_Cyrl">Ukrainian / Українська</option>
            <option value="jpn_Jpan">Japanese / 日本語</option>
            <option value="kor_Hang">Korean / 한국어</option>
            <option value="zho_Hans">Chinese (Mandarin) / 普通话</option>
            <option value="yue_Hant">Chinese (Cantonese) / 廣東話</option>
            <option value="vie_Latn">Vietnamese / Tiếng Việt</option>
            <option value="tgl_Latn">Tagalog / Wikang Tagalog</option>
            <option value="ind_Latn">Indonesian / Bahasa Indonesia</option>
            <option value="zsm_Latn">Malay / Bahasa Melayu</option>
            <option value="swh_Latn">Swahili / Kiswahili</option>
            <option value="yor_Latn">Yoruba / Yorùbá</option>
            <option value="mri_Latn">Māori / Te Reo Māori</option>
            <option value="smo_Latn">Samoan / Gagana Sāmoa</option>
            <option value="arb_Arab">Arabic / العربية</option>
            <option value="hin_Deva">Hindi / हिन्दी</option>
          </select>
        </div>

        <div class="row">
          <select id="stt-lang" aria-label="Spoken input language for Whisper" title="Spoken language (Whisper)" style="width:auto;padding:8px 10px;">
            <option value="auto">🎙️ Auto STT</option>
            <option value="en">🎙️ English</option>
            <option value="ja">🎙️ 日本語 (Japanese)</option>
            <option value="ko">🎙️ 한국어 (Korean)</option>
            <option value="zh">🎙️ 中文 (Mandarin/Cantonese)</option>
            <option value="de">🎙️ Deutsch</option>
            <option value="sv">🎙️ Svenska</option>
            <option value="fi">🎙️ Suomi</option>
            <option value="no">🎙️ Norsk</option>
            <option value="fr">🎙️ Français</option>
            <option value="es">🎙️ Español</option>
            <option value="pt">🎙️ Português</option>
            <option value="ru">🎙️ Русский</option>
            <option value="uk">🎙️ Українська</option>
            <option value="vi">🎙️ Tiếng Việt</option>
            <option value="tl">🎙️ Tagalog</option>
            <option value="sw">🎙️ Kiswahili</option>
            <option value="yo">🎙️ Yorùbá</option>
            <option value="mi">🎙️ Te Reo Māori</option>
            <option value="id">🎙️ Indonesian</option>
            <option value="ms">🎙️ Malay</option>
            <option value="ar">🎙️ العربية</option>
            <option value="hi">🎙️ हिन्दी</option>
          </select>
          <button class="btn btn-secondary btn-icon" id="mic-btn" onclick="toggleMic()" aria-label="Record speech input">🎙️ <span data-i18n="btn_speak">Speak</span></button>
          <label class="btn btn-secondary btn-icon" title="Upload Audio Clip (Whisper)" style="margin:0;">📁 <input type="file" id="audio-upload" accept="audio/*" aria-label="Upload audio file" style="display:none;" onchange="uploadAudio(this)" /></label>
          <label class="btn btn-secondary btn-icon" title="Extract Text from Image (OCR)" style="margin:0;">📷 <span style="font-size:0.85em;">OCR</span> <input type="file" id="ocr-upload-translate" accept="image/*" aria-label="Upload image for OCR translation" style="display:none;" onchange="runOcr(this, 'translate')" /></label>
          <button class="btn btn-primary" onclick="runTranslate()">⚡ <span data-i18n="btn_translate">Translate</span></button>
        </div>
      </div>

      <!-- Audio Visualizer + Cognitive & Safe Mode Quick Bar -->
      <div id="mic-vis-wrap" style="display:none; margin-bottom:10px;">
        <small>🎙️ Listening... Audio Input Level:</small>
        <div class="meter"><div class="meter-fill" id="mic-vis-bar" style="width:0%"></div></div>
      </div>

      <div class="row" style="margin-bottom:12px; background:var(--surface-2); padding:10px; border-radius:10px;">
        <label for="reading-level"><strong>📖 Reading Level:</strong></label>
        <select id="reading-level" style="width:auto;padding:6px 10px;" onchange="savePrefs()">
          <option value="standard">Standard</option>
          <option value="simple">Simple / Easy-Read</option>
          <option value="steps">Step-by-Step Chunks</option>
        </select>

        <label for="safe-mode"><strong>🛡️ Safe Language Mode:</strong></label>
        <select id="safe-mode" style="width:auto;padding:6px 10px;" onchange="savePrefs()">
          <option value="preserve">Preserve Original Wording</option>
          <option value="censor">Censor Profanity (f***)</option>
          <option value="replace">Replace with Milder Wording</option>
          <option value="child_friendly">Child-Friendly Explanation</option>
        </select>
      </div>

      <div class="grid-2">
        <div>
          <div class="row" style="justify-content:space-between;">
            <label for="src-text"><strong data-i18n="lbl_before">Before (Input / Editable Captions)</strong></label>
            <button type="button" class="btn btn-secondary btn-icon" onclick="toggleKeyboard()" aria-expanded="false" id="kbd-toggle-btn" title="Toggle ARIA On-Screen Keyboard">⌨️ Keyboard</button>
          </div>
          <textarea id="src-text" placeholder="Type, paste, or speak in any language..."></textarea>
          <div id="aria-keyboard" role="region" aria-label="On-Screen Keyboard" style="display:none; margin-top:8px; padding:10px; background:var(--surface-2); border-radius:10px; border:1px solid var(--border);"></div>
        </div>
        <div>
          <div class="row" style="justify-content:space-between;">
            <label for="tgt-text"><strong data-i18n="lbl_after">After (NLLB-200 Output)</strong></label>
            <div class="row">
              <button class="btn btn-secondary btn-icon" onclick="speakOutput(1.0)" aria-label="Play full audio" title="Play Full Audio">🔊</button>
              <button class="btn btn-secondary btn-icon" onclick="speakOutput(0.7)" aria-label="Play slow pronunciation" title="Slow Pronunciation (0.7x)">🐢 0.7x</button>
              <button class="btn btn-secondary btn-icon" onclick="playSentences()" aria-label="Play sentence by sentence" title="Sentence-by-Sentence Playback">⏯️ Sentences</button>
              <button class="btn btn-secondary btn-icon" onclick="stopAudio()" aria-label="Stop audio" title="Stop Audio">⏹️</button>
            </div>
          </div>
          <textarea id="tgt-text" readonly aria-live="polite" placeholder="Translation will appear here..."></textarea>
        </div>
      </div>

      <!-- Sentence-by-sentence Synchronized Highlighting -->
      <div id="sentence-player" style="margin-top:10px; display:none;">
        <small><strong>🎧 Sentence-by-Sentence Reader (Click any sentence to replay):</strong></small>
        <div id="sentence-chips" style="margin-top:4px;"></div>
      </div>
    </div>

    <!-- Honesty, Cognitive & Explanation Panels -->
    <div class="grid-2">
      <div class="card">
        <h3>🎯 <span data-i18n="hdr_honesty">Honesty & Alternatives</span></h3>
        <div>Confidence: <strong id="conf-val">—</strong></div>
        <div class="meter"><div class="meter-fill" id="conf-bar" style="width:0%"></div></div>
        <div style="margin-top:12px;"><strong>Other Options (Beam Search):</strong></div>
        <div id="alt-list" style="color:var(--text-muted);">Run a translation to view alternatives.</div>
      </div>

      <div class="card">
        <h3>🧠 <span data-i18n="hdr_explain">Model Explanation, Easy-Read & Key Info</span></h3>
        <p><strong>Detected Register:</strong> <span id="reg-val" class="pill">—</span></p>
        <div id="key-info-box"></div>
        <div id="glossary-box"></div>
        <div id="idiom-box"></div>
        <ul id="grammar-list" style="padding-left:18px; margin:8px 0 0;"></ul>
      </div>
    </div>
  </section>

  <!-- TAB 2: LANGUAGE ID (4.4.1) -->
  <section id="tab-lid" class="tab-panel" aria-label="Language Identification">
    <div class="card">
      <h3>🔍 4.4.1 Language Identification (Tatoeba TF-IDF vs. langdetect)</h3>
      <div class="row" style="margin-bottom:10px;">
        <input type="text" id="lid-input" aria-label="Sentence to identify" placeholder="Enter a short or ambiguous sentence (or upload an image with OCR)..." style="flex:1;" />
        <select id="lid-ocr-script" aria-label="OCR script hint" title="OCR script hint for images" style="width:auto;padding:10px;">
          <option value="auto">📷 Latin / English</option>
          <option value="jpn_Jpan">📷 日本語 (Japanese)</option>
          <option value="kor_Hang">📷 한국어 (Korean)</option>
          <option value="zho_Hans">📷 中文 (Simplified)</option>
          <option value="yue_Hant">📷 廣東話 (Traditional)</option>
          <option value="rus_Cyrl">📷 Русский</option>
          <option value="ukr_Cyrl">📷 Українська</option>
          <option value="arb_Arab">📷 العربية</option>
          <option value="hin_Deva">📷 हिन्दी</option>
        </select>
        <label class="btn btn-secondary btn-icon" title="Extract Text from Image (OCR)" style="margin:0;">📷 OCR <input type="file" id="ocr-upload-lid" accept="image/*" aria-label="Upload image for Language ID OCR" style="display:none;" onchange="runOcr(this, 'lid')" /></label>
        <button class="btn btn-primary" onclick="runLidPredict()">Identify</button>
      </div>
      <div id="lid-result" aria-live="polite"></div>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <h3>📈 Per-Language Accuracy & Confusion Matrix</h3>
        <button class="btn btn-secondary" onclick="loadLidMetrics()">Refresh Metrics</button>
      </div>
      <div id="lid-metrics-box" style="margin-top:12px;">Loading metrics...</div>
    </div>
  </section>

  <!-- TAB 3: QUALITY & chrF (4.4.2) -->
  <section id="tab-quality" class="tab-panel" aria-label="Translation Quality">
    <div class="card">
      <h3>📊 4.4.2 Translation Quality Test (NLLB-200 on FLORES+ Samples · chrF)</h3>
      <p>Select which languages to test directly with <strong>sacrebleu chrF</strong> (comparing widely served vs. less served languages) and view model outputs next to professional references:</p>
      <div class="row" id="chrf-lang-checks" style="margin-bottom:12px;">
        <label><input type="checkbox" value="fra_Latn" checked /> French</label>
        <label><input type="checkbox" value="deu_Latn" /> German</label>
        <label><input type="checkbox" value="swe_Latn" /> Swedish</label>
        <label><input type="checkbox" value="fin_Latn" /> Finnish</label>
        <label><input type="checkbox" value="nob_Latn" /> Norwegian</label>
        <label><input type="checkbox" value="spa_Latn" /> Spanish</label>
        <label><input type="checkbox" value="rus_Cyrl" /> Russian</label>
        <label><input type="checkbox" value="ukr_Cyrl" /> Ukrainian</label>
        <label><input type="checkbox" value="jpn_Jpan" /> Japanese</label>
        <label><input type="checkbox" value="kor_Hang" /> Korean</label>
        <label><input type="checkbox" value="zho_Hans" /> Mandarin</label>
        <label><input type="checkbox" value="swh_Latn" checked /> Swahili</label>
        <label><input type="checkbox" value="yor_Latn" checked /> Yoruba</label>
      </div>
      <button class="btn btn-primary" onclick="runBenchmark()">Run Selected chrF Benchmark</button>
      <div id="bench-summary" class="row" style="margin-top:14px;"></div>
      <div id="bench-table" style="margin-top:14px; overflow-x:auto;"></div>
    </div>

    <div class="card">
      <h3>✍️ Compare Your Translation vs. Known Good Reference (chrF)</h3>
      <div class="grid-2">
        <input type="text" id="user-hyp" aria-label="Your translation" placeholder="Your / Model translation..." />
        <input type="text" id="user-ref" aria-label="Reference translation" placeholder="Known good reference translation..." />
      </div>
      <div class="row" style="margin-top:10px;">
        <button class="btn btn-secondary" onclick="checkUserChrf()">Compute chrF Score</button>
        <strong id="user-chrf-out" aria-live="polite"></strong>
      </div>
    </div>
  </section>

  <!-- TAB 4: SAFETY EVALUATION -->
  <section id="tab-safety" class="tab-panel" aria-label="Safety Evaluation">
    <div class="card">
      <h3>🛡️ Safety Evaluation — High-Stakes & Safe Language Mode Testing</h3>
      <p>Test how high-stakes instructions (medication, driving directions, emergencies) and strong language are flagged without silently omitting critical warnings.</p>
      <div class="row" style="margin-bottom:12px;">
        <label style="cursor:pointer;"><input type="checkbox" id="toggle-advisory-safety" onchange="setAdvisoryBanners(this.checked)" /> <strong>Show Profanity & Language-Overlap Advisory Banners</strong> (hidden by default; Medical/Driving alerts always stay active)</label>
      </div>
      <div class="row">
        <button class="btn btn-secondary" onclick="testSafetySample('The implementation of the aforementioned medication is contingent upon eating first. Take 2 tablets every 8 hours. Do not drive if dizzy.')">💊 Test Complex Medical</button>
        <button class="btn btn-secondary" onclick="testSafetySample('Turn left at the highway exit and drive 2 km.')">🚗 Test Driving</button>
        <button class="btn btn-secondary" onclick="testSafetySample('Call an ambulance immediately, severe allergy.')">🚑 Test Emergency</button>
        <button class="btn btn-secondary" onclick="testSafetySample('That was absolutely terrible, you idiot!')">🤬 Test Safe Language Mode</button>
      </div>
    </div>
  </section>

  <!-- TAB 5: ACCESSIBILITY & JUDGE DEMO -->
  <section id="tab-accessibility" class="tab-panel" aria-label="Accessibility Settings">
    <div class="card">
      <h3>🎬 Demonstrable Accessibility Showcase (Judge Demo)</h3>
      <p>Click below to compare the exact same sentence across <strong>Standard</strong>, <strong>Simple / Easy-Read</strong>, <strong>Dyslexia-Friendly</strong>, <strong>Safe Language Mode</strong>, and <strong>Synchronized Sentence Audio</strong>.</p>
      <button class="btn btn-primary" onclick="runJudgeDemo()">✨ Run 5-Mode Accessibility Demo</button>
      <div id="judge-demo-out" style="margin-top:14px;"></div>
    </div>

    <div class="card">
      <h3>♿ Visual, Cognitive, Audio & Privacy Settings (Auto-Saved)</h3>
      <div class="grid-2">
        <div>
          <label><strong>Color Theme & Contrast</strong></label><br/>
          <button class="btn btn-secondary" onclick="setTheme('light')">☀️ Light Mode</button>
          <button class="btn btn-secondary" onclick="setTheme('dark')">🌙 Dark Mode</button>
          <select id="cb-select" style="margin-top:8px;" onchange="setCB(this.value)">
            <option value="none">Default (Plum & Gold)</option>
            <option value="deuteranopia">Protanopia / Deuteranopia Safe</option>
            <option value="tritanopia">Tritanopia Safe</option>
            <option value="high-contrast">High Contrast</option>
          </select>
        </div>
        <div>
          <label><strong>Reading Assistance</strong></label><br/>
          <button class="btn btn-secondary" onclick="toggleDyslexic()">🔤 Dyslexia-Friendly Font</button>
          <button class="btn btn-secondary" onclick="toggleMotion()">⏸️ Reduce Motion</button>
          <button class="btn btn-secondary" onclick="toggleRuler()" style="margin-top:6px;">📏 Toggle Reading Ruler</button>
          <button class="btn btn-secondary" onclick="toggleWidth()" style="margin-top:6px;">📐 Toggle Narrow Reading Column</button>
          <button class="btn btn-secondary" onclick="toggleFocusAnchor()" style="margin-top:6px;">👁️ Focus-Anchor Reading (Bold Starts)</button>
          <div class="row" style="margin-top:8px;">
            <label for="ruler-color"><strong>Ruler Color:</strong></label>
            <input type="color" id="ruler-color" value="#D4AF37" onchange="updateRulerStyle()" style="width:42px;height:32px;padding:2px;border-radius:6px;cursor:pointer;" />
            <label for="ruler-height"><strong>Height:</strong></label>
            <input type="range" id="ruler-height" min="24" max="72" value="36" style="width:90px;" oninput="updateRulerStyle()" />
          </div>
          <div class="row" style="margin-top:6px;">
            <label for="tint-select"><strong>Page Tint (Visual Stress):</strong></label>
            <select id="tint-select" style="width:auto;padding:5px 8px;" onchange="setScreenTint(this.value)">
              <option value="none">None</option>
              <option value="rgba(255, 245, 200, 0.18)">Warm Cream</option>
              <option value="rgba(173, 216, 230, 0.20)">Soft Blue</option>
              <option value="rgba(180, 238, 180, 0.18)">Soft Mint</option>
              <option value="rgba(255, 192, 203, 0.18)">Soft Rose</option>
            </select>
          </div>
        </div>
        <div>
          <label for="fs-slider"><strong>Font Size: <span id="fs-label">16px</span></strong></label>
          <input id="fs-slider" type="range" min="14" max="26" value="16" style="width:100%;" oninput="setVar('--font-size', this.value+'px', 'fs-label', this.value+'px')" />
          <label for="ls-slider"><strong>Letter Spacing: <span id="ls-label">0px</span></strong></label>
          <input id="ls-slider" type="range" min="0" max="4" step="0.5" value="0" style="width:100%;" oninput="setVar('--letter-spacing', this.value+'px', 'ls-label', this.value+'px')" />
        </div>
        <div>
          <label for="ws-slider"><strong>Word Spacing: <span id="ws-label">0px</span></strong></label>
          <input id="ws-slider" type="range" min="0" max="8" step="1" value="0" style="width:100%;" oninput="setVar('--word-spacing', this.value+'px', 'ws-label', this.value+'px')" />
          <label for="lh-slider"><strong>Line Spacing: <span id="lh-label">1.55</span></strong></label>
          <input id="lh-slider" type="range" min="1.2" max="2.4" step="0.1" value="1.55" style="width:100%;" oninput="setVar('--line-height', this.value, 'lh-label', this.value)" />
        </div>
        <div>
          <label for="spd-slider"><strong>Speech Playback Speed: <span id="spd-label">1.0x</span></strong></label>
          <input id="spd-slider" type="range" min="0.6" max="1.5" step="0.1" value="1.0" style="width:100%;" oninput="speechRate=parseFloat(this.value); document.getElementById('spd-label').textContent=this.value+'x'; savePrefs();" />
        </div>
        <div>
          <label><strong>Notification & Privacy Controls</strong></label><br/>
          <label style="display:block;margin-bottom:8px;cursor:pointer;"><input type="checkbox" id="toggle-advisory-a11y" onchange="setAdvisoryBanners(this.checked)" /> Show Profanity & Language-Overlap Banners</label>
          <button class="btn btn-secondary" onclick="clearPrivacyData()">🧹 Clear Text, Audio & Saved Settings</button>
        </div>
      </div>
    </div>
  </section>

  <div class="disclaimer">
    ⚠️ <strong>AI Research & Safety Disclaimer:</strong> PolyComs is a high school hackathon prototype powered by open ML models (NLLB-200, Tatoeba TF-IDF, Whisper, MMS-TTS). Machine translation and Safe Language Mode filters can make mistakes or miss slang—always verify medical, legal, or driving instructions with a qualified human speaker. Audio is processed transiently and not stored.
  </div>
</main>

<script>
const LANG_LABELS = {
  eng_Latn: "English", eng: "English",
  fra_Latn: "French / Français", fra: "French / Français",
  spa_Latn: "Spanish / Español", spa: "Spanish / Español",
  deu_Latn: "German / Deutsch", deu: "German / Deutsch",
  swe_Latn: "Swedish / Svenska", swe: "Swedish / Svenska",
  fin_Latn: "Finnish / Suomi", fin: "Finnish / Suomi",
  nob_Latn: "Norwegian / Norsk", nob: "Norwegian / Norsk",
  por_Latn: "Portuguese / Português", por: "Portuguese / Português",
  rus_Cyrl: "Russian / Русский", rus: "Russian / Русский",
  ukr_Cyrl: "Ukrainian / Українська", ukr: "Ukrainian / Українська",
  jpn_Jpan: "Japanese / 日本語", jpn: "Japanese / 日本語",
  kor_Hang: "Korean / 한국어", kor: "Korean / 한국어",
  zho_Hans: "Chinese (Mandarin) / 普通话", cmn: "Chinese (Mandarin) / 普通话",
  yue_Hant: "Chinese (Cantonese) / 廣東話", yue: "Chinese (Cantonese) / 廣東話",
  vie_Latn: "Vietnamese / Tiếng Việt", vie: "Vietnamese / Tiếng Việt",
  tgl_Latn: "Tagalog", tgl: "Tagalog",
  ind_Latn: "Indonesian / Bahasa Indonesia", ind: "Indonesian / Bahasa Indonesia",
  zsm_Latn: "Malay / Bahasa Melayu", zsm: "Malay / Bahasa Melayu",
  swh_Latn: "Swahili / Kiswahili", swh: "Swahili / Kiswahili",
  yor_Latn: "Yoruba / Yorùbá", yor: "Yoruba / Yorùbá",
  mri_Latn: "Māori / Te Reo Māori", mri: "Māori / Te Reo Māori",
  smo_Latn: "Samoan / Gagana Sāmoa", smo: "Samoan / Gagana Sāmoa",
  arb_Arab: "Arabic / العربية", arb: "Arabic / العربية",
  hin_Deva: "Hindi / हिन्दी", hin: "Hindi / हिन्दी"
};
function formatLangName(code) { return LANG_LABELS[code] || code; }

let speechRate = 1.0;
let rulerOn = false;
let focusAnchorOn = false;
let showAdvisoryBanners = false;

function setAdvisoryBanners(val) {
  showAdvisoryBanners = !!val;
  const s = document.getElementById("toggle-advisory-safety");
  const a = document.getElementById("toggle-advisory-a11y");
  if (s) s.checked = showAdvisoryBanners;
  if (a) a.checked = showAdvisoryBanners;
  if (!showAdvisoryBanners) {
    document.getElementById("confusion-banner").style.display = "none";
    document.getElementById("profanity-banner").style.display = "none";
  }
  savePrefs();
}
let activeAudio = null;
let stopSeq = false;

const UI_I18N = {
  en: { nav_translate: "Translate & Explain", nav_lid: "Language ID", nav_quality: "Quality (chrF)", nav_safety: "Safety Check", nav_a11y: "Accessibility", btn_speak: "Speak", btn_translate: "Translate", lbl_before: "Before (Input / Editable Captions)", lbl_after: "After (NLLB-200 Output)", hdr_honesty: "Honesty & Alternatives", hdr_explain: "Model Explanation, Easy-Read & Key Info" },
  es: { nav_translate: "Traducir y Explicar", nav_lid: "Identificar Idioma", nav_quality: "Calidad (chrF)", nav_safety: "Seguridad", nav_a11y: "Accesibilidad", btn_speak: "Hablar", btn_translate: "Traducir", lbl_before: "Antes (Entrada)", lbl_after: "Después (NLLB-200)", hdr_honesty: "Honestidad y Opciones", hdr_explain: "Explicación y Lectura Fácil" },
  fr: { nav_translate: "Traduire & Expliquer", nav_lid: "Identification", nav_quality: "Qualité (chrF)", nav_safety: "Sécurité", nav_a11y: "Accessibilité", btn_speak: "Parler", btn_translate: "Traduire", lbl_before: "Avant (Entrée)", lbl_after: "Après (NLLB-200)", hdr_honesty: "Fiabilité & Options", hdr_explain: "Explication & Facile à Lire" },
  sw: { nav_translate: "Tafsiri & Eleza", nav_lid: "Tambua Lugha", nav_quality: "Ubora (chrF)", nav_safety: "Usalama", nav_a11y: "Ufikivu", btn_speak: "Sema", btn_translate: "Tafsiri", lbl_before: "Kabla (Ingizo)", lbl_after: "Baada (NLLB-200)", hdr_honesty: "Uaminifu & Chaguo", hdr_explain: "Maelezo & Lugha Rahisi" },
  ja: { nav_translate: "翻訳と解説", nav_lid: "言語判定", nav_quality: "品質 (chrF)", nav_safety: "安全性チェック", nav_a11y: "アクセシビリティ", btn_speak: "音声入力", btn_translate: "翻訳する", lbl_before: "翻訳前（編集可能）", lbl_after: "翻訳後 (NLLB-200)", hdr_honesty: "信頼度と代替案", hdr_explain: "モデルの解説・やさしい表現" },
};

function setUiLang(lang) {
  const dict = UI_I18N[lang] || UI_I18N.en;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (dict[key]) el.textContent = dict[key];
  });
  savePrefs();
}

function switchTab(id, btn) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
  document.getElementById("tab-" + id).classList.add("active");
  if (btn) { btn.classList.add("active"); btn.setAttribute("aria-selected", "true"); }
  if (id === "lid") loadLidMetrics();
}

function setTheme(t) { document.documentElement.setAttribute("data-theme", t); savePrefs(); }
function toggleTheme() { setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"); }
function setCB(v) { document.documentElement.setAttribute("data-cb", v); savePrefs(); }
function toggleDyslexic() {
  const cur = document.documentElement.getAttribute("data-font");
  document.documentElement.setAttribute("data-font", cur === "dyslexic" ? "standard" : "dyslexic");
  savePrefs();
}
function toggleMotion() {
  const cur = document.documentElement.getAttribute("data-motion");
  document.documentElement.setAttribute("data-motion", cur === "reduced" ? "full" : "reduced");
  savePrefs();
}
function toggleWidth() {
  const cur = document.documentElement.getAttribute("data-width");
  document.documentElement.setAttribute("data-width", cur === "narrow" ? "full" : "narrow");
  savePrefs();
}
function toggleRuler() {
  rulerOn = !rulerOn;
  document.getElementById("reading-ruler").style.display = rulerOn ? "block" : "none";
  updateRulerStyle();
  savePrefs();
}
function updateRulerStyle() {
  const hex = document.getElementById("ruler-color")?.value || "#D4AF37";
  const h = document.getElementById("ruler-height")?.value || 36;
  const r = parseInt(hex.slice(1,3), 16), g = parseInt(hex.slice(3,5), 16), b = parseInt(hex.slice(5,7), 16);
  document.documentElement.style.setProperty("--ruler-bg", `rgba(${r}, ${g}, ${b}, 0.24)`);
  document.documentElement.style.setProperty("--ruler-border", hex);
  document.documentElement.style.setProperty("--ruler-height", h + "px");
  savePrefs();
}
function setScreenTint(val) {
  const el = document.getElementById("screen-tint");
  if (!val || val === "none") {
    el.style.display = "none";
  } else {
    document.documentElement.style.setProperty("--tint-color", val);
    el.style.display = "block";
  }
  savePrefs();
}
function toggleFocusAnchor() {
  focusAnchorOn = !focusAnchorOn;
  savePrefs();
  const chips = document.querySelectorAll(".sent-chip");
  chips.forEach(c => {
    const raw = c.dataset.raw || c.textContent;
    c.dataset.raw = raw;
    c.innerHTML = focusAnchorOn ? applyFocusAnchor(raw) : raw;
  });
}
function applyFocusAnchor(text) {
  return text.split(/(\s+)/).map(tok => {
    if (/^\s+$/.test(tok) || tok.length < 2) return tok;
    const cut = Math.max(1, Math.ceil(tok.length * 0.4));
    return `<b>${tok.slice(0, cut)}</b>${tok.slice(cut)}`;
  }).join("");
}
document.addEventListener("mousemove", e => {
  if (rulerOn) document.getElementById("reading-ruler").style.top = e.clientY + "px";
});

function setVar(cssVar, val, lblId, lblVal) {
  document.documentElement.style.setProperty(cssVar, val);
  document.getElementById(lblId).textContent = lblVal;
  savePrefs();
}

function savePrefs() {
  const prefs = {
    theme: document.documentElement.getAttribute("data-theme"),
    cb: document.documentElement.getAttribute("data-cb"),
    font: document.documentElement.getAttribute("data-font"),
    motion: document.documentElement.getAttribute("data-motion"),
    width: document.documentElement.getAttribute("data-width"),
    readingLevel: document.getElementById("reading-level").value,
    safeMode: document.getElementById("safe-mode").value,
    uiLang: document.getElementById("ui-lang").value,
    speechRate: speechRate,
    showAdvisoryBanners: showAdvisoryBanners,
    rulerColor: document.getElementById("ruler-color")?.value || "#D4AF37",
    rulerHeight: document.getElementById("ruler-height")?.value || 36,
    screenTint: document.getElementById("tint-select")?.value || "none",
    focusAnchorOn: focusAnchorOn,
  };
  localStorage.setItem("polycoms_prefs", JSON.stringify(prefs));
}

function loadPrefs() {
  try {
    const p = JSON.parse(localStorage.getItem("polycoms_prefs") || "{}");
    if (p.theme) document.documentElement.setAttribute("data-theme", p.theme);
    if (p.cb) { document.documentElement.setAttribute("data-cb", p.cb); document.getElementById("cb-select").value = p.cb; }
    if (p.font) document.documentElement.setAttribute("data-font", p.font);
    if (p.motion) document.documentElement.setAttribute("data-motion", p.motion);
    if (p.width) document.documentElement.setAttribute("data-width", p.width);
    if (p.readingLevel) document.getElementById("reading-level").value = p.readingLevel;
    if (p.safeMode) document.getElementById("safe-mode").value = p.safeMode;
    if (p.uiLang) { document.getElementById("ui-lang").value = p.uiLang; setUiLang(p.uiLang); }
    if (p.speechRate) { speechRate = p.speechRate; document.getElementById("spd-slider").value = p.speechRate; document.getElementById("spd-label").textContent = p.speechRate + "x"; }
    if (typeof p.showAdvisoryBanners === "boolean") setAdvisoryBanners(p.showAdvisoryBanners);
    if (p.rulerColor && document.getElementById("ruler-color")) { document.getElementById("ruler-color").value = p.rulerColor; }
    if (p.rulerHeight && document.getElementById("ruler-height")) { document.getElementById("ruler-height").value = p.rulerHeight; }
    if (p.rulerColor || p.rulerHeight) updateRulerStyle();
    if (p.screenTint && document.getElementById("tint-select")) { document.getElementById("tint-select").value = p.screenTint; setScreenTint(p.screenTint); }
    if (typeof p.focusAnchorOn === "boolean") focusAnchorOn = p.focusAnchorOn;
  } catch {}
}
loadPrefs();

function clearPrivacyData() {
  localStorage.removeItem("polycoms_prefs");
  document.getElementById("src-text").value = "";
  document.getElementById("tgt-text").value = "";
  document.getElementById("sentence-player").style.display = "none";
  stopAudio();
  alert("Cleared text, audio, and saved preferences.");
}

async function checkStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    document.getElementById("gpu-pill").textContent = "⚡ Backend: " + d.gpu;
  } catch {
    document.getElementById("gpu-pill").textContent = "⚠️ Backend offline";
  }
}
checkStatus();

async function runTranslate() {
  const text = document.getElementById("src-text").value.trim();
  if (!text) return;
  const src_lang = document.getElementById("src-lang").value;
  const tgt_lang = document.getElementById("tgt-lang").value;
  const reading_level = document.getElementById("reading-level").value;
  const safe_mode = document.getElementById("safe-mode").value;
  document.getElementById("tgt-text").value = "Translating with NLLB-200...";

  const res = await fetch("/api/translate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text, src_lang, tgt_lang, reading_level, safe_mode})
  });
  const data = await res.json();
  if (data.error) { document.getElementById("tgt-text").value = data.error; return; }

  const outText = data.translation.primary_translation;
  const outBox = document.getElementById("tgt-text");
  outBox.value = outText;
  outBox.dir = data.target_rtl ? "rtl" : "ltr";

  // Sentence-by-sentence reader chips
  const sents = outText.split(/(?<=[.!?。！？\n])\s+/).map(s => s.trim()).filter(Boolean);
  const sp = document.getElementById("sentence-player");
  const sc = document.getElementById("sentence-chips");
  if (sents.length > 0) {
    sp.style.display = "block";
    sc.innerHTML = sents.map((s, idx) => `<button type="button" class="sent-chip" data-idx="${idx}" data-raw="${s.replace(/"/g,'&quot;')}" onclick="playSingleSentence(${idx})">${focusAnchorOn ? applyFocusAnchor(s) : s}</button>`).join("");
  } else {
    sp.style.display = "none";
  }

  // Confidence & Alternatives
  const conf = data.translation.confidence;
  document.getElementById("conf-val").textContent = conf + "%";
  document.getElementById("conf-bar").style.width = conf + "%";
  document.getElementById("alt-list").innerHTML = data.translation.alternatives.map(a =>
    `<div class="alt-chip" onclick="document.getElementById('tgt-text').value=this.dataset.t" data-t="${a.text.replace(/"/g,'&quot;')}">${a.text} <small>(${a.confidence}%)</small> <span class="pill" style="font-size:0.75em;padding:2px 8px;margin-left:6px;">${a.label || "Alternative"}</span></div>`
  ).join("") || "No additional alternatives.";

  // Banners (Color + Icon + Text)
  const cb = document.getElementById("confusion-banner");
  if (showAdvisoryBanners && data.lid.confusion_note) {
    cb.style.display = "block";
    cb.textContent = "💡 Language Overlap / Did you mean: " + data.lid.confusion_note + ` (Detected: ${formatLangName(data.src_lang)} @ ${data.lid.confidence}%)`;
  } else { cb.style.display = "none"; }

  const sb = document.getElementById("safety-banner");
  if (data.analysis.safety_flags.length > 0) {
    sb.style.display = "block";
    sb.innerHTML = data.analysis.safety_flags.map(f => `⚠️ <strong>${f.category}:</strong> ${f.warning}`).join("<br>");
  } else { sb.style.display = "none"; }

  const pb = document.getElementById("profanity-banner");
  const prof = data.analysis.safe_language?.profanity_detected || [];
  if (showAdvisoryBanners && prof.length > 0) {
    pb.style.display = "block";
    const detail = (safe_mode === "child_friendly" || safe_mode === "censor") ? "" : ` (${prof.join(", ")})`;
    pb.innerHTML = `🛡️ <strong>Strong Language Detected</strong>${detail} — Mode: <em>${safe_mode}</em>. Note: Automated filters can miss slang or context-dependent insults.`;
  } else { pb.style.display = "none"; }

  // Cognitive & Grammar Explanations
  document.getElementById("reg-val").textContent = data.analysis.register;
  const cog = data.analysis.cognitive || {};
  document.getElementById("key-info-box").innerHTML = (cog.key_info || []).map(k => `<div class="alert alert-ok">🔑 ${k}</div>`).join("");
  document.getElementById("glossary-box").innerHTML = (cog.glossary || []).map(g => `<div class="alert">📘 <strong>${g.word}:</strong> ${g.definition}</div>`).join("");
  document.getElementById("idiom-box").innerHTML = data.analysis.idiom_notes.map(i => `<div class="alert">💬 <strong>Idiom ("${i.idiom}"):</strong> ${i.explanation}</div>`).join("");
  document.getElementById("grammar-list").innerHTML = data.analysis.grammar_notes.map(g => `<li>${g}</li>`).join("");
}

function stopAudio() {
  stopSeq = true;
  if (activeAudio) { activeAudio.pause(); activeAudio = null; }
  document.querySelectorAll(".sent-chip").forEach(c => c.classList.remove("speaking"));
}

async function fetchAndPlayTts(text, tgtCode, rateOverride) {
  const r = await fetch("/api/tts", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text, lang: tgtCode})
  });
  if (!r.ok) throw new Error(await r.text() || "HTTP " + r.status);
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  return new Promise((resolve, reject) => {
    activeAudio = new Audio(url);
    activeAudio.playbackRate = rateOverride || speechRate;
    activeAudio.onended = resolve;
    activeAudio.onerror = reject;
    activeAudio.play().catch(reject);
  });
}

async function speakOutput(rateOverride) {
  const txt = document.getElementById("tgt-text").value.trim();
  if (!txt) return;
  stopAudio();
  try { await fetchAndPlayTts(txt, document.getElementById("tgt-lang").value, rateOverride); }
  catch (e) { alert("Speech synthesis error: " + e.message); }
}

const KEYBOARD_SETS = {
  yor_Latn: ["ẹ", "ọ", "ṣ", "á", "à", "é", "è", "ẹ́", "ẹ̀", "í", "ì", "ó", "ò", "ọ́", "ọ̀", "ú", "ù", "gb"],
  deu_Latn: ["ä", "ö", "ü", "ß", "Ä", "Ö", "Ü"],
  swe_Latn: ["å", "ä", "ö", "Å", "Ä", "Ö"],
  fin_Latn: ["ä", "ö", "å", "Ä", "Ö", "Å"],
  nob_Latn: ["æ", "ø", "å", "Æ", "Ø", "Å"],
  fra_Latn: ["é", "è", "ê", "ë", "à", "â", "ç", "î", "ï", "ô", "ù", "û", "œ"],
  spa_Latn: ["á", "é", "í", "ó", "ú", "ü", "ñ", "¿", "¡"],
  mri_Latn: ["ā", "ē", "ī", "ō", "ū", "Ā", "Ē", "Ī", "Ō", "Ū", "ʻ"],
  smo_Latn: ["ā", "ē", "ī", "ō", "ū", "ʻ"],
  jpn_Jpan: ["あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ", "た", "ち", "つ", "て", "と", "な", "に", "の", "は", "ま", "ら", "ん", "。", "、", "？"],
  kor_Hang: ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅎ", "ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ", "ㅐ", "ㅔ"],
  rus_Cyrl: ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ы", "э", "ю", "я"],
  ukr_Cyrl: ["а", "б", "в", "г", "ґ", "д", "е", "є", "ж", "з", "и", "і", "ї", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ь", "ю", "я"],
  arb_Arab: ["ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ر", "س", "ش", "ص", "ع", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", "؟"],
  default: ["a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú", "ñ", "ç", "?", "!"]
};

function toggleKeyboard() {
  const kbd = document.getElementById("aria-keyboard");
  const btn = document.getElementById("kbd-toggle-btn");
  const open = kbd.style.display === "none";
  kbd.style.display = open ? "block" : "none";
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) renderKeyboard();
}

function renderKeyboard() {
  const kbd = document.getElementById("aria-keyboard");
  if (kbd.style.display === "none") return;
  const lang = document.getElementById("src-lang").value;
  const keys = KEYBOARD_SETS[lang] || KEYBOARD_SETS.default;
  kbd.innerHTML = keys.map(k =>
    `<button type="button" class="btn btn-secondary btn-icon" style="margin:2px;padding:6px 10px;" aria-label="Insert character ${k}" onclick="insertKbdChar('${k}')">${k}</button>`
  ).join("") +
  `<button type="button" class="btn btn-secondary btn-icon" style="margin:2px;" aria-label="Insert space" onclick="insertKbdChar(' ')">Space</button>` +
  `<button type="button" class="btn btn-secondary btn-icon" style="margin:2px;" aria-label="Backspace" onclick="backspaceKbd()">⌫</button>` +
  `<button type="button" class="btn btn-secondary btn-icon" style="margin:2px;" aria-label="Clear input text" onclick="document.getElementById('src-text').value=''">Clear</button>`;
}

function insertKbdChar(ch) {
  const ta = document.getElementById("src-text");
  const start = ta.selectionStart || ta.value.length;
  const end = ta.selectionEnd || ta.value.length;
  ta.value = ta.value.slice(0, start) + ch + ta.value.slice(end);
  ta.focus();
  ta.selectionStart = ta.selectionEnd = start + ch.length;
}

function backspaceKbd() {
  const ta = document.getElementById("src-text");
  ta.value = ta.value.slice(0, -1);
  ta.focus();
}

async function playSingleSentence(idx) {
  const chips = document.querySelectorAll(".sent-chip");
  if (!chips[idx]) return;
  stopAudio();
  chips[idx].classList.add("speaking");
  try { await fetchAndPlayTts(chips[idx].textContent, document.getElementById("tgt-lang").value); }
  catch (e) { alert("Speech synthesis error: " + e.message); }
  chips[idx].classList.remove("speaking");
}

async function playSentences() {
  const chips = Array.from(document.querySelectorAll(".sent-chip"));
  if (!chips.length) return speakOutput();
  stopAudio();
  stopSeq = false;
  const tgtCode = document.getElementById("tgt-lang").value;
  for (const chip of chips) {
    if (stopSeq) break;
    chip.classList.add("speaking");
    try { await fetchAndPlayTts(chip.textContent, tgtCode); }
    catch (e) { alert("Speech synthesis error: " + e.message); break; }
    chip.classList.remove("speaking");
  }
}

async function runOcr(input, mode) {
  const file = input.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("image", file, file.name);
  const ocrLang = mode === "lid"
    ? (document.getElementById("lid-ocr-script")?.value || "auto")
    : (document.getElementById("src-lang")?.value || "auto");
  fd.append("ocr_lang", ocrLang);

  if (mode === "translate") {
    document.getElementById("src-text").value = "Extracting text from image with EasyOCR...";
  } else {
    document.getElementById("lid-input").value = "Extracting text from image with EasyOCR...";
  }

  try {
    const r = await fetch("/api/ocr", { method: "POST", body: fd });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    if (mode === "translate") {
      document.getElementById("src-text").value = d.text || "";
      if (d.text) runTranslate();
    } else {
      document.getElementById("lid-input").value = d.text || "";
      if (d.text) {
        await runLidPredict();
        document.getElementById("lid-result").innerHTML += `<div class="alert">📷 <strong>OCR Confidence:</strong> ${d.ocr_confidence}%</div>`;
      }
    }
  } catch (e) {
    const msg = "OCR error: " + e.message;
    if (mode === "translate") document.getElementById("src-text").value = msg;
    else document.getElementById("lid-input").value = msg;
  }
  input.value = "";
}

async function uploadAudio(input) {
  const file = input.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("audio", file, file.name);
  fd.append("spoken_lang", document.getElementById("stt-lang").value);
  document.getElementById("src-text").value = "Transcribing audio with Whisper-small...";
  try {
    const r = await fetch("/api/stt", { method: "POST", body: fd });
    const d = await r.json();
    document.getElementById("src-text").value = d.transcript || "";
  } catch (e) {
    document.getElementById("src-text").value = "Audio upload error: " + e.message;
  }
  input.value = "";
}

let mediaRecorder = null, audioChunks = [], visRaf = null;
async function toggleMic() {
  const btn = document.getElementById("mic-btn");
  const visWrap = document.getElementById("mic-vis-wrap");
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    btn.textContent = "🎙️ Speak";
    visWrap.style.display = "none";
    if (visRaf) cancelAnimationFrame(visRaf);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // Audio level visualisation
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = ctx.createAnalyser();
    ctx.createMediaStreamSource(stream).connect(analyser);
    const dataArr = new Uint8Array(analyser.frequencyBinCount);
    visWrap.style.display = "block";
    function drawMeter() {
      analyser.getByteFrequencyData(dataArr);
      const avg = dataArr.reduce((a,b) => a+b, 0) / dataArr.length;
      document.getElementById("mic-vis-bar").style.width = Math.min(100, Math.round(avg * 1.8)) + "%";
      visRaf = requestAnimationFrame(drawMeter);
    }
    drawMeter();

    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      const fd = new FormData();
      fd.append("audio", blob, "speech.webm");
      fd.append("spoken_lang", document.getElementById("stt-lang").value);
      document.getElementById("src-text").value = "Transcribing with Whisper-small...";
      const r = await fetch("/api/stt", { method: "POST", body: fd });
      const d = await r.json();
      document.getElementById("src-text").value = d.transcript || "";
    };
    mediaRecorder.start();
    btn.textContent = "⏹️ Stop";
  } catch (e) {
    alert("Microphone access unavailable in this browser frame: " + e.message);
  }
}

async function runJudgeDemo() {
  const box = document.getElementById("judge-demo-out");
  box.innerHTML = "Generating live 5-mode accessibility comparison with NLLB-200...";
  const sample = "The implementation of the aforementioned medication is contingent upon eating first. That is damn important. Do not drive if dizzy.";
  const tgt = document.getElementById("tgt-lang").value;
  const [stdRes, simpRes, safeRes] = await Promise.all([
    fetch("/api/translate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text:sample, src_lang:"eng_Latn", tgt_lang:tgt, reading_level:"standard", safe_mode:"preserve"})}).then(r=>r.json()),
    fetch("/api/translate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text:sample, src_lang:"eng_Latn", tgt_lang:tgt, reading_level:"steps", safe_mode:"preserve"})}).then(r=>r.json()),
    fetch("/api/translate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text:sample, src_lang:"eng_Latn", tgt_lang:tgt, reading_level:"simple", safe_mode:"child_friendly"})}).then(r=>r.json()),
  ]);
  box.innerHTML = `
    <div class="alert"><strong>1. Standard Translation:</strong> ${stdRes.translation.primary_translation}</div>
    <div class="alert alert-ok"><strong>2. Easy-Read / Step-by-Step:</strong><br/>${simpRes.translation.primary_translation.replace(/\n/g,"<br/>")}</div>
    <div class="alert"><strong>3. Child-Friendly Safe Language Mode:</strong> ${safeRes.translation.primary_translation}</div>
    <div class="alert" style="font-family:'Atkinson Hyperlegible',Verdana,sans-serif;letter-spacing:0.05em;line-height:1.8;"><strong>4. Dyslexia-Friendly Preview:</strong> ${simpRes.translation.primary_translation}</div>
    <div class="alert"><strong>5. Synchronized Sentence Audio:</strong> Switch to Translate tab and click <em>⏯️ Sentences</em> to hear & highlight each sentence!</div>
  `;
}

async function runLidPredict() {
  const text = document.getElementById("lid-input").value.trim();
  if (!text) return;
  const r = await fetch("/api/lid/predict", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({text}) });
  const d = await r.json();
  document.getElementById("lid-result").innerHTML = `
    <div class="alert alert-ok">
      <strong>TF-IDF Model:</strong> ${formatLangName(d.predicted_iso3)} (${d.confidence}%) &nbsp;|&nbsp;
      <strong>langdetect Baseline:</strong> ${d.langdetect_comparison.map(x => `${x.lang} (${x.prob}%)`).join(", ")}
      ${d.confusion_note ? `<br>⚠️ ${d.confusion_note}` : ""}
    </div>`;
}

async function loadLidMetrics() {
  const r = await fetch("/api/lid/metrics");
  const d = await r.json();
  let html = `<p>Dataset: <strong>${d.dataset_source}</strong> (${d.dataset_size} sentences) · Overall TF-IDF Accuracy: <strong>${d.overall_tfidf_accuracy}%</strong></p>`;
  html += `<table><tr><th>Language</th><th>Test Samples</th><th>TF-IDF Accuracy</th><th>langdetect Accuracy</th></tr>`;
  d.per_language.forEach(row => {
    html += `<tr><td>${formatLangName(row.code)}</td><td>${row.n}</td><td><strong>${row.tfidf_accuracy}%</strong></td><td>${row.langdetect_accuracy}%</td></tr>`;
  });
  html += `</table>`;
  document.getElementById("lid-metrics-box").innerHTML = html;
}

async function runBenchmark() {
  const selected = Array.from(document.querySelectorAll("#chrf-lang-checks input:checked")).map(c => c.value);
  const query = selected.length ? "?langs=" + encodeURIComponent(selected.join(",")) : "";
  document.getElementById("bench-table").innerHTML = "Running NLLB-200 on FLORES+ samples...";
  const r = await fetch("/api/quality/benchmark" + query);
  const d = await r.json();
  document.getElementById("bench-summary").innerHTML = Object.values(d.summary).map(s =>
    `<div class="pill">${s.name} (${s.tier}): chrF ${s.corpus_chrf}</div>`
  ).join("");
  let tbl = `<table><tr><th>Domain</th><th>Language</th><th>chrF</th><th>Model Output</th><th>Professional Reference</th></tr>`;
  d.examples.forEach(ex => {
    tbl += `<tr><td>${ex.domain}</td><td>${ex.target_lang}</td><td><strong>${ex.chrf}</strong></td><td>${ex.model_output}</td><td>${ex.reference}</td></tr>`;
  });
  tbl += `</table>`;
  document.getElementById("bench-table").innerHTML = tbl;
}

async function checkUserChrf() {
  const hypothesis = document.getElementById("user-hyp").value;
  const reference = document.getElementById("user-ref").value;
  const r = await fetch("/api/quality/chrf", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({hypothesis, reference}) });
  const d = await r.json();
  document.getElementById("user-chrf-out").textContent = "chrF Score: " + d.chrf;
}

function testSafetySample(sample) {
  switchTab("translate", document.querySelector(".nav-btn"));
  document.getElementById("src-text").value = sample;
  runTranslate();
}
</script>
</body>
</html>
"""

# ==============================================================================
# 8. LAUNCH SERVER (Google Colab Iframe / Window + Local HTML Export)
# ==============================================================================
def find_free_port(start_port: int = 8000) -> int:
    import socket
    for p in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port

def start_server(port: int = 8000):
    port = find_free_port(port)
    # Also save standalone HTML snapshot for inspection
    with open("veritas_languages_app.html", "w", encoding="utf-8") as f:
        f.write(EMBEDDED_HTML)
    print("[Info] Saved local HTML copy to veritas_languages_app.html")

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1.5)

    cf_proc = None
    try:
        from google.colab import output  # type: ignore[import-not-found]
        print(f"[Colab] Starting HTTPS tunnel for PolyComs on port {port}...")
        cf_bin = "/tmp/cloudflared"
        if not os.path.exists(cf_bin):
            subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", cf_bin], check=True)
            os.chmod(cf_bin, 0o755)
        cf_proc = subprocess.Popen([cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        public_url = None
        for _ in range(30):
            line = cf_proc.stdout.readline()
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if m:
                public_url = m.group(0)
                break
        if public_url:
            print(f"\n>>> OPEN THIS LINK IN A NEW TAB (Microphone + Full UI work here): {public_url}\n")
        else:
            print("[Colab] Tunnel URL timed out; using inline iframe below.")
        output.serve_kernel_port_as_iframe(port, height=780)
    except ImportError:
        print(f"[Local] Open http://127.0.0.1:{port} in your browser.")

    # Keep main thread alive in Colab/script mode and clean up on Stop
    try:
        while thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        server.should_exit = True
        if cf_proc is not None:
            cf_proc.terminate()

if __name__ == "__main__":
    start_server(8000)
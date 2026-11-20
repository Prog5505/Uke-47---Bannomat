import os
import re
import subprocess
import sys
import tempfile
import wave

import numpy as np

# Konstanter
MODELL = "base"        # Prøv "small" dersom maskinen er rask nok
SPRAAK = "no"          # "no" for norsk, "en" for engelsk
OPPTAK_SEKUNDER = 10   # Endres til 30 når alt er klart
SAMPLERATE = 16000
MAC_MIKROFON = ":0"    # AVFoundation-nummer; endres ved behov

# Mikrofon og AI
_modell = None


def hent_modell():
    # Dette kan ta litt tid første gang
    global _modell
    if _modell is None:
        from faster_whisper import WhisperModel
        print("Laster Faster-Whisper...")
        _modell = WhisperModel(MODELL, device="cpu", compute_type="int8")
    return _modell


def transkriber(sekunder=OPPTAK_SEKUNDER):
    # Tar opp lyd og returnerer gjenkjent tekst
    print(f"Opptak starter nå ({sekunder} sekunder). Snakk normalt.")

    if sys.platform == "darwin":
        # macOS: FFmpeg bruker AVFoundation i stedet for PortAudio
        with tempfile.TemporaryDirectory() as mappe:
            lydfil = os.path.join(mappe, "opptak.wav")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "avfoundation", "-i", MAC_MIKROFON,
                    "-t", str(sekunder),
                    "-ar", str(SAMPLERATE), "-ac", "1",
                    "-c:a", "pcm_s16le", lydfil,
                ],
                check=True,
            )

            # Les WAV uten PyAV: unngår metadata_errors-feilen
            with wave.open(lydfil, "rb") as wav:
                if (wav.getframerate() != SAMPLERATE
                        or wav.getnchannels() != 1
                        or wav.getsampwidth() != 2
                        or wav.getcomptype() != "NONE"):
                    raise ValueError("Forventer WAV: mono, 16 kHz, 16-bit PCM")
                lyddata = wav.readframes(wav.getnframes())

            lyd = np.frombuffer(lyddata, dtype="<i2").astype(np.float32) / 32768.0
    else:
        # Windows og Linux: lyddata sendes direkte som NumPy-array
        import sounddevice as sd
        opptak = sd.rec(
            int(sekunder * SAMPLERATE),
            samplerate=SAMPLERATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        lyd = opptak[:, 0].copy()

    print("Analyserer tale ...")
    segmenter, _ = hent_modell().transcribe(
        lyd,
        language=SPRAAK,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return " ".join(segment.text.strip() for segment in segmenter).strip()


# Liste med norske og engelske banneord
BANNEORD = [
    # Norske
    "faen", "helvete", "dritt", "jævla", "jævlig",
    "jævel", "satan", "forbanna", "forbannet",
    "pokker", "søren", "drit", "drittsekk",
    "ræv", "ræva", "rass", "rasshøl",
    "kuk", "fitte",

    # Engelske
    "fuck", "fucking", "fucker", "motherfucker",
    "shit", "bullshit", "damn", "goddamn",
    "ass", "asshole", "arse", "arsehole",
    "bitch", "bastard", "crap", "douchebag",
    "jerk", "prick", "slut", "whore",
    "wanker", "twat", "bollocks", "bugger"
]


def finn_banneord(tekst):
    ordliste = re.findall(r"\b\w+\b", tekst.lower())

    treff = []

    for ordet in ordliste:
        if ordet in BANNEORD:
            treff.append(ordet)

    return treff


def beregn_poeng(treff):
    totale_treff = len(treff)
    unike_treff = len(set(treff))

    return totale_treff + unike_treff


def vis_resultat(navn, tekst, treff, poeng):
    print("\n" + "=" * 30)
    print("      BANNOMETER 3000")
    print("=" * 30)

    print(f"Deltaker: {navn}")
    print(f"Tekst: {tekst}")

    print(f"\nAntall banneord: {len(treff)}")
    print(f"Unike banneord: {len(set(treff))}")

    if treff:
        print("Banneord funnet:", ", ".join(treff))
    else:
        print("Ingen banneord funnet.")

    print(f"\nPoengsum: {poeng}")

    if poeng >= 10:
        print(" Banneekspert!")
    elif poeng >= 5:
        print(" Ganske kreativ banning!")
    else:
        print(" Rolig språkbruk!")

    print("=" * 30)


def test_better_profanity(tekst):
    # Sammenlign den engelske detektoren med egen ordliste
    from better_profanity import profanity

    profanity.load_censor_words(BANNEORD)

    print("\n--- BETTER-PROFANITY ---")
    print("Engelsk bibliotek oppdaget banning:", profanity.contains_profanity(tekst))
    print("Sensurert versjon:", profanity.censor(tekst))


def hovedprogram():
    print("=== BANNOMETER 3000 ===")
    print("1: Skriv inn testtekst (uten mikrofon)")
    print("2: Snakk i mikrofonen")
    valg = input("Velg 1 eller 2: ").strip()
    if valg not in ("1", "2"):
        print("Ugyldig valg.")
        return

    navn = input("Deltakernavn (frivillig/pseudonym): ").strip() or "Testperson"
    if valg == "1":
        tekst = input("Skriv testtekst: ").strip()
    else:
        input("Trykk ENTER når du er klar ...")
        try:
            tekst = transkriber()
        except Exception as feil:
            print("Mikrofon eller AI kunne ikke brukes:", feil)
            print("På Mac må FFmpeg være installert og mikrofonnummeret være riktig.")
            print("Sjekk mikrofontillatelse, eller test tekstmodus i mellomtiden.")
            return

    print("\nGjenkjent tekst:", tekst or "(ingen tale gjenkjent)")
    treff = finn_banneord(tekst)
    poeng = beregn_poeng(treff)
    vis_resultat(navn, tekst, treff, poeng)
    

    # Kjøres automatisk etter hver analyse
    try:
        test_better_profanity(tekst)
    except ImportError:
        print("\nbetter-profanity er ikke installert. Kjør: pip install better-profanity")


if __name__ == "__main__":
    hovedprogram()

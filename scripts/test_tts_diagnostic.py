import os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AudioConfig
from audio.speech import SpeechEngine

def run_diagnostic() -> bool:
    print('=' * 55)
    print('        NaviGlass Desktop - Standalone TTS Diagnostic')
    print('=' * 55)
    print(f'Project Root: {PROJECT_ROOT}')
    print(f'Python:       {sys.version}')
    print('-' * 55)

    print('STAGE 1: Testing Windows SAPI5 / pyttsx3 driver binding...')
    try:
        import pyttsx3
        direct_engine = pyttsx3.init()
        voices = direct_engine.getProperty('voices')
        rate = direct_engine.getProperty('rate')
        vol = direct_engine.getProperty('volume')
        print(' -> Driver initialized successfully.')
        print(f' -> Available Voices ({len(voices)}):')
        for idx, v in enumerate(voices):
            print(f'    [{idx}] ID: {v.id}')
            print(f'        Name: {v.name}')
        print(f' -> Default Speech Rate: {rate} wpm')
        print(f' -> Default Volume:      {vol * 100:.0f}%')
        direct_engine.stop()
        del direct_engine
    except Exception as e:
        print(f'[ERROR] Failed to inspect SAPI5: {e}', file=sys.stderr)
        return False

    print('\nSTAGE 2: Initializing NaviGlass SpeechEngine worker...')
    cfg = AudioConfig(enabled=True, speech_rate_wpm=180, volume=1.0)
    engine = SpeechEngine(cfg)
    try:
        engine.start()
        print(' -> Speech worker thread started.')
        print(f' -> in_use available: {engine.is_available}')
        print(f' -> is_enabled:       {engine.is_enabled}')
        print(f' -> is_muted:         {engine.is_muted}')
        if not engine.is_available:
            print('[ERROR] SpeechEngine reported unavailable state.', file=sys.stderr)
            engine.stop()
            return False
    except Exception as e:
        print(f'[ERROR] SpeechEngine.start() raised: {e}', file=sys.stderr)
        return False

    test_phrases = [
        'NaviGlass audio test.',
        'Person on your left.',
        'Chair in front of you.',
    ]

    print('\nSTEGE 3: Executing sequential speech alerts...')
    overall_success = True

    for i, phrase in enumerate(test_phrases, start=1):
        print(f'\n--- Utterance {i}/{len(test_phrases)}: "{phrase}" ---')
        t_start = time.time()
        try:
            enqueued = engine.speak_async(phrase)
            if not enqueued:
                print(f' -> [FAIL] speak_async returned False for "{phrase}"')
                overall_success = False
                continue
            print(' -> speak_async enqueued. Waiting for completion...')
            engine.wait_until_done(timeout=8.0)
            elapsed = time.time() - t_start
            print(f' -> Completed in {elapsed:.2f} seconds.')
            time.sleep(0.5)
        except Exception as ex:
            print(f' -> [EXCEPTIOND During utterance "{phrase}": {ex}', file=sys.stderr)
            overall_success = False

    print('\nSTAGE 4: Releasing audio resources and shutting down worker...')
    try:
        engine.stop()
        print(' -> Speech worker stopped cleanly.')
    except Exception as e:
        print(f'[ERROR] Failed to stop speech worker cleanly: {e}', file=sys.stderr)
        overall_success = False

    print('-' * 55)
    if overall_success:
        print('Standalone TTS Diagnostic: ALL STAGES PASSED')
    else:
        print('Standalone TTS Diagnostic: FAILURES DETECTED')
    print('=' * 55)
    return overall_success

if __name__ == '__main__':
    ok = run_diagnostic()
    sys.exit(0 if ok else 1)

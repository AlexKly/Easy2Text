import torch, librosa
from Easy2Text import ASRWrapper


class Easy2Text:
    def __init__(self, asr_modelname, grammar_corrector_modelname, device='cpu', use_grammar_corrector=True,
                 verbose=-1):
        # Common parameters:
        self.device = device
        # Init models:
        self.asr_model = ASRWrapper(modelname=asr_modelname, device=device, verbose=verbose)
        if use_grammar_corrector:
            pass
        else:
            self.grammar_corrector_model = None

    def load_samples(self, audioname):
        samples, _ = librosa.load(path=audioname, sr=16000)
        return torch.tensor(data=samples, device=self.device)

    def apply_easy2text(self, audioname, lang=None):
        # Stage 0: Load samples
        audio = self.load_samples(audioname=audioname)
        # Stage 1: Apply ASR to convert speech to text:
        if lang is None:
            lang = self.asr_model.detect_lang(samples=audio)
        text = self.asr_model.get_transcription(samples=audio, lang=lang)
        # Stage 2: Apply Grammar Corrector to get corrected text (if grammar corrected is selected):
        if self.grammar_corrector_model is None:
            return text
        else:
            pass

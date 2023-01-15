import logging
from Easy2Text import Configurations, ASRWrapper, GrammarCorrector

logging.basicConfig(format=Configurations.format_log, level=logging.INFO)
logging.getLogger('Logger is on.')


class Easy2Text:
    def __init__(self, asr_modelname, use_grammar_corrector=True, device='cpu', verbose=-1):
        # Common parameters:
        self.device = device
        self.verbose = verbose
        # Init models:
        self.asr_model = ASRWrapper(modelname=asr_modelname, device=self.device, verbose=self.verbose)
        if use_grammar_corrector:
            self.grammar_corrector_model = GrammarCorrector(device=self.device, verbose=self.verbose)
        else:
            self.grammar_corrector_model = None

    def apply_easy2text(self, audioname, lang=None):
        # Stage 0: Load samples
        audio = self.asr_model.load_samples(audioname=audioname)
        # Stage 1: Apply ASR to convert speech to text:
        if lang is None:
            lang = self.asr_model.detect_lang(samples=audio)
        text = self.asr_model.get_transcription(samples=audio, lang=lang)
        # Stage 2: Apply Grammar Corrector to get corrected text (if grammar corrected is selected):
        if self.grammar_corrector_model is not None:
            return self.grammar_corrector_model.get_corrected(text=text)
        else:
            return text

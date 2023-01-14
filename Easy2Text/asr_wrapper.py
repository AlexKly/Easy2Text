import os, logging, whisper
from Easy2Text import Configurations, Downloader


class ASRWrapper:
    def __init__(self, modelname, device='cpu', verbose=-1):
        # Common parameters:
        self.device = device
        self.verbose = verbose
        if self.verbose: logging.info(f'Initialization ASR model -->')
        # Init model:
        if os.path.isfile(Configurations.dir_models/f'{modelname}.pt'):
            if self.verbose: logging.info(f'Load downloaded model -->')
            self.asr_model = whisper.load_model(name=Configurations.dir_models/f'{modelname}.pt', device=self.device)
        else:
            if self.verbose: logging.info(f'Start downloading model to {Configurations.dir_models} -->')
            self.asr_model = Downloader().download_asr(modelname=modelname, device=self.device)

    def detect_lang(self, samples):
        mel = whisper.log_mel_spectrogram(samples).to(self.device)
        _, probs = self.asr_model.detect_language(mel)
        if self.verbose: logging.info(f'Detected language --> {probs[0]}')
        return probs[0]

    def get_transcription(self, samples, lang):
        if self.verbose: logging.info(f'Start transcription -->')
        return self.asr_model.transcribe(audio=samples, language=lang)['text']

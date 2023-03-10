import whisper, transformers
from Easy2Text import Configurations


class Downloader(Configurations):
    def download_asr(self, modelname, device):
        return whisper.load_model(name=modelname, device=device, download_root=self.dir_models)

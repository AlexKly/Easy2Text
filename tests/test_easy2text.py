import unittest
from pathlib import Path
from Easy2Text import Easy2Text, ASRWrapper, GrammarCorrector

test_filepath = Path(__file__).parent.parent.resolve()
# Init Easy2Text:
e2t = Easy2Text(asr_modelname='tiny', use_grammar_corrector=True, device='cpu', verbose=1)
# Init ASR Wrapper:
asr = ASRWrapper(modelname='tiny', device='cpu', verbose=1)
# Init Grammar Corrector:
gc = GrammarCorrector(device='cpu', verbose=1)


class TestEasy2Text(unittest.TestCase):
    # Tests for Easy2Text:
    def test_a_e2t_u0(self): print(e2t.apply_easy2text(audioname=test_filepath/'data/test.wav', lang='en'))
    def test_a_e2t_u1(self): print(e2t.apply_easy2text(audioname=test_filepath/'data/test.wav', lang=None))
    # Tests for ASRWrapper:
    # Tests for Grammar Corrector:
    def test_g_c_u0(self): print(gc.get_corrected(text='I were here'))  # Wrong case

import re, logging, transformers


class GrammarCorrector:
    def __init__(self, device, verbose):
        # Common parameters:
        self.device = device
        self.verbose = verbose
        if self.verbose: logging.info(f'Initialization Grammar Corrector model -->')
        # Init Grammar Checker model:
        self.grammar_checker_model = transformers.pipeline("text-classification", "textattack/roberta-base-CoLA")
        # Init Grammar Corrector model:
        self.grammar_corrector_model = transformers.pipeline(
            "text2text-generation",
            "pszemraj/flan-t5-large-grammar-synthesis"
        )

    @staticmethod
    def split_text(text):
        return re.split(r'; |. |! |? ', text)

    def correct_text(self, text, checker, corrector):
        sentence_batches = self.split_text(text)
        corrected_text = []
        for batch in sentence_batches:
            raw_text = " ".join(batch)
            results = checker(raw_text)
            if results[0]["label"] != "LABEL_1" or (results[0]["label"] == "LABEL_1" and results[0]["score"] < 0.9):
                corrected_batch = corrector(raw_text)
                corrected_text.append(corrected_batch[0]["generated_text"])
                print(corrected_batch)
            else:
                corrected_text.append(raw_text)
        corrected_text = '.'.join(corrected_text)

        return corrected_text

    def get_corrected(self, text):
        logging.info(f'Start grammar correction -->')
        return self.correct_text(text=text, checker=self.grammar_checker_model, corrector=self.grammar_corrector_model)

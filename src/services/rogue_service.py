from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

class RougeService:
    def __init__(self):
        pass

    def calculate_score(self, generated_text, reference_text):
        # Tokenize the texts
        reference_tokens = word_tokenize(reference_text)
        generated_tokens = word_tokenize(generated_text)
        
        # Calculate BLEU score as an approximation of ROUGE
        score = sentence_bleu([reference_tokens], generated_tokens)
        return score

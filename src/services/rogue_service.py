from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)

class RogueService:
    def __init__(self):
        pass

    def calculate_score(self, generated_text, reference_text):
        try:
            # Tokenize the texts
            reference_tokens = word_tokenize(reference_text)
            generated_tokens = word_tokenize(generated_text)
            
            # Calculate BLEU score as an approximation of ROUGE
            score = sentence_bleu([reference_tokens], generated_tokens)
            return score
        except LookupError as e:
            logger.error(f"NLTK resource error: {e}")
            logger.info("Falling back to simple whitespace tokenization")
            # Fallback to simple whitespace tokenization
            reference_tokens = reference_text.split()
            generated_tokens = generated_text.split()
            score = sentence_bleu([reference_tokens], generated_tokens)
            return score
        except Exception as e:
            logger.error(f"Error calculating ROUGE score: {e}")
            return 0.0  # Return 0 score in case of any other error

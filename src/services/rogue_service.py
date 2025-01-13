import re
from collections import Counter
from typing import List

class RogueService:
    def __init__(self):
        pass
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on spaces and removing punctuation"""
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def calculate_score(self, generated_text: str, reference_text: str) -> float:
        """
        Simplified scoring using word overlap and repetition penalty
        Returns a score between 0 and 1
        """
        try:
            # Tokenize both texts
            gen_tokens = self._tokenize(generated_text)
            ref_tokens = self._tokenize(reference_text)
            
            if not gen_tokens or not ref_tokens:
                return 0.0
            
            # Calculate word overlap
            gen_counter = Counter(gen_tokens)
            ref_counter = Counter(ref_tokens)
            
            common_words = set(gen_counter.keys()) & set(ref_counter.keys())
            
            if not common_words:
                return 0.0
            
            # Calculate overlap score
            overlap_score = len(common_words) / max(len(gen_counter), len(ref_counter))
            
            # Calculate repetition penalty
            unique_ratio = len(set(gen_tokens)) / len(gen_tokens)
            
            # Combine scores
            final_score = overlap_score * unique_ratio
            
            return min(1.0, final_score)
            
        except Exception as e:
            print(f"Error in calculate_score: {e}")
            return 0.0
    
    def calculate_advanced_score(self, generated_text: str, reference_text: str) -> float:
        """
        Wrapper for calculate_score to maintain API compatibility
        """
        return self.calculate_score(generated_text, reference_text)

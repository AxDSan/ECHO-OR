from dataclasses import dataclass

@dataclass
class Demonstration:
    question: str
    answer: str
    rationale: str
    rouge_score: float = 0.0

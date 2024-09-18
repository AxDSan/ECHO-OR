from dataclasses import dataclass

@dataclass
class Demonstration:
    question: str
    answer: str
    rationale: str
    rogue_score: float = 0.0

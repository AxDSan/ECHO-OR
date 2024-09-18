import random
from typing import List, Dict
from src.services.embedding_service import EmbeddingService
from src.services.llm_service import LLMService
from src.services.rogue_service import RogueService
from src.utils.config import Config

REFINEMENT_PROMPT_TEMPLATE = """Based on the following Q&A pairs:
{demonstrations}
Refine the rationale for the question below.
Question: {question}
Current Rationale: {current_rationale}
Improved Rationale:"""

class DemonstrationRefiner:
    def __init__(self, embedding_service: EmbeddingService, llm_service: LLMService, rogue_service: RogueService):
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.rogue_service = rogue_service

    async def refine_demonstrations(self, demonstrations: List[Dict[str, str]], num_iterations: int) -> List[Dict[str, str]]:
        for iteration in range(num_iterations):
            print(f"Refinement Iteration {iteration + 1}/{num_iterations}")
            random.shuffle(demonstrations)
            
            refinement_prompts = []
            for demo in demonstrations:
                other_demos = [d for d in demonstrations if d != demo]
                demonstrations_str = "\n".join([f"Q: {d['question']}\nA: {d['rationale']}" for d in other_demos])
                prompt = REFINEMENT_PROMPT_TEMPLATE.format(
                    demonstrations=demonstrations_str,
                    question=demo['question'],
                    current_rationale=demo['rationale']
                )
                refinement_prompts.append(prompt)
            
            refined_rationales = await self._generate_refined_rationales(refinement_prompts)
            
            for idx, (demo, new_rationale) in enumerate(zip(demonstrations, refined_rationales)):
                print(f"Processing demonstration {idx + 1}")
                if not new_rationale:
                    print(f"Empty rationale generated for question: {demo['question']}")
                    continue
                
                current_length = len(demo['rationale'])
                new_length = len(new_rationale)
                
                if 50 <= new_length <= Config.MAX_TOKENS*4 and new_length > current_length:
                    demo['rationale'] = new_rationale[:Config.MAX_TOKENS*4]
                    print(f"Updated rationale for demonstration {idx + 1}")
                else:
                    print(f"Kept original rationale for demonstration {idx + 1}")
                
                print(f"Updated demonstration: {demo}")
        
        return demonstrations

    async def _generate_refined_rationales(self, prompts: List[str]) -> List[str]:
        rationales = []
        for prompt in prompts:
            rationale = await self.llm_service.generate_text(
                prompt, 
                max_tokens=Config.MAX_TOKENS, 
                temperature=0.7, 
                top_p=0.95, 
                repetition_penalty=1.2
            )
            rationales.append(rationale)
        return rationales

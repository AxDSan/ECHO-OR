import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Union
from sklearn.metrics.pairwise import cosine_similarity

from src.services.embedding_service import EmbeddingService
from src.services.llm_service import LLMService
from src.services.rogue_service import RogueService
from src.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services at module level
embedding_service = EmbeddingService()
llm_service = LLMService()
rogue_service = RogueService()

async def fetch_most_relevant_demonstrations(question: str, k=5) -> List[Dict]:
    """Optimized version with batch processing"""
    logger.info("Starting to fetch demonstrations...")
    
    # Get question embedding - ensure it's 2D
    question_vec = embedding_service.get_embeddings(question)
    if question_vec.ndim == 3:
        question_vec = question_vec.reshape(question_vec.shape[0], -1)
    
    # Batch process all demonstrations at once
    demo_questions = [item["question"] for item in Config.dataset]
    demo_vecs = embedding_service.get_embeddings(demo_questions)
    if demo_vecs.ndim == 3:
        demo_vecs = demo_vecs.reshape(demo_vecs.shape[0], -1)
    
    logger.debug(f"Question vector shape: {question_vec.shape}")
    logger.debug(f"Demo vectors shape: {demo_vecs.shape}")
    
    # Compute similarities in one go
    similarities = cosine_similarity(question_vec, demo_vecs)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    logger.info(f"Found {k} most relevant demonstrations")
    return [Config.dataset[i] for i in top_indices]

async def generate_chain_of_thought(question: str, demonstrations: List[Dict]) -> str:
    logger.info("Generating chain of thought...")
    
    # Build the prompt
    demo_texts = []
    for demo in demonstrations:
        demo_texts.append(
            f"Q: {demo['question']}\n"
            f"Chain-of-Thought: {demo['initial_rationale']}\n"
            f"A: {demo['answer']}\n"
        )
    
    prompt = (
        "Below are several Q&A pairs with step-by-step reasoning.\n\n"
        + "\n".join(demo_texts)
        + f"\nQ: {question}\n"
        + "Chain-of-Thought:"
    )
    
    response = await llm_service.generate_text(
        prompt=prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=0.0
    )
    
    logger.info("Chain of thought generated")
    return response

async def self_consistency_final_answer(question: str, chain_of_thought: str, attempts: int = 3) -> str:
    logger.info("Generating multiple final answers for consistency...")
    
    # Truncate chain_of_thought if too long (keeping the last part)
    max_cot_length = 500
    if len(chain_of_thought) > max_cot_length:
        chain_of_thought = "..." + chain_of_thought[-max_cot_length:]
    
    prompt_template = f"""
Based on the reasoning:
{chain_of_thought}

Give a very concise final answer (1-2 sentences max) to:
Question: {question}
Final Answer:"""

    # Generate all answers in parallel
    tasks = []
    for _ in range(attempts):
        task = llm_service.generate_text(
            prompt=prompt_template,
            max_tokens=100,  # Reduced significantly
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=0.0
        )
        tasks.append(task)
    
    answers = await asyncio.gather(*tasks)
    
    # Score all answers
    reference_text = question + " " + chain_of_thought
    attempts_data = [
        (answer, rogue_service.calculate_advanced_score(answer, reference_text))
        for answer in answers if answer  # Only score non-empty answers
    ]
    
    if not attempts_data:
        return "Unable to generate a consistent answer."
        
    best_answer, best_score = max(attempts_data, key=lambda x: x[1])
    logger.info("Final answer selected")
    return best_answer

async def process_question(question: str) -> str:
    """Main processing function"""
    logger.info(f"Processing question: {question}")
    
    start_time = time.time()
    
    try:
        # Fetch demonstrations
        demo_start = time.time()
        top_demos = await fetch_most_relevant_demonstrations(question, k=3)
        logger.info(f"Fetching demonstrations took: {time.time() - demo_start:.2f}s")
        
        # Generate chain-of-thought
        cot_start = time.time()
        chain_of_thought = await generate_chain_of_thought(question, top_demos)
        logger.info(f"Generating chain-of-thought took: {time.time() - cot_start:.2f}s")
        
        # Generate final answer
        answer_start = time.time()
        final_answer = await self_consistency_final_answer(question, chain_of_thought, attempts=3)
        logger.info(f"Generating final answer took: {time.time() - answer_start:.2f}s")
        
        logger.info(f"Total processing time: {time.time() - start_time:.2f}s")
        return final_answer
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

def main():
    """Entry point function"""
    question_input = """Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?"""

    
    try:
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async function
        answer = loop.run_until_complete(process_question(question_input))
        
        # Close the loop
        loop.close()
        
        print(f"\nQuestion: {question_input}")
        print(f"Answer: {answer}\n")
        
    except Exception as e:
        logger.error(f"Failed to process question: {e}")
        raise
    finally:
        # Ensure we clean up the loop
        try:
            loop.close()
        except:
            pass

if __name__ == "__main__":
    main()
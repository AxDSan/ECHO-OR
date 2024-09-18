import asyncio
import os
import random
from typing import Dict, List

from litellm import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from src.services.embedding_service import EmbeddingService
from src.services.llm_service import LLMService
from src.services.rogue_service import RogueService
from src.demonstration_refiner import DemonstrationRefiner
from src.utils.config import Config
import logging
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Configuration Parameters
API_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_APP_NAME = "ECHO"  # Replace with your actual app name

LLM_MODEL_NAME = "google/gemini-flash-1.5"  # You can change this to other models supported by OpenRouter
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_CLUSTERS = 5
NUM_ITERATIONS = 3
TOP_DEMONSTRATIONS = 5
DIVERSITY_THRESHOLD = 0.7
BATCH_SIZE = 4

INITIAL_PROMPT_TEMPLATE = """You are an advanced and complex AI that allows critical and complex logic thinking\n{question}\nLet's think step by step and find out whats unique in the question or the logic itself to find the answer."""
REFINEMENT_PROMPT_TEMPLATE = """Based on the following Q&A pairs:
{demonstrations}
Refine the rationale for the question below.
Question: {question}
Current Rationale: {current_rationale}
Improved Rationale:"""

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AsyncOpenAI(
    base_url=Config.API_URL,
    api_key=Config.OPENROUTER_API_KEY,
)

embedding_service = EmbeddingService()
llm_service = LLMService()
rogue_service = RogueService()
        
refiner = DemonstrationRefiner(embedding_service, llm_service, rogue_service)

async def send_prompt_async(prompts: List[str], max_length: int, temperature: float, top_p: float, repetition_penalty: float) -> List[str]:
    """
    Asynchronously send prompts to the hosted model and retrieve responses using OpenAI client.
    """
    async def process_prompt(prompt):
        try:
            completion = await client.chat.completions.create(
                extra_headers={
                    "X-Title": YOUR_APP_NAME,
                },
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=repetition_penalty,
                n=1,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating completion: {e}", exc_info=True)
            return ""

    rationales = await asyncio.gather(*[process_prompt(prompt) for prompt in prompts])
    return rationales

async def generate_zero_shot_cot_batch_api(questions_batch: List[str]) -> List[str]:
    prompts = [INITIAL_PROMPT_TEMPLATE.format(question=q) for q in questions_batch]
    rationales = await send_prompt_async(prompts, max_length=4096, temperature=0.0, top_p=0.95, repetition_penalty=1.2)
    return rationales

async def generate_refined_cot_batch_api(refinement_prompts: List[str]) -> List[str]:
    rationales = await send_prompt_async(refinement_prompts, max_length=4096, temperature=0.7, top_p=0.95, repetition_penalty=1.2)
    return rationales

def load_sentence_transformer_model() -> SentenceTransformer:
    """Loads the Sentence-BERT embedding model."""
    try:
        logger.info("Loading Sentence-BERT model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        return embedding_model
    except Exception as e:
        logger.error(f"Error loading Sentence-BERT model: {e}", exc_info=True)
        raise e

def cluster_questions(questions: List[str], embedding_model: SentenceTransformer, num_clusters: int) -> Dict[int, List[int]]:
    """Clusters questions based on their semantic embeddings."""
    if len(questions) < num_clusters:
        logger.warning(f"Number of clusters ({num_clusters}) is greater than the number of questions ({len(questions)}). Adjusting num_clusters to {len(questions)}.")
        num_clusters = len(questions)
    
    logger.info("Clustering questions...")
    question_embeddings = embedding_model.encode(questions, convert_to_tensor=False, show_progress_bar=True)
    clustering = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = clustering.fit_predict(question_embeddings)
    
    clustered_questions = {}
    for idx, cluster_id in enumerate(clusters):
        clustered_questions.setdefault(cluster_id, []).append(idx)
    
    return clustered_questions

def sample_demonstrations(
    dataset: List[Dict[str, str]],
    clustered_questions: Dict[int, List[int]]
) -> List[Dict[str, str]]:
    """Selects representative questions from each cluster."""
    demonstrations = []
    logger.info("Sampling demonstrations...")
    for cluster_id, indices in clustered_questions.items():
        for idx in indices:
            question = dataset[idx]['question']
            answer = dataset[idx]['answer']
            initial_rationale = dataset[idx]['initial_rationale']
            demonstrations.append({
                "question": question,
                "answer": answer,
                "rationale": initial_rationale,
                "rogue_score": 0.0,
            })
            break  # Select the first suitable question in the cluster
    return demonstrations

def select_top_demonstrations(
    demonstrations: List[Dict[str, str]],
    embedding_model: SentenceTransformer,
    top_k: int,
    diversity_threshold: float,
) -> List[Dict[str, str]]:
    logger.info("Selecting top demonstrations based on length, ROGUE score, and diversity...")
    
    # Sort demonstrations by ROGUE score and rationale length in descending order
    sorted_demos = sorted(demonstrations, key=lambda x: (x['rogue_score'], len(x['rationale'])), reverse=True)
    
    selected = []
    embeddings = [embedding_model.encode(demo['question']) for demo in sorted_demos]
    
    for idx, demo in enumerate(sorted_demos):
        if len(selected) >= top_k:
            break
        demo_embedding = embeddings[idx]
        is_diverse = True
        for selected_demo in selected:
            selected_embedding = embedding_model.encode(selected_demo['question'])
            similarity = cosine_similarity([demo_embedding], [selected_embedding])[0][0]
            if similarity > diversity_threshold:
                is_diverse = False
                break
        if is_diverse:
            selected.append(demo)
    
    return selected

async def refine_demonstrations(
    demonstrations: List[Dict[str, str]],
    embedding_model: SentenceTransformer,
    num_iterations: int,
) -> List[Dict[str, str]]:
    """Refines rationales by iterative regeneration using other demonstrations as context."""
    logger.info("Refining demonstrations iteratively...")
    for iteration in range(num_iterations):
        logger.info(f"Refinement Iteration {iteration + 1}/{num_iterations}")
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
        
        refined_rationales = await generate_refined_cot_batch_api(refinement_prompts)
        
        for idx, (demo, new_rationale) in enumerate(zip(demonstrations, refined_rationales)):
            logger.info(f"Processing demonstration {idx + 1}")
            if not new_rationale:
                logger.warning(f"Empty rationale generated for question: {demo['question']}")
                continue
            
            # Calculate rogue score
            rogue_score = rogue_service.calculate_score(new_rationale, demo['rationale'])
            
            # Simple heuristic: prefer longer rationales with higher rogue scores
            current_length = len(demo['rationale'])
            new_length = len(new_rationale)
            
            if 50 <= new_length <= 1024*4 and (new_length > current_length or rogue_score > demo['rogue_score']):
                demo['rationale'] = new_rationale[:1024*4]
                demo['rogue_score'] = rogue_score
                logger.info(f"Updated rationale for demonstration {idx + 1}")
            else:
                logger.info(f"Kept original rationale for demonstration {idx + 1}")
            
            logger.info(f"Updated demonstration: {demo}")
    
    return demonstrations

async def generate_answer_api(new_question: str, selected_demonstrations: List[Dict[str, str]]) -> str:
    """Generates an answer to a new question using the selected demonstrations via API."""
    try:
        prompt = ""
        for demo in selected_demonstrations:
            prompt += f"Q: {demo['question']}\nA: {demo['rationale']}\n\n"
        prompt += f"Q: {new_question}\nA: Let's think step by step."
        
        completion = await client.chat.completions.create(
            extra_headers={
                "X-Title": YOUR_APP_NAME,
            },
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=1.2,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        return "Error: Unable to generate answer."

async def main():
    try:
        embedding_model = load_sentence_transformer_model()
    
        # Replace with actual dataset loading
        dataset = Config.dataset
        
        clustered_questions = cluster_questions(
            questions=[item["question"] for item in dataset],
            embedding_model=embedding_model,
            num_clusters=NUM_CLUSTERS,
        )
        
        demonstrations = sample_demonstrations(
            dataset=dataset,
            clustered_questions=clustered_questions,
        )
        
        # Refine rationales
        refined_demonstrations = await refine_demonstrations(
            demonstrations=demonstrations,
            embedding_model=embedding_model,
            num_iterations=NUM_ITERATIONS,
        )
        
        # Select top demonstrations
        selected_demonstrations = select_top_demonstrations(
            demonstrations=refined_demonstrations,
            embedding_model=embedding_model,
            top_k=TOP_DEMONSTRATIONS,
            diversity_threshold=DIVERSITY_THRESHOLD,
        )
        
        # Inference with new question
        new_question = "A man has 53 socks in his drawer: 21 identical blue, 15 identical black and 17 identical red. The lights are out and he is completely in the dark. How many socks must he take out to make 100 percent certain he has at least one pair of black socks?"
        answer = await generate_answer_api(new_question, selected_demonstrations)
        logger.info(f"Q: {new_question}\nA: {answer}")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
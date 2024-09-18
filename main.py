import os
import asyncio
import logging
import random
from typing import List, Dict
import json
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from tqdm.asyncio import tqdm_asyncio

# Configuration Parameters
API_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_APP_NAME = "ECHO"  # Replace with your actual app name

LLM_MODEL_NAME = "google/gemini-flash-1.5"  # You can change this to other models supported by OpenRouter
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_CLUSTERS = 3
NUM_ITERATIONS = 3
TOP_DEMONSTRATIONS = 5
DIVERSITY_THRESHOLD = 0.7
BATCH_SIZE = 4

INITIAL_PROMPT_TEMPLATE = """You are an advanced and complex AI that allows critical and complex logic thinking\n{question}\nLet's think step by step to find the answer."""
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
    base_url=API_URL,
    api_key=API_KEY,
)

# Initialize Rouge
rouge = Rouge()

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
                "rouge_score": 0.0,
            })
            break  # Select the first suitable question in the cluster
    return demonstrations

def select_top_demonstrations(
    demonstrations: List[Dict[str, str]],
    embedding_model: SentenceTransformer,
    top_k: int,
    diversity_threshold: float,
) -> List[Dict[str, str]]:
    """Selects top demonstrations based on length and diversity."""
    logger.info("Selecting top demonstrations based on length and diversity...")
    
    # Sort demonstrations by rationale length in descending order
    sorted_demos = sorted(demonstrations, key=lambda x: len(x['rationale']), reverse=True)
    
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
            
            # Simple heuristic: prefer longer rationales, but not too long
            current_length = len(demo['rationale'])
            new_length = len(new_rationale)
            
            if 50 <= new_length <= 1024*4 and new_length > current_length:
                demo['rationale'] = new_rationale[:1024*4]
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
        dataset = [
    {
        "question": "There are 15 trees originally. After some more were planted, there are now 21 trees. How many trees were planted?",
        "answer": "6",
        "initial_rationale": "Step 1: Start with the original number of trees (15).\nStep 2: Subtract from the current number of trees (21).\n21 - 15 = 6\nTherefore, 6 trees were planted."
    },
    {
        "question": "Jason had 20 lollipops. He gave some to Denny and now has 12 lollipops left. How many lollipops did Jason give to Denny?",
        "answer": "8",
        "initial_rationale": "Step 1: Start with Jason's original lollipops (20).\nStep 2: Subtract the lollipops he has left (12).\n20 - 12 = 8\nTherefore, Jason gave Denny 8 lollipops."
    },
    {
        "question": "There are 3 cars in the parking lot. 2 more cars arrive. How many cars are in the parking lot now?",
        "answer": "5",
        "initial_rationale": "Step 1: Start with the initial number of cars (3).\nStep 2: Add the number of new cars (2).\n3 + 2 = 5\nTherefore, there are now 5 cars in the parking lot."
    },
    {
        "question": "A librarian displays six book titles on a screen: \"Zen Fox Joy Quip Vow Bard.\" She hands out slips of paper, each containing one letter from one of the titles, to three patrons: Emma, Finn, and Grace. The librarian then asks, \"Emma, can you identify the book?\" Emma promptly answers yes. Next, she inquires, \"Finn, do you know the book?\" After brief consideration, Finn also replies yes. Finally, she poses the same question to Grace, who ponders momentarily before confirming yes. Which book title did the librarian choose?",
        "answer": "Fox",
        "initial_rationale": "Emma responds immediately because she has received one of the unique letters that appear only once across all titles: Z, J, Q, I, V, or D. This rules out \"Bard\" as a possibility. These unique letters are distributed among different titles, except for \"Q\" and \"I\" in \"Quip.\" Finn can deduce the title from the remaining unique letters: Z, J, V, or X. This eliminates \"Zen\" and \"Joy\" from consideration. Grace can then narrow it down further. Since there's only one unique letter left, \"F\", the chosen title must be \"Fox.\""
    },
    {
        "question": "Sophia has twice as many candies as Liam. Together, they have 36 candies. How many candies does Sophia have?",
        "answer": "24",
        "initial_rationale": "Step 1: Let the number of Liam's candies be x.\nStep 2: Then Sophia has 2x candies.\nStep 3: Together, x + 2x = 36.\nStep 4: Combine like terms: 3x = 36.\nStep 5: Divide both sides by 3: x = 12.\nStep 6: Sophia has 2x = 24 candies.\nTherefore, Sophia has 24 candies."
    },
    {
        "question": "A train travels from City A to City B in 3 hours. On the return trip, it travels the same distance in 4 hours. What is the average speed of the train for the entire trip if the distance between City A and City B is 180 miles?",
        "answer": "75 mph",
        "initial_rationale": "Step 1: Calculate the speed from City A to City B: Speed = Distance / Time = 180 miles / 3 hours = 60 mph.\nStep 2: Calculate the speed from City B to City A: Speed = 180 miles / 4 hours = 45 mph.\nStep 3: The average speed for the entire trip is Total Distance / Total Time.\nTotal Distance = 180 + 180 = 360 miles.\nTotal Time = 3 + 4 = 7 hours.\nAverage Speed = 360 miles / 7 hours ≈ 51.43 mph.\nHowever, since average speed is generally calculated using the harmonic mean when dealing with two speeds over the same distance:\nAverage Speed = (2 * 60 * 45) / (60 + 45) = (5400) / 105 ≈ 51.43 mph.\nTherefore, the average speed of the train for the entire trip is approximately 51.43 mph."
    },
    {
        "question": "A rectangular garden has a length that is 3 meters longer than its width. If the perimeter of the garden is 26 meters, what are the dimensions of the garden?",
        "answer": "Length = 8 meters, Width = 5 meters",
        "initial_rationale": "Step 1: Let the width be x meters.\nStep 2: Then the length is x + 3 meters.\nStep 3: Perimeter of a rectangle is 2*(Length + Width) = 26 meters.\nStep 4: Substitute the expressions: 2*(x + (x + 3)) = 26.\nStep 5: Simplify inside the parentheses: 2*(2x + 3) = 26.\nStep 6: Distribute the 2: 4x + 6 = 26.\nStep 7: Subtract 6 from both sides: 4x = 20.\nStep 8: Divide both sides by 4: x = 5.\nStep 9: Length = x + 3 = 8 meters.\nTherefore, the garden is 8 meters long and 5 meters wide."
    },
    {
        "question": "In a class of 30 students, 18 play soccer, 15 play basketball, and 10 play both sports. How many students play neither soccer nor basketball?",
        "answer": "7",
        "initial_rationale": "Step 1: Use the principle of inclusion-exclusion.\nNumber of students playing at least one sport = Number playing soccer + Number playing basketball - Number playing both.\n= 18 + 15 - 10 = 23.\nStep 2: Total students = 30.\nStep 3: Number playing neither sport = Total students - Number playing at least one sport.\n= 30 - 23 = 7.\nTherefore, 7 students play neither soccer nor basketball."
    },
    {
        "question": "Emma is twice as old as Noah was when Emma was as old as Noah is now. If Emma is currently 30 years old and Noah is 20 years old, how old was Noah when Emma was as old as Noah is now?",
        "answer": "10 years old",
        "initial_rationale": "Let's define the variables:\nLet the number of years ago when Emma was as old as Noah is now be y.\nStep 1: Emma's current age = 30.\nStep 2: Noah's current age = 20.\nStep 3: y years ago, Emma was 20 years old (since Emma was as old as Noah is now).\nStep 4: y years ago, Noah was (20 - y) years old.\nStep 5: According to the problem, Emma is now twice as old as Noah was then: 30 = 2*(20 - y).\nStep 6: Solve for y: 30 = 40 - 2y → 2y = 10 → y = 5.\nStep 7: Noah was (20 - 5) = 15 years old when Emma was 20.\nHowever, there's a discrepancy. Re-evaluating:\nIf y is the number of years ago when Emma was as old as Noah is now (20), then y years ago, Noah was (20 - y).\nGiven that Emma is now twice as old as Noah was then:\n30 = 2*(20 - y) → 30 = 40 - 2y → 2y = 10 → y = 5.\nTherefore, Noah was 20 - 5 = 15 years old at that time.\nBut the answer provided earlier was 10. Correct answer is 15."
    }
]
    
        
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
        new_question = "How many r's are in the word strawberry?"
        answer = await generate_answer_api(new_question, selected_demonstrations)
        logger.info(f"Q: {new_question}\nA: {answer}")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
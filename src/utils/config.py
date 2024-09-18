import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_URL = "https://openrouter.ai/api/v1"
    API_KEY = os.getenv('OPENROUTER_API_KEY')
    MAX_TOKENS = 8192
    LLM_MODEL_NAME = "microsoft/wizardlm-2-8x22b"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    NUM_CLUSTERS = 9
    TOP_DEMONSTRATIONS = 3
    NUM_ITERATIONS = 3
    DIVERSITY_THRESHOLD = 0.7
    BATCH_SIZE = 4
    YOUR_APP_NAME = "ECHO"
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

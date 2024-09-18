# ECHO-OR

## Overview

ECHO-OR is an advanced AI-driven system designed to handle complex question-answering tasks. Leveraging natural language processing, semantic embeddings, and clustering algorithms, ECHO-OR refines rationales and generates accurate answers through iterative refinement and selection of top demonstrations.

## Features

- **Configuration Management:** Centralized configurations using environment variables and a `Config` class.
- **Semantic Embedding:** Utilizes Sentence-BERT for generating semantic embeddings of questions.
- **Clustering:** Groups similar questions using K-Means clustering to enhance demonstration selection.
- **Refinement Engine:** Iteratively refines rationales for demonstrations to improve answer quality.
- **Integration with LLMs:** Interfaces with OpenAI's GPT models for generating and refining rationales.
- **Rogue Scoring:** Implements BLEU score-based evaluation to assess the quality of generated rationales.
- **Asynchronous Processing:** Efficiently handles API calls and processing using asynchronous programming.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- `pip` package manager

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/0x90/echo-or.git
   cd echo-or
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Environment Variables**

   Create a `.env` file in the root directory and add the following variables:

   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

2. **Configuration File**

   The `src/utils/config.py` file contains various configuration parameters such as API URLs, model names, and dataset definitions. Adjust these settings as needed.

### Usage

Run the main application using:

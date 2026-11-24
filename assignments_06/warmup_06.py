# --- Environment Setup ---

from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"

if load_dotenv(dotenv_path=env_path, override=True):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing.")

    # --- RAG Concepts ---

# Concepts Question 1

# Scenario A: RAG
# RAG is the best approach because the legal assistant needs to search
# hundreds of internal PDF documents that are updated every quarter.
# The system can retrieve information from the latest documents without
# retraining the language model.

# Scenario B: Fine-tuning
# Fine-tuning is the best approach because the startup has 3,000 examples
# of its preferred brand voice. These examples can be used to train the
# model to consistently produce content in the desired writing style.

# Scenario C: Prompt engineering
# Prompt engineering is the best approach because the analyst only needs
# to work with one short report. She can include the report text directly
# in the prompt along with her question, without building a separate RAG system.


# Concepts Question 2

# A confidently wrong answer is more harmful than an answer that says
# "I am not sure" because people may trust the information and act on it
# without checking whether it is accurate.
#
# For example, if an AI assistant confidently provides an incorrect
# medication dosage, a healthcare worker might follow that information
# and potentially harm a patient.
#
# The tone of the response also affects trust. When a model uses confident
# language and presents information as a fact, people may assume that
# the answer is reliable, even when it is actually a hallucination.
# An answer that clearly expresses uncertainty encourages people to
# verify the information before making important decisions.


# Concepts Question 3

# The correct order of the RAG pipeline:

steps = [
    # 1. Extract text from source documents
    # Extract readable text from PDF files or other source documents.

    # 2. Split text into chunks
    # Divide long documents into smaller pieces of text.

    # 3. Convert text chunks into embeddings
    # Represent each text chunk as a numerical vector.

    # 4. Receive the user's query
    # Get the question submitted by the user.

    # 5. Embed the user's query
    # Convert the user's question into a numerical vector.

    # 6. Retrieve the most relevant chunks
    # Find document chunks that are semantically similar to the query.

    # 7. Inject retrieved chunks into the prompt
    # Add the retrieved information to the prompt as context.

    # 8. Generate a response from the LLM
    # The language model generates an answer using the question and context.

    "Extract text from source documents",
    "Split text into chunks",
    "Convert text chunks into embeddings",
    "Receive the user's query",
    "Embed the user's query",
    "Retrieve the most relevant chunks",
    "Inject retrieved chunks into the prompt",
    "Generate a response from the LLM",
]

print("\n--- RAG Pipeline Steps ---")
for number, step in enumerate(steps, start=1):
    print(f"{number}. {step}")

    # --- Keyword RAG ---

import string


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


    # Keyword Question 1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

# Explanation:
# The query contains the keywords "your" and "weekends".
# The word "weekends" appears in hours.txt, while "your"
# appears in hiring.txt and loyalty.txt.
# All three documents have one keyword match.
# Because scores.sort(reverse=True) sorts tuples in descending
# order, loyalty.txt comes before hiring.txt and hours.txt
# when their scores are tied.


# Keyword Question 2

query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

# Explanation:
# No relevant document is selected because none of the query keywords
# match the words in the documents.
#
# Keyword RAG does not understand that espresso and cold brew are
# coffee drinks and that the question is about caffeine.
# Semantic retrieval using embeddings would work better because
# it can identify similarities in meaning, not just exact word matches.

# Keyword Question 3

# Prediction:
# I predict that the function will return "None found".
# The loyalty.txt document contains information about joining
# a loyalty program, but it does not contain the exact keywords
# "sign up" or "rewards".

query = "How do I sign up for rewards?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

# Was the prediction correct?
# Yes. The function returns "None found" because there are no
# overlapping keywords between the query and the documents.
#
# This result demonstrates a limitation of keyword retrieval:
# it cannot recognize synonyms or understand the meaning of a question.
# Semantic RAG would be able to recognize that "sign up for rewards"
# is related to "join our loyalty program".


# --- Semantic RAG Concepts ---

# Semantic Question 1

# 1. What is a vector embedding?
# A vector embedding is a list of numbers that represents the meaning
# of a piece of text. It allows a computer to compare texts based on
# their meaning rather than just matching individual words.

# 2. Which chunk is more relevant: 0.85 or 0.30?
# The chunk with a cosine similarity score of 0.85 is more relevant.
# A higher score means that the meaning of the chunk is more similar
# to the meaning of the query. A score of 0.30 indicates a weaker
# semantic relationship.

# 3. Why can semantic search find relevant chunks without exact words?
# Semantic search compares vector embeddings, which represent the
# meaning of text. Because of this, it can recognize related concepts
# and synonyms even when the query and document use different words.

# Semantic Question 2: Keyword RAG vs Semantic RAG
#
# Semantic Question 2: Keyword RAG vs Semantic RAG
#
# | What is compared?    | Exact word overlap           | Semantic/meaning similarity via embeddings |
# |----------------------|------------------------------|---------------------------------------------|
# | What's retrieved?    | Entire documents             | Relevant chunks of text                    |
# | Synonym handling     | Does not recognize synonyms  | Recognizes similar meanings                 |
# | Storage format       | Plain text dictionary        | Vector store / vector index                |
# | Relevance score      | Keyword match count          | Cosine similarity                           |

# --- LlamaIndex ---

from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

# Path to the Brightleaf Solar PDF documents
pdf_dir = Path("C:\\Users\\Maksi\\CTD-repos\\python200-homework\\assignments_06\\brightleaf_pdfs")

if not pdf_dir.exists():
    raise FileNotFoundError(f"PDF directory not found: {pdf_dir.resolve()}")

# Load PDF documents
docs = SimpleDirectoryReader(str(pdf_dir)).load_data()

print("\nNumber of documents loaded:", len(docs))

# Set the embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Set the LLM model
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Build an in-memory vector index
index = VectorStoreIndex.from_documents(docs)

print("Vector index created successfully.")
# LlamaIndex Question 1

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

# Create a query engine that retrieves the top 3 chunks
query_engine = index.as_query_engine(similarity_top_k=3)

for question in questions:
    print("\n" + "=" * 60)
    print("Question:", question)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    print("\nRetrieved source nodes:")

    for node_with_score in response.source_nodes:
        print("\nSimilarity score:", node_with_score.score)
        print("Chunk text:")
        print(node_with_score.node.get_content()[:150])

        # Observations for Question 1:
# The retrieved chunks were relevant / not relevant because ...
# The model's response sounded confident / cautious because ...
# Unexpected retrievals: ...

# Observations for Question 2:
# The retrieved chunks were relevant / not relevant because ...
# The model's response sounded confident / cautious because ...
# Unexpected retrievals: ...


# Observations for Question 1:
#
# Query 1: What employee benefits does BrightLeaf offer?
#
# The first retrieved chunk was highly relevant because it contained
# information about BrightLeaf's employee benefits.
# However, the other two chunks about partnerships and the company
# mission were not directly relevant to the question.
#
# The model provided a detailed and specific answer.
# Its tone sounded confident and it did not express uncertainty.
#
# Query 2: What are BrightLeaf's security policies?
#
# The first retrieved chunk was highly relevant because it contained
# information about network and data security.
# However, the other two chunks about employee benefits and the
# company mission were unexpected and not directly relevant.
#
# The model provided a detailed and confident answer about security
# policies without using phrases such as "I'm not sure".
#
# Overall, semantic retrieval found the most relevant information,
# but it also retrieved some unrelated chunks.
# This shows that retrieving more context does not always mean
# retrieving only relevant information.


# LlamaIndex Question 2

q = "What employee benefits does BrightLeaf offer?"

# Run with similarity_top_k=1
query_engine_1 = index.as_query_engine(similarity_top_k=1)
response_1 = query_engine_1.query(q)

print("\n--- Results with similarity_top_k=1 ---")
print("Answer:")
print(response_1)

for node in response_1.source_nodes:
    print("\nSimilarity score:", node.score)
    print("Chunk:", node.node.get_content()[:150])


# Run with similarity_top_k=5
query_engine_5 = index.as_query_engine(similarity_top_k=5)
response_5 = query_engine_5.query(q)

print("\n--- Results with similarity_top_k=5 ---")
print("Answer:")
print(response_5)

for node in response_5.source_nodes:
    print("\nSimilarity score:", node.score)
    print("Chunk:", node.node.get_content()[:150])

    # Observations for Question 2:
#
# With similarity_top_k=1, the model retrieved only one chunk
# with a similarity score of approximately 0.762.
# This chunk contained relevant information about employee benefits,
# and the model generated a detailed and specific answer.
#
# With similarity_top_k=5, the model retrieved five chunks.
# The first chunk was still the most relevant, while the other four
# contained less relevant information about partnerships, the company
# mission, financial performance, and product specifications.
#
# The answer remained detailed and covered the main employee benefits.
# However, retrieving five chunks did not provide an obvious improvement
# compared to retrieving only one chunk.
#
# More retrieved context is not always better because unrelated
# information can distract the model and potentially reduce answer quality.
# The retrieval system should focus on finding the most relevant chunks.

# --- LlamaIndex Question 3 ---

q = "What is the favorite color of BrightLeaf's CEO?"

response = query_engine.query(q)

print("\n--- Difficult Query ---")
print("Question:", q)

print("\nAnswer:")
print(response)

print("\nRetrieved source nodes:")

for node in response.source_nodes:
    print("\nSimilarity score:", node.score)
    print("Chunk text:")
    print(node.node.get_content())

    # Observations for Question 3:
#
# I expected the system to struggle with this question because
# the BrightLeaf documents do not contain information about
# the CEO's favorite color.
#
# The model correctly answered that the favorite color is not
# mentioned in the provided information instead of making up an answer.
#
# The retrieved chunks were about employee benefits, the company
# mission, and partnerships. None of them was directly relevant
# to the question.
#
# The similarity scores were relatively low, ranging from
# approximately 0.48 to 0.52.
#
# To improve the system, I would add a similarity threshold
# to filter out chunks that are not relevant enough.
# I would also improve the prompt to instruct the model to say
# when the answer cannot be found in the provided documents.

# --- LlamaIndex Question 4 ---

# Configure the judge LLM
judge_llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

def evaluate_rag_response(q):
    print("\n" + "=" * 60)
    print("Evaluation question:", q)

    response = query_engine.query(q)

    faithfulness_result = faithfulness_evaluator.evaluate_response(
        query=q,
        response=response,
    )

    relevancy_result = relevancy_evaluator.evaluate_response(
        query=q,
        response=response,
    )

    print("\nAnswer:")
    print(response)

    print("\nFaithfulness score:", faithfulness_result.score)
    print("Faithfulness feedback:", faithfulness_result.feedback)

    print("\nRelevancy score:", relevancy_result.score)
    print("Relevancy feedback:", relevancy_result.feedback)

    # First query: information available in the documents
q1 = "What employee benefits does BrightLeaf offer?"

evaluate_rag_response(q1)


# Second query: information not available in the documents
q2 = "What is the favorite color of BrightLeaf's CEO?"

evaluate_rag_response(q2)

# Evaluation Observations:
#
# 1. A faithfulness score of 1.0 means that the response is
# supported by the retrieved context. A score of 0.0 indicates
# that the response is not considered faithful to the context.
#
# 2. Relevancy measures how well the response addresses the user's
# question. Faithfulness focuses on whether the response is supported
# by the retrieved information.
#
# 3. For the employee benefits question, both scores were 1.0.
# The model provided an accurate and relevant answer based on
# the BrightLeaf documents.
#
# For the CEO's favorite color question, faithfulness was 1.0
# and relevancy was 0.0. The model correctly stated that the
# information was not available, but the relevancy evaluator
# did not consider the response relevant.
#
# 4. LLM-as-a-judge uses another language model to evaluate
# AI-generated responses. It is useful for RAG evaluation because
# answers can be expressed in different ways, making exact-match
# accuracy insufficient. A language model can evaluate meaning,
# relevance, and whether an answer is supported by context.
#
# However, LLM-as-a-judge is not perfect and its evaluations
# can sometimes be subjective or inconsistent.
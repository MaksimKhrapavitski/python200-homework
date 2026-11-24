# --- Groundwork Coffee Co. Q&A Assistant ---

# --- Step 1: Setup ---

from dotenv import load_dotenv
import os
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# Load API key from .env
env_path = Path(__file__).resolve().parent.parent / ".env"

if load_dotenv(dotenv_path=env_path, override=True):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key.")

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is missing."


# Check that the document directory exists
docs_dir = Path("groundwork_docs")

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print("Groundwork documents directory found.")

# --- Step 2: Load the Documents ---

docs = SimpleDirectoryReader(str(docs_dir)).load_data()

print("\nNumber of documents loaded:", len(docs))

print("\nLoaded documents:")

for doc in docs:
    print(doc.metadata["file_name"])


    # --- Step 3: Build the Index and Query Engine ---

# Set the embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Set the LLM model
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Build the vector index
index = VectorStoreIndex.from_documents(docs)

# Create the query engine
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")

# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n" + "=" * 60)
    print("Question:", question)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    print("\nTop retrieved source node:")

    # Get the most relevant retrieved chunk
    top_node = response.source_nodes[0]

    print("Document:", top_node.node.metadata.get("file_name", "Unknown"))
    print("Similarity score:", top_node.score)
    print("Chunk text:")
    print(top_node.node.get_content()[:200])


# Step 4 Reflection:
#
# The assistant provided confident and mostly accurate answers
# to all five questions.
#
# Most answers matched the information in the source documents.
# However, some retrieved source nodes were not the most relevant.
#
# For example, the weekend hours question returned our_story.txt
# instead of faq.txt, even though the answer was correct.
# The dairy-free milk question returned seasonal_specials.txt
# instead of menu.txt.
#
# This shows that a correct answer does not always mean
# that the retrieval step selected the best source document.
#
# I was surprised that the model could provide correct answers
# even when the top retrieved chunk was not the most relevant.

# --- Step 5: Find a Failure ---

failure_question = "What is the exact monthly revenue of Groundwork Coffee Co.?"

print("\n" + "=" * 60)
print("Failure Test Question:", failure_question)

failure_response = query_engine.query(failure_question)

print("\nFull Answer:")
print(failure_response)

print("\nAll Retrieved Source Nodes:")

for node in failure_response.source_nodes:
    print("\nDocument:", node.node.metadata.get("file_name", "Unknown"))
    print("Similarity score:", node.score)
    print("Chunk text:")
    print(node.node.get_content()[:200])
    

# Step 5 Reflection:
#
# I asked: "What is the exact monthly revenue of Groundwork Coffee Co.?"
# I expected this question to be difficult because none of the
# available documents contains financial revenue information.
#
# The system retrieved our_story.txt, menu.txt, and
# wholesale_catering.txt. These documents were related to the
# company but did not contain the requested financial data.
#
# The model correctly stated that the exact monthly revenue
# was not provided in the available information.
# It did not invent a number or make unsupported claims.
#
# The model maintained a confident tone, but its answer
# appropriately acknowledged the missing information.
#
# To improve the system, I would add a similarity threshold
# to reject weak or irrelevant retrieval results.
# I would also improve the prompt to explicitly instruct
# the model not to guess when the answer is missing from the context.

# --- Step 6: Reflection ---

# Question 1:
# LlamaIndex made the implementation much simpler.
# My project required only a few lines of code to load documents,
# create embeddings, build the vector index, and create a query engine.
#
# Without a framework, I would need to write separate code for
# extracting text, splitting documents into chunks, generating
# embeddings, storing vectors, calculating similarity, and retrieving
# relevant information.
#
# The keyword RAG implementation was more complicated because
# I had to write the tokenization, stopword filtering, keyword
# matching, and scoring logic myself.
#
# This shows that frameworks like LlamaIndex save development time,
# reduce complexity, and allow developers to focus on building
# useful applications instead of implementing every component manually.


# Question 2:
# A useful application of RAG would be an internal assistant
# for a pharmaceutical manufacturing company.
#
# The assistant could search through SOPs, quality procedures,
# training documents, and equipment instructions.
# Employees could ask questions and receive answers based
# on the company's approved documentation.
#
# This could save time, make information easier to access,
# and help employees find the correct procedures.


# Question 3:
# One failure mode that RAG cannot fully prevent is hallucination.
# Even when retrieval works correctly, the LLM may misunderstand
# the retrieved context, combine facts incorrectly, or generate
# unsupported details.
#
# RAG reduces the risk of hallucinations by grounding answers
# in external documents, but it does not guarantee that every
# generated answer will be accurate.
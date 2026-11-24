
# --- Imports and Setup ---
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

client = OpenAI()


# --- Completions API ---
# API Q1

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)

# Print the response text
print("Model response:")
print(response.choices[0].message.content)

# Print the model name
print("Model name:")
print(response.model)

# Print total tokens used
print("Total tokens used:")
print(response.usage.total_tokens)


# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."

temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp
    )

    print(f"\nTemperature: {temp}")
    print("Response:")
    print(response.choices[0].message.content)


# Observation:
# Temperature 0 produces more consistent and predictable outputs.
# Higher temperatures produce more varied and creative outputs.
# I would use temperature 0 for consistent, reproducible results.



# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)

# Print all three completions
for i, choice in enumerate(response.choices, start=1):
    print(f"\nCompletion {i}:")
    print(choice.message.content)



# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print("\nAPI Q4 - Response:")
print(response.choices[0].message.content)


# Observation:
# The response is cut short because max_tokens limits the
# maximum number of tokens generated.
# In a real application, max_tokens helps control response
# length, API costs, and resource usage.


# --- System Messages and Personas ---

# System Q1 - First personality

messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q1 - Patient Python Tutor:")
print(response.choices[0].message.content)


# System Q1 - Second personality

messages = [
    {
        "role": "system",
        "content": "You are a strict and concise senior Python developer. Explain technical concepts directly, using precise terminology and no unnecessary encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q1 - Strict Python Developer:")
print(response.choices[0].message.content)


# Observation:
# The first personality gives a patient and encouraging explanation.
# The second personality gives a more direct and technical explanation.
# The system message changes the model's tone and communication style.



# System Q2 - Conversation History

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q2 - Conversation History:")
print(response.choices[0].message.content)


# Explanation:
# The API is stateless, but we manually provide the previous
# conversation messages in the messages list.
# The model knows the name Jordan because the conversation
# history contains the user's earlier message.



# --- Prompt Engineering ---

# Prompt Q1 - Zero-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Reviews:
{reviews}

Return one sentiment for each review, labeled by review number.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q1 - Zero-Shot:")
print(response.choices[0].message.content)



# Prompt Q2 - One-Shot

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify these reviews:

{reviews}

Return the results in this format:
Review 1: sentiment
Review 2: sentiment
Review 3: sentiment
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q2 - One-Shot:")
print(response.choices[0].message.content)


# Observation:
# Adding one example helps the model understand the expected
# output format and can make the results more consistent.



# Prompt Q3 - Few-Shot

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Examples:

Review: "The product is excellent and easy to use."
Sentiment: positive

Review: "The application is slow and full of bugs."
Sentiment: negative

Review: "Great features, but the customer service is terrible."
Sentiment: mixed

Now classify these reviews:

{reviews}

Return the results in this format:
Review 1: sentiment
Review 2: sentiment
Review 3: sentiment
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q3 - Few-Shot:")
print(response.choices[0].message.content)


# Comparison:
# Zero-Shot is useful for simple tasks without examples.
# One-Shot is useful when one example can clarify the task or format.
# Few-Shot is useful for more complex tasks requiring consistent
# classification based on multiple examples.


# Prompt Q4 - Chain of Thought

prompt = """
Solve the following problem.

A data engineer earns $85,000 per year. She gets a 12% raise,
then 6 months later takes a new job that pays $7,500 more per
year than her post-raise salary.

Explain the calculation briefly, step by step, and clearly
label the final answer.

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q4 - Chain of Thought:")
print(response.choices[0].message.content)


# Observation:
# Asking the model to break a problem into steps can help it
# organize calculations and reduce mistakes on multi-step tasks.



# Prompt Q5 - Structured Output
import json

review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

prompt = f"""
Analyze the sentiment of this review:

{review}

Return ONLY valid JSON with these keys:
- sentiment: positive, negative, or mixed
- confidence: a float between 0 and 1
- reason: one sentence explaining the sentiment

Do not include Markdown code fences or any text outside the JSON.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

raw_response = response.choices[0].message.content

print("\nPrompt Q5 - Raw JSON Response:")
print(raw_response)

try:
    data = json.loads(raw_response)

    print("\nSentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])

except json.JSONDecodeError:
    print("\nInvalid JSON. Raw response for debugging:")
    print(raw_response)



# Prompt Q6 - Delimiters

user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q6 - Instructions:")
print(response.choices[0].message.content)



# Prompt Q6 - Regular Prose

user_text = (
    "The Pacific Ocean is the largest ocean on Earth. "
    "It covers more area than all the land on Earth combined."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q6 - Regular Prose:")
print(response.choices[0].message.content)


# Observation:
# Delimiters help separate user-provided text from instructions.
# They reduce confusion and help prevent user text from being
# interpreted as instructions to the model.


# ==========================================
# Local Models with Ollama
# ==========================================

# Ollama Question 1

# Ollama output:
"""
Okay, the user wants me to explain a large language model in two sentences. Let me start by recalling what a large 
language model is. First, they're pretty common, right? So, they use a lot of training data, like books, articles, 
or even web content. Then, they process that data to generate complex text. Wait, but how does that work? Oh right, 
they use a lot of training data, and then they have a lot of parameters. But I need to make sure the two sentences 
are clear and concise.

First sentence: Introduce the model as a system that uses vast data to create complex text. Second sentence: Explain 
how it works by processing that data and generating new text. Let me check if that's two sentences. Yes, each 
sentence is separate. Also, ensure that the terms are correct. Maybe mention that it's trained on a lot of data. 
Alright, that should do it.

A large language model is a complex system that uses vast amounts of training data to process and generate text, 
enabling it to understand and create meaningful content. It operates by analyzing this data, learning patterns, and 
using those patterns to produce new, relevant text.
"""

# Run the same prompt using the OpenAI API
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)

print("\n--- OpenAI API Response ---")
print(response.choices[0].message.content)

# Comparison:
# The OpenAI response was more concise and structured.
# The Ollama response included additional text before the final answer,
# but both models explained the concept of a large language model
# in a similar way.

# Advantage of running a model locally:
# It provides more privacy because prompts and data can stay
# on the local computer without being sent to an external API.

# Disadvantage:
# A small local model may produce less accurate or detailed responses
# and may have limited capabilities compared to larger cloud-based models.
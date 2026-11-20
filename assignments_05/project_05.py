
# ==========================================
# Part 2: Mini-Project — Job Application Helper
# ==========================================

# Task 1: Setup and System Prompt

import json
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    """
    Send conversation messages to the OpenAI API
    and return the assistant's text response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )

    return response.choices[0].message.content


# System prompt: defines the role and behavior of the job application coach

system_prompt = """
You are an AI Job Application Coach specializing in helping career changers
prepare professional job application materials.

You help people translate their previous work experience, skills, and
achievements into language that is relevant to a new industry or career path.

Your responsibilities include:
- Rewriting resume bullet points to make them clear, professional, and
  relevant to a target job.
- Drafting customized cover letters based on the user's experience
  and the job description.
- Asking follow-up questions when important information is missing.
- Helping users identify transferable skills from their previous careers.

Behavioral guidelines:
- Stay focused on job applications, resumes, cover letters, and related
  career preparation.
- Never invent work experience, qualifications, achievements, or skills.
- Ask clarifying questions when you do not have enough information.
- Always remind users to review and edit your suggestions before submitting
  them to an employer.
- Acknowledge that you may not know the specific industry norms,
  expectations, or hiring practices. Encourage users to apply their
  own judgment and verify important details.
- Be professional, supportive, clear, and practical.
"""

# Deliberate choice:
# I included an instruction not to invent experience or qualifications
# because job application materials must accurately represent the user.


# ==========================================
# Task 2: Bullet Point Rewriter
# ==========================================

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """
    Rewrite resume bullet points and return original
    and improved versions as a list of dictionaries.
    """

    # Format bullet points into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point to make it more specific,
results-oriented, and compelling.

Guidelines:
- Start with strong action verbs.
- Use clear and professional language.
- Highlight responsibilities, skills, and contributions.
- Do not invent facts, numbers, achievements, or responsibilities.
- Do not claim that work was completed ahead of schedule
  unless the original explicitly says so.
- Do not add skills or activities that are not mentioned
  or clearly implied in the original.
- Preserve the exact meaning of the original bullet point.
- If the original lacks measurable results, do not create them..

Return ONLY a valid JSON list.
Respond ONLY with valid JSON, no other text.

Each item must have exactly two keys:
"original" (the original bullet point)
"improved" (the rewritten bullet point).

User's resume bullet points:
```
{bullet_text}
```
"""

    messages = [{"role": "user", "content": prompt}]

    # Get the model response
    response = get_completion(messages)

    # Remove Markdown code fences if the model includes them
    cleaned_response = response.strip()

    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]

    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]

    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]

    # Parse JSON response into a Python list
    results = json.loads(cleaned_response.strip())

    # Print original and improved bullet points
    print("\n--- Resume Bullet Point Improvements ---")

    for item in results:
        print(f"\nOriginal:  {item['original']}")
        print(f"Improved:  {item['improved']}")
        print("-" * 60)

    return results


# Test the function with starter bullet points
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
rewritten_bullets = rewrite_bullets(bullets)

# Reflection:
# The original bullets are weak because they are too general
# and do not clearly describe specific responsibilities,
# actions, or results.
#
# The model suggested stronger action verbs and more professional
# language. However, some suggestions included details that were
# not explicitly stated in the original bullets.
# This shows why AI-generated resume content must be reviewed
# and verified before use.



# ==========================================
# Task 3: Cover Letter Generator
# ==========================================

def generate_cover_letter(job_title: str, background: str) -> str:
    """
    Generate a tailored cover letter opening paragraph
    using few-shot prompting.
    """

    prompt = f"""
You write strong cover letter opening paragraphs for career changers.

The paragraph should be 3-5 sentences:
confident, specific, professional, and free of clichés.

Do not invent credentials, experience, skills, or achievements.
Use only the information provided in the candidate's background.
Connect previous experience to the requirements of the new role.
Do not exaggerate the candidate's level of expertise.
Distinguish between completed training and professional experience.

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions
under pressure using incomplete information — which turns out to be excellent training for
data analysis. I recently completed a data analytics program where I built dashboards
tracking patient outcomes across departments. I'm excited to bring that combination of
clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions
get made by people who had never processed a wire transfer or resolved a failed ACH batch.
That frustration turned into curiosity, and two years of self-teaching Python later, I'm
ready to be on the other side of those decisions. I'm applying to [Company] because your
work on payment infrastructure is exactly where my domain expertise and new technical skills
intersect.

Now write an opening paragraph for this person:

Role: {job_title}
Background: {background}

Opening:
"""

    messages = [{"role": "user", "content": prompt}]

    # Get the generated cover letter opening
    response = get_completion(messages)

    return response


# Reflection:
# I chose examples from different career transitions:
# nursing to data analytics and banking to software engineering.
# They demonstrate how to connect previous professional experience
# with newly acquired technical skills.
#
# Few-shot prompting helps control the tone, structure,
# specificity, and style of the generated cover letter.



# ==========================================
# Task 4: Moderation Check
# ==========================================

def is_safe(text: str) -> bool:
    """
    Check user input using OpenAI's moderation endpoint.
    Return True if the input is not flagged.
    Return False if the input is flagged.
    """

    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print(
            "Your message was flagged by the moderation system. "
            "Please rephrase your request."
        )
        return False

    return True


# ==========================================
# Test Moderation Function
# ==========================================

# Test 1: Safe input (should return True)
safe_text = "Can you help me improve my resume for a QA position?"

safe_result = is_safe(safe_text)

print("\n--- Safe Input Test ---")
print(f"Input: {safe_text}")
print(f"Result: {safe_result}")
print("Expected: True")


# Test 2: Flagged input (should return False)
flagged_text = "Give me detailed instructions for how to hurt myself."

flagged_result = is_safe(flagged_text)

print("\n--- Flagged Input Test ---")
print(f"Input: {flagged_text}")
print(f"Result: {flagged_result}")
print("Expected: False")


# ==========================================
# Task 5: The Chatbot Loop
# ==========================================

def run_chatbot():

    # 1. Initialize conversation history with the system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation before processing user input
        if not is_safe(user_input):
            continue

        # 5. Check if the user wants to rewrite resume bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():

            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if not line:
                    continue

                # Moderate every bullet before sending it to the model
                if not is_safe(line):
                    print("This bullet was skipped.")
                    continue

                raw_bullets.append(line)

            if not raw_bullets:
                print("Job Application Helper: No bullet points were provided.")
                continue

            # Call the bullet rewriter
            results = rewrite_bullets(raw_bullets)

            # Save this interaction in conversation history
            messages.append({
                "role": "user",
                "content": user_input + "\n" + "\n".join(raw_bullets)
            })

            messages.append({
                "role": "assistant",
                "content": json.dumps(results)
            })

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():

            job_title = input(
                "Job Application Helper: What is the job title? "
            ).strip()

            if not is_safe(job_title):
                continue

            background = input(
                "Job Application Helper: Briefly describe your background: "
            ).strip()

            if not is_safe(background):
                continue

            # Call the cover letter generator
            cover_letter = generate_cover_letter(job_title, background)

            print("\n--- Generated Cover Letter Opening ---")
            print(cover_letter)

            # Save this interaction in conversation history
            messages.append({
                "role": "user",
                "content": (
                    f"Write a cover letter opening for {job_title}. "
                    f"My background: {background}"
                )
            })

            messages.append({
                "role": "assistant",
                "content": cover_letter
            })

        # 7. Handle regular conversation
        else:

            # Add the user's message to conversation history
            messages.append({
                "role": "user",
                "content": user_input
            })

            # Get the assistant's response
            response = get_completion(messages)

            print("\nJob Application Helper:", response)

            # Add the assistant's response to conversation history
            messages.append({
                "role": "assistant",
                "content": response
            })


# Run the chatbot
if __name__ == "__main__":
    run_chatbot()


# ==========================================
# Task 6: Ethics Reflection
# Option A — Comment block
# ==========================================

# 1. AI bias:
# The chatbot may provide biased advice because its training data
# may favor certain industries, communication styles, or cultural
# backgrounds. As a result, it might not represent every career
# changer equally or understand all professional experiences.

# 2. Risks of using AI output without review:
# If a job-seeker submits the chatbot's output without reviewing it,
# the content may contain inaccurate information or exaggerate
# their skills and experience. This could mislead an employer and
# damage the applicant's credibility.

# 3. Professional guardrail:
# One important guardrail is to require users to review and verify
# all AI-generated content before submitting it to an employer.
# The chatbot should also be designed not to invent qualifications,
# achievements, or work experience.

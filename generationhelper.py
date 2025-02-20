import os
from groq import Groq
import time
import tiktoken

groq_token = os.getenv("GROQ_TOKEN")

groq_client = Groq(
    api_key = groq_token
)

# Initialize token counter and timestamp
tokens_used = 0
start_time = time.time()

def Count_tokens(text: str, model="gpt-3.5-turbo"):
    """Counts tokens in the given text using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def Enforce_token_limit(prompt, max_tokens_per_minute=6000):
    """Ensures that token usage stays within the allowed rate limit."""
    global tokens_used, start_time

    tokens = Count_tokens(prompt)
    elapsed_time = time.time() - start_time

    # If the token limit is exceeded, wait until the reset
    if tokens_used + tokens > max_tokens_per_minute:
        if elapsed_time < 60:
            sleep_time = 60 - elapsed_time
            print(f"Rate limit reached! Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

        # Reset counter after sleeping
        tokens_used = 0
        start_time = time.time()

    # Update token count
    tokens_used += tokens


def GenerateAnswer(query, top_documents, prompt_model, timeout_seconds: int = 30):
    """
    Generates an answer using an AI model based on the top retrieved documents.
    """
    try:
        # Convert each document (if it's a list) into a string before joining
        documents = "\n\n".join([" ".join(doc) if isinstance(doc, list) else str(doc) for doc in top_documents["document"]])

        # Construct the prompt
        prompt = f"""
        You are an AI assistant tasked with answering a question based on the information provided in the given documents. Your response should be accurate, concise, and directly address the question using only the information from the documents. If the documents do not contain sufficient information to answer the question, state that clearly.

        Documents:
        {documents}

        Question: {query}
        Answer:
        """

        Enforce_token_limit(prompt)

        # Call Groq API (Llama 3.3-70B)
        completion = groq_client.chat.completions.create(
            model=prompt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            timeout=timeout_seconds 
        )

        # Extract and print the response
        response_text = completion.choices[0].message.content
        print("\nGenerated Response:\n", response_text)

    except Exception as e:
        print(f"Error generating answer: {e}")
        return None

    return response_text
import pandas as pd
from llama import respond
import os


def evaluate_answer(prompt, response):
    eval_prompt = f"""
                    Below is a Turkish question (query) and a response given to that question.

                    ### Question:
                    {prompt}

                    ### Response:
                    {response}

                    ### Task:
                    Does this response correctly and sufficiently answer the question?  
                    Please return **only one of the following**:

                    1 → If the response is fully relevant to the question  
                    0 → If the response is incorrect or irrelevant

                    Return only 0 or 1.
                    """

    reply = respond(eval_prompt).strip()

    return 1 if reply.startswith("1") else 0


def main():
    input_file = "generated_responses.csv"
    output_file = "auto_eval_results.csv"

    if not os.path.exists(input_file):
        print("❌ Missing generated_responses.csv. Run driver.py first.")
        return

    df = pd.read_csv(input_file)
    results = []

    print(">>> Running automatic evaluation...")

    for i, row in df.iterrows():
        prompt = row["prompt"]
        response = row["response"]
        score = evaluate_answer(prompt, response)
        results.append({
            "prompt": prompt,
            "response": response,
            "score": score
        })
        print(f"[{i+1}/{len(df)}] Score: {score}")

    # Write results
    pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n✅ Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()

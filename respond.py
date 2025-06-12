import pandas as pd
import os
from llama import respond


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    grouped = df.groupby("prompt")
    return grouped


def format_docs(docs):
    return "\n\n".join([
        f"[{i+1}] Title: {doc['title']}\nSummary: {doc['summary']}\nText: {doc['text'][:300]}"
        for i, doc in enumerate(docs)
    ])


def select_relevant_news(prompt, docs):
    context = format_docs(docs)
    selection_prompt = f"""
                        Below is a Turkish news query and 10 different news articles related to it.
                        Select only the 3 articles that are most relevant to the query.

                        ### Query:
                        {prompt}

                        ### News Articles:
                        {context}

                        ### Task:
                        Select the 3 news articles that are most relevant to the query and return only their numbers as a list.  
                        Example output: [1, 4, 7]
                        
                        Response:
                        """

    text = respond(selection_prompt, task='select').strip()
    selected_indices = [int(i) - 1 for i in text.replace("[",
                                                         "").replace("]", "").split(",") if i.strip().isdigit()]
    return [docs[i] for i in selected_indices if 0 <= i < len(docs)]


def generate_response(prompt, top_docs):
    top_docs_text = format_docs(top_docs)
    answer_prompt = f""" Below is a Turkish news query along with the 3 most relevant news articles related to it.

                        ### Query:
                        {prompt}

                        ### Relevant News:
                        {top_docs_text}

                        ### Task:
                        Using the news articles above, provide a comprehensive and accurate answer to the query in Turkish.
                        Your response must be based solely on the information in the news articles.
                        
                        Response:
                     """

    return respond(answer_prompt, task='generate').strip()


def save_response_to_csv(prompt, response, filename="generated_responses.csv"):
    df = pd.DataFrame([{"prompt": prompt, "response": response}])
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode="a", index=False,
              encoding="utf-8", header=not file_exists)


def main():
    results = load_results("top_k_results.csv")

    for prompt, group in results:
        docs = group.to_dict("records")

        print(f"\n\n=== Sorgu: {prompt} ===")

        # Step 1: Select top 3
        top_docs = select_relevant_news(prompt, docs)
        print("Relevant news are extracted.")
        
        # Step 2: Generate response
        response = generate_response(prompt, top_docs)

        print(f"\nYanıt:\n{response}")
        print("=" * 100)

        # Step 3: Save response to CSV
        save_response_to_csv(prompt, response)


if __name__ == "__main__":
    main()

import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

CHROMA_PATH = "chroma"


PROMPT_TEMPLATE = """
You are a computer science developer well adept in programming with languages like java, python and are very adept at 
making detailed documentation on your work. Based on the context given below answer the presented question with a detailed 
response on the features functionalities and any specific requests related to the question.
Context:
{context}

---

Answer the question based on the above context: {question}
"""

inference_api_key="YOUR API KEY HERE"


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # if len(results) == 0 or results[0][1] < 0.5:
    if len(results) == 0:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    #model = ChatOpenAI()
    #response_text = model.predict(prompt)

    #sources = [doc.metadata.get("source", None) for doc, _score in results]
    #formatted_response = f"Response: {response_text}\nSources: {sources}"
    #print(formatted_response)

    llm = Ollama(model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    llm.invoke(prompt)


if __name__ == "__main__":
    main()

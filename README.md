# KT-RAG: Langchain RAG + LLM for knowledge transfer            

This working repo contains the code files for the KT-RAG: Interactive RAG + LLM for Github repo code and documentation knowledge transfer project.

The proposed KT-RAG essentially combines the ability of RAG to learn your specific domain's features.
In specific, this RAG can learn knowledge from your project's Github repo, and is designed to read both code and documentation.

! In this version of the project, the RAG reads files of type .java, 

## Planned updates:
- Accept mutliple code file types for RAG understanding
- Include functionality to provide just github repo as a web link / handle web pages as inputs in general
- Include follow up interaction functionality

## Requirements

You will need to install the Ollama app to use the llama LLM. Its open source and can be found [here](https://ollama.com/)


Install requirements.

```python
pip install -r requirements.txt
```

## How to use

### Setup

Download the github repo and place into the 'data/' folder 

(Note
 - during execution a folder called "MDFiles" will be created during code execution
 - for the sake of an example I'm using a github repo of my own project - found (here)[https://github.com/Gugan0905/Image-Manipulation-and-Enhancement-Java-UI-UX])


Paste your hugging face API key in the "query_data.py" file for this code to work.

### Running the code

Create the Chroma DB to store the chunks.

```python
python create_database.py
```

Query the Chroma DB with a question related to the project.

```python
python query_data.py "In which file do i find the code to read input images uploaded by the user?"
```

You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work.

## References

This project was inspired from this [repo](https://github.com/pixegami/langchain-rag-tutorial) by pixegami.
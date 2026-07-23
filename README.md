# RAG with the Training Wheels Off

Native retrieval-augmented generation demo built around a single PDF, OpenAI embeddings, NumPy similarity search, and a chat completion step for grounded answers.

## What It Does

- Loads the PDF in [`input_data/`](input_data/).
- Splits the document into overlapping chunks with LangChain text splitters.
- Generates embeddings with OpenAI.
- Retrieves the most similar chunks for a question using cosine similarity in NumPy.
- Sends the retrieved context to a chat model for a final answer.

The current demo document is the EU AI High-Level Expert Group ethics guidelines PDF included in the repo.

## Project Files

- [`rag_openai_native.py`](rag_openai_native.py): end-to-end RAG pipeline and demo queries.
- [`chunking_data.py`](chunking_data.py): PDF loading and chunk creation.
- [`config.py`](config.py): environment loading and PDF path configuration.
- [`input_data/`](input_data/): source PDF used by the pipeline.
- [`lab_proof.md`](lab_proof.md): generated traceability and failure-case notes.
- [`screenshots/`](screenshots/): example output images.

## Requirements

The code imports:

- `openai`
- `numpy`
- `python-dotenv`
- `pypdf`
- `langchain-core`
- `langchain-text-splitters`

The repo does not currently include a pinned requirements file, so install the packages directly in your environment.

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install openai numpy python-dotenv pypdf langchain-core langchain-text-splitters
```

3. Create a `.env` file in the repository root with your OpenAI key:

```env
OPENAI_API_KEY=your_key_here
```

`config.py` loads this value automatically and also points to the bundled PDF.

## Run

Run the main script:

```bash
python rag_openai_native.py
```

The script will:

1. Load and chunk the PDF.
2. Create embeddings for every chunk.
3. Retrieve relevant chunks for a test query.
4. Print the answer.

## Notes

- Chunking is currently configured with overlapping windows to reduce boundary loss between adjacent sections.
- Retrieval is purely semantic similarity over chunk embeddings; there is no vector database.
- If you change the PDF path, update `PDF_PATH` in [`config.py`](config.py).

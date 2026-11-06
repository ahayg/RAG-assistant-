# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines a retriever with a generative language model. Instead of relying only on knowledge stored in model weights, the system fetches relevant passages from an external corpus at query time and places them in the prompt. This reduces hallucination, lets the system answer questions about private or recent data, and makes answers traceable through citations.

## Pipeline stages

A typical RAG pipeline has two phases. In the indexing phase, documents are loaded, split into chunks, converted to embeddings, and stored in a vector index. In the query phase, the user question is embedded, the nearest chunks are retrieved, optionally reranked, and then passed to the language model together with instructions to answer only from the supplied context.

## Chunking

Chunk size is a key trade-off. Small chunks of roughly 200 to 500 tokens give precise retrieval but may lose surrounding context. Large chunks keep context but dilute the embedding and waste prompt budget. Overlap of 10 to 20 percent between neighbouring chunks prevents answers from being cut at a boundary. Splitting on paragraph and sentence boundaries usually beats fixed-width splitting.

## Hybrid retrieval

Dense embeddings capture meaning but can miss exact identifiers, error codes, and rare names. BM25 keyword search handles those well. Hybrid retrieval runs both and merges the rankings with Reciprocal Rank Fusion, which scores each document as the sum of 1 divided by (k + rank) across rankers, with k commonly set to 60. Hybrid search is a strong default because it is robust to both paraphrased and keyword-heavy queries.

## Failure modes

Common failures include retrieving irrelevant chunks, retrieving the right chunk but ranking it too low, prompt injection hidden in documents, and the model ignoring the context. Mitigations include hybrid search, reranking with a cross-encoder, instructing the model to treat context as data, and requiring citations.

# Vector Databases and Similarity Search

A vector database stores high-dimensional embeddings and answers nearest-neighbour queries. The query vector is compared against stored vectors using a similarity measure, most often cosine similarity, dot product, or Euclidean distance. When vectors are L2-normalised, cosine similarity and dot product give identical rankings, so a single matrix multiplication scores every stored vector.

## Exact versus approximate search

Brute-force exact search is perfectly accurate and is fine up to roughly a few hundred thousand vectors. Beyond that, approximate nearest neighbour (ANN) indexes trade a little recall for large speedups.

HNSW builds a layered proximity graph and searches it greedily from the top layer down. It offers high recall and low latency but uses substantial memory. IVF partitions vectors into clusters with k-means and searches only the closest few clusters, controlled by the nprobe parameter. Product quantisation compresses vectors into short codes to cut memory by an order of magnitude at some accuracy cost.

## Popular systems

FAISS is a library for efficient similarity search from Meta. Chroma, Qdrant, Weaviate, Milvus, and pgvector are databases that add persistence, metadata filtering, and horizontal scaling. pgvector is attractive when the application already uses PostgreSQL.

## Metadata filtering

Production systems usually combine vector search with filters such as tenant, document type, or date. Pre-filtering narrows candidates before search; post-filtering removes results afterwards but may return too few. Good indexes support filtered search natively.

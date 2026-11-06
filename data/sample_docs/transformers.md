# Transformers and Embedding Models

The Transformer architecture, introduced in the paper "Attention Is All You Need" in 2017, replaced recurrence with self-attention. Each token produces a query, key, and value vector. Attention weights are computed as softmax(QK^T / sqrt(d_k)) and applied to the values, letting every token attend directly to every other token in a single layer. Multi-head attention runs several attention functions in parallel so the model can capture different relationships.

## Positional information

Because self-attention is permutation-invariant, models add positional information. The original paper used sinusoidal encodings; many modern models use rotary position embeddings (RoPE), which rotate query and key vectors by position-dependent angles and extrapolate better to long contexts.

## Encoder and decoder models

Encoder-only models such as BERT read text bidirectionally and are used for classification and embeddings. Decoder-only models such as GPT generate text autoregressively using a causal mask. Encoder-decoder models like T5 map an input sequence to an output sequence.

## Sentence embeddings

Embedding models are encoders fine-tuned with contrastive learning so that semantically similar sentences land close together. Training uses pairs of related texts, with other examples in the batch serving as negatives. BGE and E5 are widely used open embedding families. Many retrieval models expect a special instruction prefix on the query side, such as "Represent this sentence for searching relevant passages", but not on documents.

## Context windows

The context window limits how many tokens a model can read at once. Attention cost grows quadratically with sequence length, which motivates techniques such as sliding-window attention and FlashAttention.

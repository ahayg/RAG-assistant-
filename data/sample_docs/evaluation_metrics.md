# Evaluating Retrieval Systems

Retrieval quality should be measured separately from answer quality. If the right passage is never retrieved, no language model can answer correctly.

## Retrieval metrics

Hit rate at k (also called recall at k for single-answer questions) is the fraction of queries where at least one relevant document appears in the top k results. Mean Reciprocal Rank (MRR) averages 1 divided by the rank of the first relevant result, so it rewards putting the right passage first. Normalised Discounted Cumulative Gain (nDCG) handles graded relevance by discounting gains logarithmically with rank. Precision at k measures the share of the top k results that are relevant.

## Answer metrics

Faithfulness checks whether every claim in the answer is supported by the retrieved context. Answer relevance checks whether the answer addresses the question. Context precision and context recall measure how much of the retrieved context was useful and how much of the needed information was retrieved. These are commonly scored with an LLM acting as a judge, which should be spot-checked against human labels.

## Building an evaluation set

Start with 30 to 100 real questions, each labelled with the source document or passage that answers it. Synthetic questions generated from chunks are a cheap supplement but tend to be easier than real traffic. Track metrics over time and run the evaluation in continuous integration so regressions in chunking or retrieval are caught before release.

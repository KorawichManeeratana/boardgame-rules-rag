# Retrieval results (k=4)

Index: 74 chunks, strategy=paragraph, size=450, overlap=90

| system | recall@4 | precision@4 | MRR |
|---|---|---|---|
| random (floor) | 0.050 | 0.025 | 0.067 |
| BM25 (keyword) | 0.400 | 0.212 | 0.254 |
| bge-m3 (vector) | 0.683 | 0.350 | 0.642 |

Abstention at min_score=0.45: 1/4 unanswerable questions correctly returned nothing.
Highest top-1 score among them: 0.578 -- a floor below this would start hallucinating.

# Retrieval results (k=4)

Index: 111 chunks, strategy=paragraph, size=300, overlap=45

| system | recall@4 | precision@4 | MRR | nDCG@4 |
|---|---|---|---|---|
| random (floor) | 0.025 | 0.013 | 0.050 | 0.031 |
| BM25 (keyword) | 0.267 | 0.125 | 0.212 | 0.189 |
| bge-m3 (vector) | 0.533 | 0.225 | 0.463 | 0.453 |

Abstention at min_score=0.45: 1/4 unanswerable questions correctly returned nothing.
Highest top-1 score among them: 0.583 -- a floor below this would start hallucinating.

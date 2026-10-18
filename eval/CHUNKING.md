# Chunking sweep (k=4)

20 answerable questions, 4 unanswerable. Gold answers matched by text containment, not chunk id (see `eval/align.py`), because ids are not comparable across chunkings.

| config | chunks | recall@4 | precision@4 | MRR | min answerable top-1 | max unanswerable top-1 |
|---|---|---|---|---|---|---|
| paragraph size=300 overlap=45 | 111 | 0.533 | 0.275 | 0.512 | 0.558 | 0.583 |
| fixed size=300 overlap=0 | 99 | 0.583 | 0.275 | 0.508 | 0.558 | 0.583 |
| paragraph size=300 overlap=0 | 108 | 0.558 | 0.250 | 0.492 | 0.558 | 0.583 |
| paragraph size=450 overlap=90 | 74 | 0.683 | 0.350 | 0.642 | 0.540 | 0.578 |
| paragraph size=600 overlap=120 | 59 | 0.683 | 0.312 | 0.642 | 0.545 | 0.578 |
| paragraph size=200 overlap=40 | 160 | 0.558 | 0.312 | 0.533 | 0.568 | 0.558 |

Baseline (paragraph size=300 overlap=45): recall@4 = 0.533
Best     (paragraph size=450 overlap=90): recall@4 = 0.683  (+0.150)

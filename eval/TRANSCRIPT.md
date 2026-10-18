# Demo transcript

One run against the index of 74 chunks (paragraph, size=450, overlap=90), top_k=4, min_score=0.45, temperature=0.0.

Reproduce any line below with the command shown above it.

## What is the egg limit on a bird?

The ordinary case: search once, answer, cite the passages used.

**agent**

```
$ python -m rag.cli --agent --show-chunks "What is the egg limit on a bird?"
step 1: search_docs({"query": "egg limit on a bird"})
          -> c0009, c0068, c0067, c0032
  stopped: model answered

The rulebooks state that each bird has an egg limit, and no power allows you to lay more eggs on a bird than its limit allows [c0009]. When you choose to lay eggs, you place an egg token on a bird card according to its egg limit [c0067, c0032].

Sources:
[c0009] wingspan-appendix.pdf p.8  (score 0.659)
[c0068] wingspan-rulebook.pdf p.8  (score 0.658)
[c0067] wingspan-rulebook.pdf p.8  (score 0.655)
[c0032] wingspan-asia.pdf p.8  (score 0.648)
```

## thanks, that helped!

Side by side. The RAG path retrieves for a social message because it always retrieves; the agent skips the tool entirely. This is the difference that justifies the agent loop.

**retrieve-then-read**

```
$ python -m rag.cli --show-chunks "thanks, that helped!"
[1] 0.458  wingspan-rulebook.pdf p.1
      from the supply on this bird. 85cm WS_EE_BirdSwiftStart_r4.indd 1WS_EE_BirdSwiftStart_r4.indd 1 6/9/21 4:19 PM6/9/21 4:19 PM WS_SwiftStartPlayerGuides_r9.indd 1...

I couldn't find that in the rulebooks.
```

**agent**

```
$ python -m rag.cli --agent --show-chunks "thanks, that helped!"
(the agent answered without searching)
  stopped: model answered

You're welcome! Let me know if you have any other questions while you're playing.
```

## What is the capital of France?

Out of scope. Retrieval returns nothing above the floor, so no model call is made at all and the answer is the refusal line.

**retrieve-then-read**

```
$ python -m rag.cli --show-chunks "What is the capital of France?"
(retrieval returned nothing above the min_score floor)

I couldn't find that in the rulebooks.
```

## Can a bird power let me go over my egg limit?

THE FAILURE CASE (FC-5). The answer is 'no power allows it', cited correctly -- and wrong, because the Asia expansion card holding the exception never reached the top-4. Nothing in the output is false and a player following it would still play the rule wrong.

**agent**

```
$ python -m rag.cli --agent --show-chunks "Can a bird power let me go over my egg limit?"
step 1: search_docs({"query": "\"bird power egg limit\""})
          -> c0009, c0068, c0067, c0032
  stopped: model answered

No, no power allows you to lay more eggs on a bird than its limit allows [c0009].

Sources:
[c0009] wingspan-appendix.pdf p.8  (score 0.611)
[c0068] wingspan-rulebook.pdf p.8  (score 0.607)
[c0067] wingspan-rulebook.pdf p.8  (score 0.605)
[c0032] wingspan-asia.pdf p.8  (score 0.600)
```

## go 2 over its egg limit once between turns

The exception IS in the corpus. FC-5 is retrieval missing it, not the rulebook lacking it.

**retrieve-then-read**

```
$ python -m rag.cli --show-chunks "go 2 over its egg limit once between turns"
[1] 0.608  wingspan-rulebook.pdf p.8
      It is possible that you will have more capacity to lay eggs (based on your player mat) than you have spaces on your birds. Any excess beyond your egg limit is l...
  [2] 0.597  wingspan-asia.pdf p.8
      8 OPTION 3. Lay Eggs and Activate Grassland Bird Powers Eggs are part of the cost of playing bird cards in columns 2–5. Also, each egg on the bird cards on your...
  [3] 0.593  wingspan-asia.pdf p.1
      from the birdfeeder on this bird, if there is one. / / WS_Asia_BirdCards_r7.indd 88WS_Asia_BirdCards_r7.indd 88 3/27/22 9:08 AM3/27/22 9:08 AM A Willow Tit Poec...
  [4] 0.593  wingspan-appendix.pdf p.8
      8 Egg-Laying Powers IN GENERAL: Remember that each bird has an egg limit (see Action Option 3 in the rulebook). No power allows you to lay more eggs on a bird t...

I couldn't find that in the rulebooks.
```

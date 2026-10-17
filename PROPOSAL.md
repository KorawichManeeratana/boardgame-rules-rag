# Proposal — Board Game Rules Assistant

**ส่งอาจารย์เพื่อเช็ค scope ก่อนเริ่ม build**

A rules dispute mid-game is answered by one paragraph on one page of a rulebook
nobody wants to flip through. **Board Game Rules Assistant** is a grounded Q&A
app over the rulebook PDFs of three board games: a player asks "can I build a
road through another player's settlement?" in plain language, and the app
retrieves the relevant passages, answers only from them, and cites the file and
page so the table can check it. It is for players mid-game and for anyone
teaching a game they half-remember. I will use three capability families from
the course: **retrieval/RAG** (bge-m3 embeddings, a chunked index carrying
source+page metadata, a min_score floor and a prompt that is allowed to say "I
don't know"), **retrieval evaluation** (a hand-labelled set of ~30 questions
mapped to the chunk ids that genuinely answer them, scored with recall@k and
MRR against a no-retrieval and a keyword-overlap baseline), and an **agent loop
with tools** (retrieval exposed as a `search_docs` tool so the model decides
whether to retrieve, with a max-steps cap and repeated-query detection as the
stop conditions). The measured improvement I will report is the chunking
strategy: a naive fixed 300-word/no-overlap split versus paragraph-boundary
chunking with ~15% overlap, with before/after recall@4 and MRR from the same
eval set. If time remains I will add a FastAPI endpoint with SSE streaming.

---

### คำถามที่ควรถามอาจารย์พร้อมส่ง proposal

1. ใช้ rulebook PDF ที่ publisher แจกฟรี แล้ว commit ลง repo สาธารณะได้ไหม
   หรือควรใช้ fetch script ดึงตอน setup แทน (spec อนุญาตทั้งสองแบบ)
2. 3 family (RAG + eval + agent) พอไหม หรือควรดัน FastAPI+SSE ให้เป็น family ที่ 4
3. eval set 30 ข้อแบบ hand-label ทั้งหมด เทียบกับ bootstrap ด้วย LLM แล้ว audit
   10% — อาจารย์อยากเห็นแบบไหนมากกว่า

# Corpus — where each rulebook came from

Four Wingspan rulebooks, 44 pages in total, committed to this repo as the input
corpus for the evaluation. All four are published free of charge by the game's
publisher on their own site.

| File | Game | Publisher | Downloaded from | Date |
|---|---|---|---|---|
| `wingspan-rulebook.pdf` | Wingspan (base game) | Stonemaier Games | [Wingspan rules page](https://stonemaiergames.com/games/wingspan/rules/) → rulebook Dropbox folder | 2026-10-17 |
| `wingspan-asia.pdf` | Wingspan Asia (expansion) | Stonemaier Games | [Wingspan rules page](https://stonemaiergames.com/games/wingspan/rules/) → expansion Dropbox folder | 2026-10-17 |
| `wingspan-appendix.pdf` | Wingspan — bird power appendix | Stonemaier Games | [Wingspan rules page](https://stonemaiergames.com/games/wingspan/rules/) → appendix | 2026-10-17 |
| `wingspan-automa.pdf` | Wingspan — Automa (solo) rules | Stonemaier Games | [Wingspan rules page](https://stonemaiergames.com/games/wingspan/rules/) → Automa rulebook | 2026-10-17 |

The Asia expansion was chosen over the other regional expansions because it is
the one the intended players own. It also makes the corpus harder in a useful
way: it restates much of the base game in near-identical wording, which is the
cause of the retrieval failures recorded in `eval/FAILURE_CASES.md` (FC-2, FC-5).

## Why these rulebooks

They fit what this project needs:

- **text-heavy, not diagram-heavy** — `pypdf` extracts text, not pictures. The
  limits of that choice are measured rather than hidden: questions whose answer
  is only in a diagram were dropped during labelling and are listed in
  `eval/skipped.txt` (see §5.2 of the report).
- **English** — `bge-m3` is multilingual, so Thai would work too, but mixing
  languages adds a variable the evaluation would have to account for.
- **free and official** — published by Stonemaier Games on their own site.
- **answers have a page number** — every rules question has a right answer
  printed on a specific page, so a wrong citation is visibly wrong and a
  hand-labelled evaluation set is cheap to build and honest to report.

## Licence note

These rulebooks are Stonemaier Games' copyright and are included here only as
the input corpus for a university coursework evaluation, not redistributed as a
product. The table above records where each one came from.
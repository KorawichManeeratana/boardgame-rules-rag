# Corpus — where each rulebook came from

Put the PDFs in this folder, named `<game-name>.pdf` (the filename becomes the
`game` metadata field, so use lowercase and hyphens: `catan.pdf`,
`ticket-to-ride.pdf`).

| File | Game | Publisher | Downloaded from | Date |
|---|---|---|---|---|
| `____.pdf` | | | | |
| `____.pdf` | | | | |
| `____.pdf` | | | | |

## Choosing rulebooks

Good picks for this project are games whose rulebook is:

- **text-heavy, not diagram-heavy** — `pypdf` extracts text, not pictures, so a
  rulebook that explains in prose indexes far better than one that explains in
  arrows and icons;
- **English** — bge-m3 is multilingual, so Thai works too, but mixing languages
  adds a variable the evaluation would have to account for;
- **10–30 pages each**, 2–3 games total — enough chunks that retrieval is a
  real problem, few enough to hand-label 30 questions in an evening;
- **free and official** — most publishers host rulebook PDFs on their own site.

## Licence note

These rulebooks are the publishers' copyright and are included here only as the
input corpus for a university coursework evaluation. The table above records
where each one came from.

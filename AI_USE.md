# AI Use Disclosure

Per the DSA 8401 course policy, this file discloses AI assistance used in this project.

**Tool:** Claude Code (Anthropic), used inside VS Code.

**How it was used:** as a step-by-step guide and pair-programmer. It explained
design options, scaffolded code, and helped debug. I reviewed, ran, and am able
to explain every part of the submitted code, and all modelling decisions,
interpretations of results, and the final report are my own.

## Log

| Date | Step | What the AI helped with | What I did / decided |
|------|------|-------------------------|----------------------|
| 2026-10-28 | 1. Project skeleton | Proposed folder layout, created config, README and .gitignore stubs | Chose project location; approved structure |
| 2026-11-16 | Planning | Summarised the course labs so methods stay within the syllabus; drafted a work plan | Decided on Keras, Optuna, a report and a slide deck, 10–12 species, Season 1 |
| 2026-11-16 | 2. Labels download | Wrote `src/camtrap/data/download.py` (download with retries, unzip) | Reviewed and ran it; chose Season 1 in the config |
| 2026-11-17 | 3. Data audit | Drafted and tested the cells of `notebooks/01_data_audit.ipynb`; proposed cleaning rules from the results | Chose the audit structure, ran the notebook, reviewed and edited the findings, agreed the class rule |

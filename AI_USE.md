# AI Use Disclosure

This file discloses the use of AI tools in this project, as required by the DSA 8401 course policy.

**Tool:** Claude (Anthropic), used throughout the capstone project (28 October and 16–20 November 2026) to think through and write the code, and to organise the repository and its documentation so that every part fits together and works.

## Statements

1. I chose the problem, the dataset and the research question: whether a species classifier still works on camera sites it has never seen.
2. I made the project decisions: Snapshot Serengeti Season 1, twelve species plus "other" and "empty", splitting the data by camera, Keras and Optuna, comparing transfer learning with a CNN from scratch, Grad-CAM with a check of the date strip, macro-F1 as the main metric, and a final report and slide deck.
3. Claude was used to write, edit and iterate on the project code (data pipeline, models, evaluation, API, Docker and GitHub Actions), and to debug and fix errors. It also ran the scripts that do not need a GPU.
4. I ran the photo download, feature extraction, CNN training and Grad-CAM on Google Colab, and checked their outputs.
5. Claude was used to draft and properly word the README, the data audit notebook, the slide deck and the final report, keeping every statement genuine and true to the project's results. I reviewed and edited these drafts.
6. Every number in the report and the slides comes from the output files of the scripts in `reports/results/`.
7. I reviewed the code, the results and all documents. I can explain the methods, the decisions and the results, and I take responsibility for the submitted work.

## Timeline

- **28 October 2026:** project skeleton (folders, configuration, README stub).
- **16 November 2026:** planning within the course syllabus; label download script.
- **17 November 2026:** data audit notebook; cleaning, camera split and photo sample; Colab runner; features and baselines; CNN training script.
- **18 November 2026:** test evaluation; API, Docker and GitHub Actions; drift report and Grad-CAM; one-command pipeline; README; slide deck; final report draft.
- **19–20 November 2026:** checks from a fresh copy of the repository; this disclosure; final edits.

# WETBench: A Benchmark for Detecting Task-Specific Machine-Generated Text on Wikipedia

This repository provides the code and dataset links for the paper  **_WETBench: A Benchmark for Detecting Task-Specific Machine-Generated Text on Wikipedia_**.

**Note:** This repository is still under active development. 🚧🙂  

---

## Abstract

Given Wikipedia's role as a trusted source of high-quality, reliable content, growing concerns have emerged about the proliferation of low-quality machine-generated text (MGT) on its platform, undermining its knowledge integrity.  
Reliable detection of MGT is therefore essential, yet existing work primarily evaluates MGT detectors on generic generation tasks, neglecting the various forms in which MGT arises from editorial workflows.  
This misalignment can lead to poor generalizability when applied to real-world Wikipedia contexts.

We introduce **WETBench**, a multilingual, multi-generator, and _task-specific_ benchmark for MGT text detection, grounded in Wikipedia editors’ perceived use cases for LLM-assisted editing.  
We define three editing tasks—**Paragraph Writing**, **Summarization**, and **Text Style Transfer**—and implement them with two new datasets across three languages.  
For each task, we test three prompting strategies and evaluate detectors from diverse families.

We find that, across settings, training-based detectors achieve an average accuracy of 78%, while zero-shot detectors average 58%, with considerable variation across tasks, generators, and languages.  
These results suggest that detectors struggle to generalize to diverse text generation scenarios, and that reliable detection may not easily scale to editor-driven platforms.

---

## Code

The `code/` directory is organized into the following subdirectories:

- `/collection/` — for collecting article-level samples for **WikiPS**
- `/detectors/` — for all detectors used for benchmarking
- `/mgt/` — for producing MGT data
- `/paragraphs/` — for extracting paragraphs from the article-level samples
- `/scorers/` — for automatic evaluation metrics used to assess prompting strategies for our tasks
- `/summaries/` — for generating summaries from the article-level samples
- `/tst/` — for collecting and extending the WNC corpus ([Pryzant et al., 2020](https://arxiv.org/pdf/1911.09709)) to the **mWNC** and running style classifiers

---

## Data

We provide all data (WikiPS, mWNC, and the MGTs) via Hugging Face:

👉 [**WETBench Dataset (Anonymised)**](https://huggingface.co/datasets/cs928346/WETBench/blob/main/README.md)

---


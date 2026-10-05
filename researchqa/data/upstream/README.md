# Upstream ResearchQA data

We do not redistribute ResearchQA. To rebuild the ResearchQA inputs from scratch, download the
human preference annotations released by the ResearchQA authors
([Yifei et al., 2025](https://arxiv.org/abs/2509.00496)) from their
[pairwise human evaluation folder](https://drive.google.com/drive/folders/1E55AP9xnpjsnk_BTCMceg41ndPFLSdSn)
(linked from the [ResearchQA repository](https://github.com/realliyifei/ResearchQA)) and place
these three files here:

```
researchqa/data/upstream/
├── nlp_plurality_preferences.json
├── cv_plurality_preferences.json
└── multi_domain_plurality_preferences.json
```

`python -m researchqa.features` then combines them with the question metadata from the
[`realliyifei/ResearchQA`](https://huggingface.co/datasets/realliyifei/ResearchQA) Hugging Face
dataset, which it downloads automatically, and computes the predictors.

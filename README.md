# Chain-of-Thought to Explanation-Label pairs (CoT2EL)
This repository contains the implementation for the paper "Threading the Needle: Reweaving Chain-of-Thought Reasoning to Explain Human Label Variation", accepted to the [EMNLP 2025](https://2025.emnlp.org/) Main Conference as ***Oral*** Presentation. ([paper](https://arxiv.org/abs/2505.23368))

![Image text](https://github.com/mainlp/CoT2EL/blob/main/main_structure.png)

The repository is organized into two main projects, each contained within its own directory:

`/Pipeline`: This project implements the complete 5-stage pipeline for generating, extracting, structuring, normalizing, and filtering high-quality explanations from the Chain-of-Thought reasoning of Large Language Models. As a working example, this implementation uses the DeepSeek model family for CoT and explanation generation.

![Image text](https://github.com/mainlp/CoT2EL/blob/main/pipeline.png)

`/Evaluation`: This project provides a comprehensive framework for conducting rank-based human label variation (HLV) evaluations on the explanations produced by the `/Pipeline`. It assesses the impact of these explanations on a model's performance in MCQA tasks across various metrics, including distribution, score, and ranking. As a working example, this implementation uses the Qwen model as the judge.

![Image text](https://github.com/mainlp/CoT2EL/blob/main/evaluation.png)

Both projects are designed to be modular and extensible. Researchers and developers are encouraged to adapt the code to experiment with other language models for both explanation generation and evaluation.

## Getting Started
For detailed instructions on setup, usage, and configuration for each project, please refer to the specific README files located within their respective directories:

For the explanation generation pipeline: `/Pipeline/README.md`

For the evaluation framework: `/Evaluation/README.md`


## Citation
If you use this code&data, please cite the papers below:

[Threading the Needle: Reweaving Chain-of-Thought Reasoning to Explain Human Label Variation](https://arxiv.org/abs/2505.23368)

```
@article{chen-2025-cot2el,
  author       = {Beiduo Chen and
                  Yang Janet Liu and
                  Anna Korhonen and
                  Barbara Plank},
  title        = {Threading the Needle: Reweaving Chain-of-Thought Reasoning to Explain
                  Human Label Variation},
  journal      = {CoRR},
  volume       = {abs/2505.23368},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2505.23368},
  doi          = {10.48550/ARXIV.2505.23368},
  eprinttype    = {arXiv},
  eprint       = {2505.23368},
  timestamp    = {Sun, 29 Jun 2025 21:59:19 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2505-23368.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```

## License 
The code under this repository is licensed under the [Apache 2.0 License](https://github.com/mainlp/CoT2EL/blob/main/LICENSE).

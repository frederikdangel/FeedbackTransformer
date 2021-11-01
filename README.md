# FeedbackTransformer

Modularization of:
@misc{patil2021-feedback-github,
    author       = {Rajaswa Patil},
    title        = {feedback-and-memory-in-transformers},
    month        = apr,
    year         = 2021,
    publisher    = {Github},
    url          = "https://github.com/rajaswa/feedback-and-memory-in-transformers"
    }
    
implementing

@misc{fan2021addressing,
      title={Addressing Some Limitations of Transformers with Feedback Memory}, 
      author={Angela Fan and Thibaut Lavril and Edouard Grave and Armand Joulin and Sainbayar Sukhbaatar},
      year={2021},
      eprint={2002.09402},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

and using the COGS Benchmark:

@inproceedings{kim-linzen-2020-cogs,
    title = "{COGS}: A Compositional Generalization Challenge Based on Semantic Interpretation",
    author = "Kim, Najoung  and
      Linzen, Tal",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.731",
    doi = "10.18653/v1/2020.emnlp-main.731",
    pages = "9087--9105",
    abstract = "Natural language is characterized by compositionality: the meaning of a complex expression is constructed from the meanings of its constituent parts. To facilitate the evaluation of the compositional abilities of language processing architectures, we introduce COGS, a semantic parsing dataset based on a fragment of English. The evaluation portion of COGS contains multiple systematic gaps that can only be addressed by compositional generalization; these include new combinations of familiar syntactic structures, or new combinations of familiar words and familiar structures. In experiments with Transformers and LSTMs, we found that in-distribution accuracy on the COGS test set was near-perfect (96{--}99{\%}), but generalization accuracy was substantially lower (16{--}35{\%}) and showed high sensitivity to random seed (+-6{--}8{\%}). These findings indicate that contemporary standard NLP models are limited in their compositional generalization capacity, and position COGS as a good way to measure progress.",
}

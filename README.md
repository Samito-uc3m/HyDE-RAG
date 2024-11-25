# Academic Research Assistance System

## Overview
A crucial part of research is comparing current work with previous studies (state-of-the-art review). This application assists researchers by allowing them to input a research topic, and the system provides relevant articles on the subject. Additionally, the system explains how existing work differs from the researcher's proposed ideas.

---

## Instalation
1. ```poetry install```
2. ```poetry shell```
3. ```pre-commit install```

---

## Functional Requirements
- **Article Retrieval:** The system retrieves relevant articles based on the research topic provided by the user.
- **Difference Analysis:** Identifies and displays key differences between the user's research topic and existing works.
- **No Fabrication of Articles:** If no relevant articles are found, the system does not invent articles to provide a response.
- **Language Consistency:** Responses are in the same language as the query. Article titles may remain in their original language.

---

## Interaction Example
- **Researcher:** I am researching federated implementations of neural topic models.
- **System:** I have found several relevant articles, including *"Federated Topic Modeling"* and *"Federated Nonnegative Matrix Factorization for Short Texts Topic Modeling with Mutual Information."* These works focus on federated implementations of Bayesian topic models, such as LDA or NMF, but none offer an implementation based on neural topic models.
- **Researcher:** I am researching systems for assisting researchers.
- **System:** I'm sorry, but I do not have specific articles on this topic in the current corpus. This does not imply that none exist. You might consider expanding your search in other databases.

---

## Supported Databases
The system relies on traditional academic databases, including:
- **ACL Articles:** Contains 80,000 articles/posters dated up to September 2022.
- **PubMed:** A collection of scientific articles related to medical and health sciences, available through its [official website](https://pubmed.ncbi.nlm.nih.gov/).
- **arXiv:** Kaggle offers a dataset with over 1.7 million abstracts and metadata from arXiv articles.

---

## Additional Features
- **Similarity Metrics:** Provides similarity metrics between the proposed research idea and the retrieved articles.
- **Automatic State-of-the-Art Classification:** Organizes retrieved articles into categories based on relevance, methodological approach, or contributions.
- **Automatic Summarization:** Generates summaries highlighting key points for each retrieved article using Transformer models and/or large language models (LLMs).
- **Multilingual Support:**  Handles queries in multiple languages, translating both questions and responses using pre-trained or fine-tuned translation models.

---

## License
MIT License

---

## Contact
- **Samito-uc3m** - 100429112@alumnos.uc3m.es

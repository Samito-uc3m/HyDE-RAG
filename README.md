# Academic Research Assistance System


## Description
A crucial part of academic research is comparing current work with previous studies (state-of-the-art review). This application streamlines this process by enabling researchers to input a research topic and receive relevant articles on the subject. Additionally, the system provides explanations on how existing work differs from the researcher's proposed ideas.

The system leverages **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLM)** for document filtering and text generation. This project was developed as part of the final assignment for the **Natural Language Processing course** in the **Master's in Artificial Intelligence** at **Universidad Carlos III de Madrid**.

---

## Table of Contents
1. [Description](#description)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Common Installation Issues](#common-installation-issues)
5. [Running the Project](#running-the-project)
6. [Usage Examples](#usage-examples)
7. [Project Structure](#project-structure)
8. [Technical Details and Configuration](#technical-details-and-configuration)
   1. [Database](#database)
   2. [Customizing the Database](#customizing-the-database)
   3. [Functional Requirements](#functional-requirements)
   4. [Additional Features](#additional-features)
      - [Similarity Metrics](#similarity-metrics)
      - [Translation and Multilingual Support](#translation-and-multilingual-support)
      - [Summaries for Retrieved Articles](#summaries-for-retrieved-articles)
9. [System Workflow](#system-workflow)
10. [License](#license)
11. [Contact](#contact)

---

## Technologies Used
- **Python 3.11** (Recommended: Python 3.11.11)
- **Poetry** (Dependency Management)
- **Streamlit** (Web Interface)
- **RAG Framework** (Document Retrieval)
- **Large Language Models (LLMs)** – GPT-4o Mini
- **bge-small-en** (Vectorization Model)
- **ChromaDB** (Database Backend)
- **LlamaIndex** (Indexing and Querying)
- **ArXiv Dataset** (Kaggle)

---

## Installation

To install the necessary dependencies, we use **Poetry**, which simplifies dependency management and project setup. We recommend using **pyenv** or similar tools to manage your Python version and ensure compatibility.

1. Install Poetry by following the official guide: [Poetry Installation Guide](https://python-poetry.org/docs/#installing-with-the-official-installer)

2. Verify that Poetry is using Python 3.11 (preferably version 3.11.11):
   ```bash
   poetry run python --version
   ```

3. Install project dependencies:
   ```bash
   poetry install
   ```

4. Set up your OpenAI API key:
   - **Using the terminal:**
     ```bash
     export OPENAI_API_KEY='your_openai_api_key'
     ```
   - **Using a .env file:** Create a `.env` file in the root directory and add the following line:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

Make sure you are in the project's root directory before executing these commands.

---

## Common Installation Issues

### Problem: Missing `sqlite3` Module

One common issue encountered during the installation of Python 3.11 is the absence of the `sqlite3` module, even if Python has been downloaded, compiled, and installed successfully. This can cause errors such as:

- **Error during installation or execution:** `ModuleNotFoundError: No module named 'sqlite3'`
- **Error in Streamlit:** `AppSession object has no attribute '_state'`

#### Cause
This issue occurs because `sqlite3` and its related development libraries were not available on the system at the time Python was compiled. Python relies on several external libraries during its build process, and missing dependencies can result in incomplete installations.

---

### Solution: Installing Required Dependencies

To resolve this issue, follow these steps:

1. **Install Required System Libraries**
   Use your package manager (e.g., `apt-get` for Ubuntu/Debian or `dnf` for Fedora) to install the necessary libraries. Execute the following command:

   #### Ubuntu/Debian:
   ```bash
   sudo apt-get install sqlite sqlite-devel gcc make zlib1g-dev bzip2 libbz2-dev libreadline-dev libssl-dev libtk8.6 libffi-dev xz-utils

   #### Fedora (Replace apt-get with dnf):
    ```bash
    sudo dnf install sqlite sqlite-devel gcc make zlib-devel bzip2 bzip2-devel readline readline-devel openssl openssl-devel tk-devel libffi-devel xz-devel
    ```

2. **Recompile and Install Python 3.11** After installing the required libraries, you need to recompile Python. We recommend version 3.11.11 for compatibility. Follow these steps:

3. **Verify Python Installation** After recompiling and installing Python, verify that the `sqlite3` module is available:
     ```bash
        python --version
        python -c "import sqlite3; print(sqlite3.sqlite_version)"
     ```
If the module is successfully imported and the version is displayed, the installation was successful.

---

## Running the Project

Once the dependencies are installed in the Poetry environment, run the application using the following command from the project's root directory:

```bash
poetry run streamlit run project/streamlit_app.py [max_docs]
```

- `max_docs` is an optional argument to limit the number of documents stored in the database. This can improve performance during testing, especially since the ArXiv database contains over 30,000 documents.

### Example with Document Limit
```bash
poetry run streamlit run project/streamlit_app.py 500
```

### Without Document Limit
If you prefer not to limit the number of documents:
```bash
poetry run streamlit run project/streamlit_app.py
```

---


## Usage Examples

### Database Loading Behavior
- Each time the user starts the web application, they need to load the database.
- **First Load:** When the **Load Database** button is pressed for the first time, the system downloads the database from Kaggle and stores up to `max_docs` documents (or all if no limit is set).
- **Subsequent Loads:** If a database is already present on the user's device, pressing **Load Database** will use the existing database, even if a different `max_docs` value is provided.
- **Reset Database:** To change the number of stored documents, the user must **manually delete the existing database** from their device.

### Querying the System
Once the database is loaded:
- Users can input queries such as:
  > *I am researching renormalized quasi-particles in antiferromagnetic states of the Hubbard model. Could you please find relevant documents?*
- The system will retrieve the most relevant papers from the database and compare them with the user's query.
- Queries can be performed in **any language supported by GPT-4o Mini**.

### Demo Video
A demonstration video showcasing the full capabilities of the system.

https://github.com/user-attachments/assets/f1eaeafc-23d9-4ffb-9202-c44a586ac501

### Test Screenshots

We have verified that the system is capable of identifying similarities between the user's query and the documents found in the database.

![imagen](https://github.com/user-attachments/assets/3d44513c-61d9-4cfa-b6d0-271192a9c5ee)

If no relevant documents are found, this is clearly indicated by the system.

![imagen](https://github.com/user-attachments/assets/b7e4f722-6c70-4491-9d14-22948d740528)

In addition to English and Spanish, we have confirmed that the system works with other languages, such as French.

![imagen](https://github.com/user-attachments/assets/6fbebaf8-4b8a-4162-af3b-2cbfe0a005a1)



---

## Project Structure
The project follows a modular structure to separate the main application logic from the data and models. The main components are organized as follows:
```
HYDE-RAG/
├── data/       # Loaded databases
├── models/     # Language detection models (e.g., FASTTEXT)
└── project/    # Main application modules
```

---

## Technical Details and Configuration

### Database
The system uses the **Cornell-University/arxiv** dataset, available on Kaggle, which includes metadata and abstracts of over **31,000 articles**. This ensures an extensive and diverse corpus to provide comprehensive and up-to-date coverage across various fields of academic research.

#### Customizing the Database
- You can change the dataset or modify how it is loaded by editing the file: `project/data_loader.py`.
- Ensure that any alternative dataset used contains abstracts and metadata relevant to academic articles to maintain the system's purpose and quality.


### Functional Requirements
- **Article Retrieval:** The system retrieves relevant articles based on the research topic provided by the user.
- **Difference Analysis:** Identifies and displays key differences between the user's research topic and existing works.
- **No Fabrication of Articles:** If no relevant articles are found, the system does not invent articles to provide a response.
- **Language Consistency:** Responses are in the same language as the query. Article titles may remain in their original language.

### Additional Features
To make the system more useful and accessible for all users, additional features have been implemented as follows:

#### Similarity Metrics
The system introduces similarity metrics between the user's query and the retrieved documents. These metrics provide greater clarity on the relevance of the results. This is achieved by leveraging similarity scores returned by the **RAG framework** and displaying them as percentages on the interface.

#### Translation and Multilingual Support
The system supports multilingual input and output. Queries are translated into the database's language (English) for processing, and the final responses are returned in the user's original language. The process includes:
1. **Language Detection:** A **FASTTEXT** model is used for quick and accurate detection of the input language.
2. **Translation:** Input queries are translated into English, and the output is configured to match the detected language. 
   - This enables the system to support all languages compatible with **GPT-4o Mini**, with particular emphasis on Spanish and English.

#### Summaries for Retrieved Articles
For each retrieved article, the **LLM** generates summaries highlighting the key points. This feature:
- Displays the similarity percentage for each document alongside the summary.
- Helps researchers quickly assess the relevance of each document without needing to read the full text.

### Summary of the Workflow

![imagen](https://github.com/user-attachments/assets/672cfdde-8b12-44a0-92c9-ad2feec1a768)

1. **Query Submission:** The user sends a query to the system.
2. **Language Detection:** The system detects the query language using the **FASTTEXT model**.
3. **Query Transformation:** The query is refined by **GPT-4o Mini**, removing unnecessary words and translating it into English if necessary.
4. **Document Retrieval:** Using **bge-small-en**, **ChromaDB**, and **LlamaIndex**, the system retrieves the top 3 most relevant documents.
5. **Response Generation:** The original query, detected language, and retrieved documents are sent to **GPT-4o Mini**, which generates a summary focusing on differences and key insights.
6. **Multilingual Support:** The final response is generated in the user's original language.

This workflow ensures precise results and seamless multilingual support.

---

## License
MIT License

---

## Contributors
- **Samito-uc3m** - correo?
- **JavGGon** - correo?
- Rafa
- Rodri

Problema SQLLite3
Hemos encontrado un problema al ejecutar el programa indicando que AppSession 

## License
This project is licensed under the [MIT License](LICENSE).

---

**Developed with passion for academic excellence.** ✍️🎓

For any questions or issues, feel free to contact the project maintainers.

---

**Universidad Carlos III de Madrid** | Master's in Artificial Intelligence | Natural Language Processing Course

<h1 align="center">RAG Badger</h1>
<p align="center">
<image align="center" src='./docs/images/rag_badger_logo_c.png' width="200" >
</p>

<p align="center"><b>Rag-Badger is a minimal and effective open source toolkit for RAG systems.</b><br>It focuses on the core signals that matter while judging RAG quality.</p>

## Key Features:
Rag-Badger provides the following evaluation metrics:
1. **Faithfulness:** To what extent the LLM is relying on the retrieved context to generate the answer?
2. **Factual correctness:** How close the generated answer is to ground truth or fact?
3. **Answer relevance:** How relevant the answer is with respect to the question? Or does the answer actually address the user query?
4. **Context relevance:** How relevant the retrieved context is with respect to the question?

## 📦 Installation steps:
**Get the distribution artifacts and source distributions in -> [Releases](https://github.com/adityapradhan202/Rag-Badger/releases)**
```
pip install rag_badger-0.0.1-py3-none-any.whl
```
> ❗It is highly recommended to install this package inside a virtual enviroment.
Donwload the distribution artificat in your project folder. **Activate the venv** and execute the pip command mentioned above. It will succesfully install the package inside your virtual environment and then you can delete this distribution artifact.

## 🚀 Quick Start:

**Some instructions:**  
1. **Recommended:** For the judge LLM use a model with atleast 7 billion parameters.
2. So far this toolkit has been tested with Ollama's local models only. So it is recommended to use Ollama's models for the evaluation.
3. The code for the basic usage is given below. To understand better checkout - **[examples/basic_usage.py](./examples/basic_usage.py)**
4. **Recommended** For evaluation use **atleast 20 questions, contexts, answers and ground_truths** to get consistent results.
> ⚠️ Make sure Ollama's HTTP server is running in the background to avoid connection errors.
```py
from rag_badger import Evaluate
from langchain_ollama import ChatOllama, OllamaEmbeddings

questions = []
contexts = []
answers = []
ground_truths = []

dataset = Evaluate(
    questions=questions,
    contexts=contexts,
    answers=answers,
    ground_truths=ground_truths
)

judge_model = ChatOllama(model='mistral', temperature=0.0)
score_dict_faithfulness = dataset.faithfulness(
    model=judge_model
)
score_dict_factual_correctness = dataset.factual_correctness(
    model=judge_model
)
score_dict_answer_relevance = dataset.answer_relevance(
    model=judge_model,
    embed_model=OllamaEmbeddings(model='nomic-embed-text:v1.5')
)
score_dict_context_relevance = dataset.context_relevance(
    model=judge_model
)

print("-> Evaluation complete!")
print("\n\n-> Here are the final results:\n\n")
print(f"Faithfulness: {score_dict_faithfulness['total_score']}")
print(f"Factual correctness: {score_dict_factual_correctness['total_score']}")
print(f"Answer relevance: {score_dict_answer_relevance['total_score']}")
print(f"Context relevance: {score_dict_context_relevance['total_score']}")
```

**Output format:**

Each of the four functions which are mentioned in the above code snippet return a python dictionary like this:
```
{   
    'total_score':0.875,
    'individual_scores':[0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 1.0]
}
```

## Observed Metric stability:
The code given in the [basic.py](./examples/basic_usage.py) builds a vector database using a custom sample dataset.

The script was excuted four times and evaluation metrics of each run were recorded. For the testing purpose a smaller dataset was created. The dataset had only 12 questions. However the results were remarkably consistent. **This proves that it will function properly and will be more precise and consistent in case of 20 or more than 20 questions, provided that a bigger and stronger LLM is used for judgement.**
|Runs|Faitfulness|Factual correctness|Answer relevance|Context relevance|
|----|-----------|-------------------|----------------|-----------------|
|1|0.4166|0.5|0.9583|0.33|
|2|0.416|0.5|0.91|0.33|
|3|0.41|0.5|0.87|0.33|
|4|0.41|0.5|0.9583|0.33|

## ⭐ Support the project
If you find Rag-Badger helpful, a star on the repository is greatly appreciated!


<h1 align="center">RAG Badger</h1>
<p align="center">
<image align="center" src='./docs/images/rag_badger_logo_c.png' width="200" >
</p>

<p align="center"><b>Rag-Badger is a minimal and effective open source toolkit for evaluating  Retrieval Augmented Generation(RAG) systems.</b><br>It focuses on the core signals that matter while judging RAG quality.</p>

### Key Features:
Rag-Badger provides the following evaluation metrics:
1. **Faithfulness:** To what extent the LLM is relying on the retrieved context to generate the answer?
2. **Factual correctness:** How close the generated answer is to ground truth or fact?
3. **Answer relevance:** How relevant the answer is with respect to the question? Or does the answer actually address the user query?
4. **Context relevance:** How relevant the retrieved context is with respect to the question?

### Quick Start:
**Some instructions:**  
1. Recommended: For the judge LLM use a model with atleast 7 billion parameters.
2. So far this toolkit has been tested with Ollama's local models only. So it is recommended to use Ollama's models for the evaluation.
3. The code for the basic usage is given below. To understand better checkout - **[examples/basic_usage.py](./examples/basic_usage.py)**
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

### License
**This repository is licensed under [MIT LICENSE](./LICENSE).**
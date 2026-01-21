# All the prompt templates here
from langchain_core.prompts import ChatPromptTemplate

prompt_t_faithfulness = ChatPromptTemplate.from_template(
    """
    You are a faithfulness score evaluator.

    Context:
    {context}

    Answer:
    {answer}

    Your task: Is the answer supported by the context or is the answer present in the context?
    Use discrete values to measure. Use either 0, 1 or 2. The meanings of these discrete numbers are mentioned below:
    
    0 - not supported
    1 - partially supported or somewhat present
    2 - supported and the answer is present
    """
)

prompt_t_factual_correctnes = ChatPromptTemplate.from_template(
    """
    You are a factual-correctness score evaluator.

    Ground truth:
    {ground_truth}

    Answer:
    {answer}

    Your task: Ground truth is a fact. Is this fact present in the answer.
    Use discrete values to measure the factual correctness of the answer. Use either 0, 1, or 2.
    The meanings of these discrete numbers are mentioned below:

    0 - The answer contradicts the ground truth or contains major factual errors
    1 - The answer is partially correct or has minor or some inaccuracies
    2 - The answer is factually correct and is consistent with ground truth
    """
)

prompt_t_answer_relevance = ChatPromptTemplate.from_template(
    """
    You are an answer to question convertor.

    Answer:
    {answer}

    Your task:
    Create an appropriate question from the given answer.
    Only return the question that you created in your response!
    """
)

prompt_t_context_relevance = ChatPromptTemplate.from_template(
    """
    You are an evaluator who calculates context relevance for RAG systems.

    Question:
    {question}

    Retrieved context:
    {context}

    Your task:
    Evaluate how relevant the retrieved context is for answering the question.
    Use discrete values to measure the context relevance. Use either 0, 1 or 2.
    The meanings of these discrete numbers are mentioned below:

    0 - The context is irrelevant and does not help in answering the question
    1 - The context is partially relevant but is missing important details
    2 - The context contains the information needed to answer the question

    Import instructions:
    - Do not judge the factual correctness.
    - Judge only the relevance of the context to the question.
    """
)
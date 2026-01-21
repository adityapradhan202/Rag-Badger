# Imports for the rag pipeline

# READ THIS ---> The code for the basic usage is written bellow - SCROLL DOWN!
# SCROLL DOWN TO SEE THE BASIC USAGE
# THE INITIAL PART OF THIS SCRIPT IS RELATED TO THE RAG PIPELINE

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from pydantic import BaseModel, Field

from langchain_chroma import Chroma
import os
from typing import Annotated, List, Literal

# IMPORT Evaluate class
from rag_badger import Evaluate

# --------------------- CODE FOR RAG --------------------------------------------------
loader = TextLoader(file_path='../sample_data/20_movies_detailed_descriptions.txt', encoding='utf-8')

print("-> Loading docs")
docs = loader.load()
# Docs is a list of langchain document objects

print("-> Creating chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)
print(f"Chunks length: {len(chunks)}")

#  Defining persistent directory path
db_dir = './chroma_database/'

def create_vector_store(embeddings:Annotated[OllamaEmbeddings, "embedding model"],
                        db_dir:Annotated[str, "path of persistent directory"],
                        chunks:Annotated[List[Document], "list of chunks"]) -> str:
    """Creates vector database if it doesn't exist"""

    if os.path.exists(db_dir):
        print("-> Path already exists")
        return
    
    print(f"-> Initializating vector db creation at {db_dir}")
    try:
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        db.add_documents(chunks)
        print("-> Succesfully create vector database")
    except Exception as e:
        print(f"Some exception occured: {e}")

# Embedding model
embeddings = OllamaEmbeddings(model='nomic-embed-text:v1.5')

# call the function
create_vector_store(
    embeddings=embeddings,
    db_dir=db_dir,
    chunks=chunks
)

def retrieve_docs(query:Annotated[str, "user query"],
                  db_dir:Annotated[str, "persistent directory for chroma"],
                  embeddings:Annotated[OllamaEmbeddings, "embedding model"]) -> str:
    """Retrieves related documents for the user query"""
    
    if not os.path.exists(db_dir):
        raise FileNotFoundError("Vector database doesn't exist!")
    
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    docs = db.similarity_search(
        query=query,
        k=2
        # 2 because the text file is just sample data and is way to simple
    )
    
    all_docs = []
    for doc in docs:
        all_docs.append(doc.page_content)
    output = " ".join(all_docs)
    return output

model = ChatOllama(model='gemma3:4b', temperature=0.8)

# Pydantic's BaseModel for structured output
class Classification(BaseModel):
    type:Literal["movie_related", "not_movie_related"] = Field(description="Is the query 'movie_related' or 'not_movie_related'")

def initial_classification(x:Annotated[dict, "dictionary with user query"]) -> dict:
    """Classifies the user query as 'movie_related' or 'not_movie_related'"""
    struct_model = model.with_structured_output(Classification)
    
    promp_t = ChatPromptTemplate.from_messages(
        messages=[
            ("system","classify user query into 'movie_related' or 'not_movie_related'"),
            ("system", "query must be strongly related to movies domain!"),
            ("user", "{query}")
        ]
    )
    chain = promp_t | struct_model
    class_obj = chain.invoke({'query':x['query']})
    # This Runnable sould return a string value for parallel branching's junction point
    return {
        'query':x['query'],
        'type':class_obj.type
    }

# Putting everything together - RAG pipeline
def retrieve_augment_generate(x:Annotated[dict, "output returned by initial chain"]) -> str:
    """Retrieve, augments and generate - the RAG process"""
    query = x['query']
    chunk = retrieve_docs(db_dir=db_dir, embeddings=embeddings, query=query)
    prompt_t = ChatPromptTemplate.from_template("This is the user query: {query} Generate answer from this context: {chunk}")
    chain = prompt_t | model | StrOutputParser()
    response_final = chain.invoke({'query':query, 'chunk':chunk})

    return response_final


rag_chain = RunnableLambda(retrieve_augment_generate)
default_prompt = ChatPromptTemplate.from_template("Tell the user that their query doesnt seem movie related, so you cant answer it")
default_chain = default_prompt | model | StrOutputParser()

junction_point = RunnableBranch(
    (lambda x: x['type'] == 'movie_related', rag_chain),
    default_chain
)
chain = (
    RunnableLambda(initial_classification)
    | junction_point
)

print("-> Final chain has been created")
print("-> Invoking chain for with a sample query")

print("-> Generating")
result = chain.invoke({'query':'Whose death is mentioned in the movie the lion king?'})

print("\n-> Here's the output of the rag pipeline")
print(result)
# -------------------------------------------------------------------------------------------------





# ---------------CODE FOR BASIC USAGE of RAG BADGER package------------------------------------


# FIRST CREATING SIMPLE DATASET
# THIS IS HOW WE CAN CREATE DATASET FOR EVALUATION
# Creating a sample dataset
print("-> Initializing data preparation")

# STEP-0 Create a set of questions, ground truths, also store the corresponding retrieved context and answer generated by rag in separate lists
questions = [
    "What is the movie inception about?",
    "What is the movie parasite about?",
    "Is there fighting scenes in the movie 'Fight Club'?",
    "What kind of relationship are jack and rose in the movie Titanic?",
    "Whose death is mentioned in the movie the lion king?",
    "What is the movie intersteller about?",
    "Tell me about the movie matrix!",
    "Tell me about the movie pulp finction",
    "Who is the main villain of the movie - Avengers End Game",
    "What is the plot of the movie - The Silence of lambs",
    "Why Aurthur Fleck in the movie Jocker loses his sanity eventually?",
    "What was Chris Gardener trying to achieve in the movie The pursuit of happyness"
]
ground_truths = [
    'Inception is about entering and manipulating dreams and stealing secrets within the dreams',
    "A poor family cleverly invades a wealthy household",
    "Narrator of the movie forms an underground fight club in the form of rebellion and self expression",
    "Titanic is a romanitc darama movie in which Jack and Rose face a shipwrek",
    "Simba becomes the lion king after the death of his father",
    "Earth is becoming slowly uninhabitable and a pilot travels through a wormwhole in search of a new home",
    "A computer hacker learns the truth about the matrix which is simulated environment for humans to live",
    "A crime film which shows how ordinary people get caught in violent and absurd sitatuations",
    "Avengers reunite to take down Thanos who has wiped out half of the life in the universe",
    "FBI trainee seeks help from an imprisoned serial killer to catch another murderer",
    "A mentally ill commedian faces constant rejection, poverty and negligence of society and loses his sanity",
    "A homeless salesman tries to pursue a carrer option through hardwork and perseverance to get a better life"
]

answers = []
contexts = []
for question in questions:
    context = retrieve_docs(
        query=question,
        db_dir=db_dir, embeddings=embeddings
    )
    contexts.append(context)

    answer = chain.invoke({'query':question})
    answers.append(answer)

print("\n-> Successfully created necessary components for creating dataset")
print("-> Initializing Rag evaluation pipeline")

# STEP-1 create dataset object
# Make sure to do: from rag_badger import Evaluate

dataset = Evaluate(
    questions=questions,
    contexts=contexts,
    ground_truths=ground_truths,
    answers=answers
)

# Get a judge model
# Is is recommended to use a powerful model with least temperature!
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
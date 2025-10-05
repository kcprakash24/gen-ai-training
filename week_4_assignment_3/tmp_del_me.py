# Agentic RAG System with LangGraph and MLflow
# Assignment 3: Self-Critique Loop with Azure GPT-4 mini

## 1. Installation and Imports

# !pip install langgraph langchain-openai langchain-pinecone pinecone-client mlflow python-dotenv -q

import os
import json
import time
import mlflow
import numpy as np
from typing import List, Dict, TypedDict, Annotated
from dotenv import load_dotenv

# LangChain
# LangChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangGraph
from langgraph.graph import StateGraph, END
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

print("✓ All imports successful")

## 2. Initialize Azure Models

client = AzureChatOpenAI(
    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT'],
    model_name="gpt-4o",
    temperature=0  # Temperature = 0 as per assignment
)

# Test LLM
test_response = client.invoke('What is the capital of France?').content
print(f"LLM Test: {test_response}")

embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
)

# Test Embeddings
test_emb = embedding_model.embed_query("The quick brown fox jumps over the lazy dog")
print(f"Embedding dimension: {len(test_emb)}")
print(f"Sample embedding values: {test_emb[:10]}")

emb_model_len = len(test_emb)

# Test similarity
sim = cosine_similarity(
    np.array(embedding_model.embed_query("It's a lovely day outside")).reshape(-1, 1),
    np.array(embedding_model.embed_query("The weather today is beautiful")).reshape(-1, 1)
)[0][0]
print(f"Cosine Similarity Test: {sim:.4f}")

print("\n✓ Azure models initialized successfully")

## 3. Note: Using GPT-4o for All Nodes

# Using the same Azure GPT-4o client for all nodes (Answer + Critique)
# No need for separate Gemini initialization

print("\n✓ Using GPT-4o for both answer generation and critique")

## 4. Load Dataset

# Load the knowledge base JSON
with open('self_critique_loop_dataset.json', 'r') as f:
    kb_data = json.load(f)

print(f"Loaded {len(kb_data)} knowledge base entries")
print(f"\nSample entry:")
print(json.dumps(kb_data[0], indent=2))

## 5. Initialize Pinecone Vector DB

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found. Please set it in your environment.")
else:
    print("✓ PINECONE API KEY found")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "agentic-rag-kb"
METRIC = "cosine"

# Clean up existing index
if INDEX_NAME in [idx["name"] for idx in pc.list_indexes()]:
    print(f"Deleting existing index '{INDEX_NAME}'...")
    pc.delete_index(INDEX_NAME)
    time.sleep(5)

# Create new index
print(f"Creating index '{INDEX_NAME}'...")
pc.create_index(
    name=INDEX_NAME,
    dimension=emb_model_len,
    metric=METRIC,
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
time.sleep(10)  # Wait for index to be ready

index = pc.Index(INDEX_NAME)
print(f"\n✓ Index created successfully")
print(index.describe_index_stats())

## 6. Prepare and Index Documents

# Create LangChain documents
documents = []
for entry in kb_data:
    doc_id = entry['doc_id']
    question = entry['question']
    answer = entry['answer_snippet']
    source = entry.get('source', 'unknown')
    
    # Combine question and answer for better semantic search
    content = f"Question: {question}\nAnswer: {answer}"
    
    doc = Document(
        page_content=content,
        metadata={
            'doc_id': doc_id,
            'question': question,
            'source': source
        }
    )
    documents.append(doc)

print(f"Prepared {len(documents)} documents for indexing")
print(f"\nSample document:")
print(f"  ID: {documents[0].metadata['doc_id']}")
print(f"  Content preview: {documents[0].page_content[:100]}...")

# Initialize vectorstore
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding_model,
    namespace=None,
    pinecone_api_key=PINECONE_API_KEY,
)

# Index documents
print("Indexing documents to Pinecone...")
vectorstore.add_documents(documents)
time.sleep(5)  # Wait for indexing

# Verify indexing
stats = index.describe_index_stats()
print(f"\n✓ Indexing complete. Total vectors: {stats['total_vector_count']}")

## 7. Define LangGraph State

class AgentState(TypedDict):
    """State that flows through the graph"""
    query: str
    retrieved_snippets: List[Dict]
    initial_answer: str
    critique_result: str
    refined_answer: str
    final_answer: str
    needs_refinement: bool
    refinement_snippet: Dict

print("✓ State schema defined")

## 8. Node 1: Retriever Node

def retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves top-5 KB snippets based on the query.
    Each snippet is up to 5 KB in size.
    """
    query = state['query']
    print(f"\n🔍 RETRIEVER NODE")
    print(f"Query: {query}")
    
    # Retrieve top 5 documents
    results = vectorstore.similarity_search_with_score(query, k=5)
    
    snippets = []
    for doc, score in results:
        snippet = {
            'kb_id': doc.metadata['kb_id'],
            'content': doc.page_content[:5000],  # Limit to 5KB
            'score': float(score)
        }
        snippets.append(snippet)
        print(f"  Retrieved: {snippet['kb_id']} (score: {score:.4f})")
    
    state['retrieved_snippets'] = snippets
    return state

print("✓ Retriever node defined")

## 9. Node 2: LLM Answer Node

def llm_answer_node(state: AgentState) -> AgentState:
    """
    Generates initial answer using Azure GPT-4 mini with citations.
    Citations format: [KBxxx]
    """
    query = state['query']
    snippets = state['retrieved_snippets']
    
    print(f"\n🤖 LLM ANSWER NODE")
    
    # Build context with citations
    context = ""
    for snippet in snippets:
        context += f"\n[{snippet['kb_id']}]:\n{snippet['content']}\n"
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided knowledge base snippets.

IMPORTANT: You MUST cite sources using the format [KBxxx] where xxx is the knowledge base ID.
- Use citations after each relevant statement
- Only use information from the provided snippets
- If information is not in the snippets, say so clearly"""),
        ("user", """Question: {query}

Knowledge Base Snippets:
{context}

Provide a comprehensive answer with proper citations [KBxxx].""")
    ])
    
    chain = prompt | client
    response = chain.invoke({"query": query, "context": context})
    
    answer = response.content
    state['initial_answer'] = answer
    
    print(f"Generated answer length: {len(answer)} characters")
    print(f"Answer preview: {answer[:200]}...")
    
    return state

print("✓ LLM Answer node defined")

## 10. Node 3: Self-Critique Node

def self_critique_node(state: AgentState) -> AgentState:
    """
    Uses GPT-4o to critique if the answer is COMPLETE or needs REFINE.
    """
    query = state['query']
    answer = state['initial_answer']
    
    print(f"\n🔍 SELF-CRITIQUE NODE")
    
    # Critique prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict answer quality evaluator. 
Evaluate if the answer is COMPLETE or needs REFINEMENT.

Return ONLY one word:
- "COMPLETE" if the answer fully addresses the question with sufficient detail
- "REFINE" if the answer is incomplete, vague, or missing important details

Be strict in your evaluation."""),
        ("user", """Question: {query}

Answer: {answer}

Evaluation (COMPLETE or REFINE):""")
    ])
    
    chain = prompt | client  # Using the same GPT-4o client
    response = chain.invoke({"query": query, "answer": answer})
    
    critique = response.content.strip().upper()
    
    # Ensure valid response
    if "COMPLETE" in critique:
        critique = "COMPLETE"
    elif "REFINE" in critique:
        critique = "REFINE"
    else:
        critique = "COMPLETE"  # Default to complete if unclear
    
    state['critique_result'] = critique
    state['needs_refinement'] = (critique == "REFINE")
    
    print(f"Critique Result: {critique}")
    
    return state

print("✓ Self-Critique node defined")

## 11. Node 4: Refinement Node

def refinement_node(state: AgentState) -> AgentState:
    """
    Retrieves 1 additional snippet and regenerates answer.
    """
    query = state['query']
    initial_snippets = state['retrieved_snippets']
    initial_answer = state['initial_answer']
    
    print(f"\n🔄 REFINEMENT NODE")
    
    # Get one more snippet (6th result)
    results = vectorstore.similarity_search_with_score(query, k=6)
    
    if len(results) > 5:
        doc, score = results[5]
        additional_snippet = {
            'kb_id': doc.metadata['kb_id'],
            'content': doc.page_content[:5000],
            'score': float(score)
        }
        print(f"  Additional snippet: {additional_snippet['kb_id']} (score: {score:.4f})")
    else:
        additional_snippet = None
        print("  No additional snippet available")
    
    state['refinement_snippet'] = additional_snippet
    
    # Build enhanced context
    all_snippets = initial_snippets.copy()
    if additional_snippet:
        all_snippets.append(additional_snippet)
    
    context = ""
    for snippet in all_snippets:
        context += f"\n[{snippet['kb_id']}]:\n{snippet['content']}\n"
    
    # Regenerate answer with additional context
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided knowledge base snippets.

IMPORTANT: You MUST cite sources using the format [KBxxx] where xxx is the knowledge base ID.
- Use citations after each relevant statement
- Only use information from the provided snippets
- Provide a more comprehensive answer than before"""),
        ("user", """Question: {query}

Previous Answer (needs improvement):
{previous_answer}

Enhanced Knowledge Base Snippets:
{context}

Provide an improved, more comprehensive answer with proper citations [KBxxx].""")
    ])
    
    chain = prompt | client
    response = chain.invoke({
        "query": query,
        "previous_answer": initial_answer,
        "context": context
    })
    
    refined_answer = response.content
    state['refined_answer'] = refined_answer
    
    print(f"Refined answer length: {len(refined_answer)} characters")
    
    return state

print("✓ Refinement node defined")

## 12. Decision Function

def decide_refinement(state: AgentState) -> str:
    """
    Decision function: routes to refinement or end based on critique.
    """
    if state['needs_refinement']:
        print("\n➡️  Routing to REFINEMENT")
        return "refine"
    else:
        print("\n➡️  Routing to END (answer complete)")
        return "end"

print("✓ Decision function defined")

## 13. Build LangGraph Workflow

# Create graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retriever", retriever_node)
workflow.add_node("llm_answer", llm_answer_node)
workflow.add_node("critique", self_critique_node)
workflow.add_node("refinement", refinement_node)

# Define edges
workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "llm_answer")
workflow.add_edge("llm_answer", "critique")

# Conditional edge based on critique
workflow.add_conditional_edges(
    "critique",
    decide_refinement,
    {
        "refine": "refinement",
        "end": END
    }
)

# After refinement, end
workflow.add_edge("refinement", END)

# Compile graph
app = workflow.compile()

print("\n✓ LangGraph workflow compiled successfully")
print("\nWorkflow structure:")
print("  1. Retriever → 2. LLM Answer → 3. Critique")
print("     ├─ If COMPLETE → END")
print("     └─ If REFINE → 4. Refinement → END")

## 14. Setup MLflow Tracking

mlflow.set_experiment("agentic_rag_self_critique")

print("\n✓ MLflow experiment set: 'agentic_rag_self_critique'")

## 15. Test Queries

test_queries = [
    "What are best practices for caching?",
    "How should I set up CI/CD pipelines?",
    "What are performance tuning tips?",
    "How do I version my APIs?",
    "What should I consider for error handling?"
]

print(f"\n✓ Loaded {len(test_queries)} test queries")

## 16. Run Evaluation on All Queries

print("\n" + "="*100)
print("STARTING EVALUATION - AGENTIC RAG SYSTEM")
print("="*100)

results = []

for i, query in enumerate(test_queries, 1):
    print(f"\n{'#'*100}")
    print(f"QUERY {i}/{len(test_queries)}")
    print(f"{'#'*100}")
    print(f"Question: {query}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"query_{i}"):
        
        # Log query
        mlflow.log_param("query", query)
        mlflow.log_param("query_number", i)
        
        # Initialize state
        initial_state = {
            'query': query,
            'retrieved_snippets': [],
            'initial_answer': '',
            'critique_result': '',
            'refined_answer': '',
            'final_answer': '',
            'needs_refinement': False,
            'refinement_snippet': None
        }
        
        # Run the graph
        final_state = app.invoke(initial_state)
        
        # Determine final answer
        if final_state['needs_refinement'] and final_state.get('refined_answer'):
            final_answer = final_state['refined_answer']
            answer_type = "refined"
        else:
            final_answer = final_state['initial_answer']
            answer_type = "initial"
        
        final_state['final_answer'] = final_answer
        
        # Log to MLflow
        mlflow.log_param("answer_type", answer_type)
        mlflow.log_param("critique_result", final_state['critique_result'])
        mlflow.log_param("num_retrieved_snippets", len(final_state['retrieved_snippets']))
        
        mlflow.log_metric("initial_answer_length", len(final_state['initial_answer']))
        if final_state.get('refined_answer'):
            mlflow.log_metric("refined_answer_length", len(final_state['refined_answer']))
        
        # Log snippets
        snippet_ids = [s['kb_id'] for s in final_state['retrieved_snippets']]
        mlflow.log_param("retrieved_kb_ids", ",".join(snippet_ids))
        
        # Log answers as artifacts
        with open("initial_answer.txt", "w") as f:
            f.write(final_state['initial_answer'])
        mlflow.log_artifact("initial_answer.txt")
        
        if final_state.get('refined_answer'):
            with open("refined_answer.txt", "w") as f:
                f.write(final_state['refined_answer'])
            mlflow.log_artifact("refined_answer.txt")
        
        # Display results
        print(f"\n{'='*100}")
        print(f"RESULTS FOR QUERY {i}")
        print(f"{'='*100}")
        print(f"\n📊 Critique: {final_state['critique_result']}")
        print(f"📝 Answer Type: {answer_type.upper()}")
        print(f"📚 Retrieved Snippets: {', '.join(snippet_ids)}")
        
        print(f"\n🎯 FINAL ANSWER:")
        print("-"*100)
        print(final_answer)
        print("-"*100)
        
        # Store result
        results.append({
            'query_num': i,
            'query': query,
            'critique': final_state['critique_result'],
            'answer_type': answer_type,
            'retrieved_snippets': snippet_ids,
            'final_answer': final_answer
        })

print("\n" + "="*100)
print("EVALUATION COMPLETE")
print("="*100)

## 17. Summary Report

import pandas as pd

summary_df = pd.DataFrame([
    {
        'Query #': r['query_num'],
        'Query': r['query'][:50] + "...",
        'Critique': r['critique'],
        'Answer Type': r['answer_type'],
        'Snippets Used': len(r['retrieved_snippets'])
    }
    for r in results
])

print("\n" + "#"*100)
print("SUMMARY REPORT")
print("#"*100)

print("\n", summary_df.to_string(index=False))

print(f"\n📊 Statistics:")
print(f"   Total Queries: {len(results)}")
print(f"   COMPLETE on first try: {sum(1 for r in results if r['critique'] == 'COMPLETE')}")
print(f"   Required REFINEMENT: {sum(1 for r in results if r['critique'] == 'REFINE')}")
print(f"   Average snippets per query: {sum(len(r['retrieved_snippets']) for r in results) / len(results):.1f}")

print("\n✓ All results logged to MLflow")
print(f"✓ View results: Run 'mlflow ui' in terminal")

print("\n" + "#"*100)
print("END OF EVALUATION")
print("#"*100)

## 18. Individual Query Analysis (Optional - Run for detailed review)

# Uncomment to analyze a specific query result
"""
query_to_analyze = 1  # Change this to analyze different queries (1-5)
result = results[query_to_analyze - 1]

print(f"\n{'='*100}")
print(f"DETAILED ANALYSIS - QUERY {query_to_analyze}")
print(f"{'='*100}")
print(f"\nQuestion: {result['query']}")
print(f"\nRetrieved KB IDs: {', '.join(result['retrieved_snippets'])}")
print(f"\nCritique Result: {result['critique']}")
print(f"Answer Type: {result['answer_type']}")
print(f"\nFinal Answer:")
print("-"*100)
print(result['final_answer'])
print("-"*100)
"""

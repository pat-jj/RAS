from load_data import load_hotpot_qa, load_adv_hotpot_qa, load_asqa, load_eli5, load_2wiki_multi_hop, load_pop_qa, load_pubhealth, load_qald10
from structuring import structure, load_t2t_model
from theme_scoping import theme_scoping
from verification import verify
from generation import generate_subquery, generate_answer
from doc_retrieval import doc_retrieval

# This is the main file for the framework




# Data Loading




# FIRAS Framework (query -> answer)
def run_firas(query, evolving_graph=None, top_k=5, iteration=0, dataset=None):
    print(f"Iteration {iteration}")
    print(f"Current Query/Subquery: {query}")
    
    ## Step 1: Theme Scoping
    target_documents = theme_scoping(query)
    ## Step 2: Text Retrieval
    retrieved_documents = doc_retrieval(query, target_documents, top_k=5)
    
    ## Steo 3.1: Structure (Text to Graph)
    triples = []
    model, tokenizer = load_t2t_model(model_path="pat-jj/text2triple-flan-t5")
    for document in retrieved_documents:
        triples.extend(structure(document, tokenizer, model))
        
    ## Step 3.2: Merge (Graph to Evolving Graph)
    if evolving_graph is None:
        evolving_graph = triples
    else:
        evolving_graph.extend(triples)

    ## Step 3.3 Verify (Evolving Graph, Query -> Boolean)
    is_answerable = verify(evolving_graph, query, dataset)

    
    if not is_answerable:
        ## (Step 3.4): Generate Sub-query and Start Next Iteration
        print("Generating Sub-query...")
        subquery = generate_subquery(evolving_graph, query, dataset)
        run_firas(subquery, evolving_graph, top_k=5, iteration=iteration+1)
    else:
        ## Step 4: Knowledge Serving and LLM Generation
        print("Generating Answer...")
        answer = generate_answer(evolving_graph, query, dataset)
        print("Answer: ", answer)


# Evaluation



if __name__ == "__main__":
    pass






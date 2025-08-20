import fastbmrag.fastbmrag as fastbmrag
from ast import literal_eval
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FastBioMedRAG is relationship-based Retrieval-Augmented Generation tool for deep analysis of biological papers.")
    parser.add_argument('--job', type=str, required=True, help="'either update' or 'query': to add new document or query", default='update')
    parser.add_argument('--document', type=str,  help="csv file name, it should have at least three columns: abstract, main_text, paper_id.  'main_text' is supposed to be a list[str]. If not, it will be splited by '\n' ", default="./example/demo.csv")
    parser.add_argument('--question', type=str,  help="The question for query", default="Which genes are associated with endometriosis?")
    parser.add_argument('--working_dir', type=str,  help="The directory to store database", default='./test')
    parser.add_argument('--collection_name', type=str, help="The database name", default='paper')
    parser.add_argument('--llm_model_name', type=str,help="The ollama LLM model ", default='phi4:latest')
    parser.add_argument('--embed_model_name', type=str, help="The ollama embedding model", default='mxbai-embed-large:latest')
    parser.add_argument('--embedding_smilarity', type=float,  help="The embedding similiarity cutoff", default=0.8)
    parser.add_argument('--top_match', type=int, help="The record number returned for query", default=20)

    args=parser.parse_args()
    
    rag=fastbmrag.RAG(working_dir=args.working_dir, 
                   collection_name=args.collection_name, 
                   llm_model_name=args.llm_model_name, 
                   embed_model_name=args.embed_model_name, 
                   embedding_smilarity=args.embedding_smilarity)
                   
    if args.job=='update':
        input_text = pd.read_csv(args.document, converters={'main_text': literal_eval})
        main_text=list(input_text['main_text'])
        if not isinstance(main_text[0], list):
            main_text=[ "\n".split(x) for x in main_text ]
        
        abstract=list(input_text['abstract'])
        paper_id=list(input_text['paper_id'])
        input_text=input_text.drop(['abstract','main_text'], axis=1)
        #print(input_text.columns)
        meta_info = input_text.to_dict(orient='records')
        #print(meta_info[0])
        documents=[ {'abstract': abstract[i], 'main_text': main_text[i], 'meta_info': meta_info[i] } for i in range(len(abstract))]
        #print(documents[0])
        rag.insert_paper(documents)
        print("Done")
    
    if args.job=='query':
        out=rag.query(args.question)
        print(out)
    
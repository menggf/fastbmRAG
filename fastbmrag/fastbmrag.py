import os
import json
from qdrant_client import QdrantClient,models
from qdrant_client.models import PointStruct, Distance, VectorParams
import asyncio
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cosine
from ollama import chat, embed, ChatResponse, AsyncClient

from fastbmrag.prompt import system_prompt_abstract, system_prompt_main_text, system_prompt_query, system_prompt_gene, system_prompt_disease, system_prompt_process_query, system_prompt_query2
from fastbmrag.utils import chunking_by_word_size, safe_unicode_decode, locate_json_string_body_from_string, standarized_llm_df, chat_llm, embed_llm
from fastbmrag.utils import convert_response_to_json, remove_thinking

import logging

logging.basicConfig(
    filename="log_" + str(datetime.now().timestamp()) + '.log',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


class RAG:
    '''
    FastBioMedRAG is Retrieval-Augmented Generation tool for deep analysis of biological/medical papers. 
    The key idea is extract the entities from abstract of scientific paper and then extract the 
    relationship from main-text section (if no main-text available, abstract will be used again).  
    
    The input document should be formatted like 
        [{'abstract':'....', 'main_text':'...', 'meta_info':{'paper_id':[1234,1234,...], 'journal':'...'}},
         {'abstract':'....', 'main_text':'...', 'meta_info':{'paper_id':[1234,1234,...], 'journal':'...'}},
         ...
        ]
    Among them, 'abstract','main_text','meta_info' and 'paper_id' are mendatory.
    
    import FastBioMedRAG
    import pandas as pd
    
    # initilize an RAG:
    rag=FastBioMedRAG.RAG(working_dir="test",                              # the working directory 
                   collection_name="paper"                                 # the database name to store indexed graphs
                   llm_index_model_name="phi4:latest",                     # LLM index model
                   llm_query_model_name="gemma3:27b-it-qat",               # LLM query model
                   embed_model_name="mxbai-embed-large:latest",            # embedding model
                   embedding_smilarity=0.8                                 # the retrieval cutoff of embedding similarity
                   )
    documents=[]                                                           # input data, extract info can be stored in 'meta_info'
    
    # add document into database
    rag.insert_paper(documents)                                            # insert paper and related information into database
    
    # query
    out=rag.query("What is the causal mechanism of endometriosis?")        # query again database
    print(out)
    
    Note: Current, only support ollama LLM and embedding models.  
    '''
    def __init__(self,
                 working_dir=f"./FastBioMedRAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
                 collection_name="paper",
                 llm_index_model_name="phi4:latest",
                 llm_query_model_name="gemma3:27b-it-qat",
                 embed_model_name="mxbai-embed-large:latest",
                 embed_size=1024,
                 embedding_smilarity=0.8):
        self.working_dir=working_dir
        self.llm_index_model_name=llm_index_model_name
        self.llm_query_model_name=llm_query_model_name
        self.embed_size=embed_size
        
        self.embed_model_name=embed_model_name
        self.embedding_smilarity=embedding_smilarity
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.collection_name=collection_name
        self.client=QdrantClient(path=working_dir+"/"+ collection_name)
        #print(list(self.client.get_collections().collections))
        if not any( [self.collection_name == x.name for x in self.client.get_collections().collections] ):
            collection = self.client.create_collection(collection_name=self.collection_name,
                    vectors_config = models.VectorParams(size=embed_size, distance=Distance.COSINE)
                )
        
    
    
    def process_abstract_async(self, abstracts, has_main_text, temprature=[0.3, 0.7]):
        messages=[
                [ { 'role': 'system', 'content': system_prompt_abstract},
                  { 'role': 'user', 'content': 
                  "Extract the relationship of entities according to following information: " + str([ab,]) 
                  }]
                for ab in abstracts]
        paper_temprature=[temprature[0] if x=="Yes" else temprature[1] for x in has_main_text]
        
        responses= asyncio.run(chat_llm(self.llm_index_model_name, messages, paper_temprature))
        return responses
    
    def summary_abstract_async(self, abstracts, temprature=0.8):
        messages=[
                [ { 'role': 'system', 'content': "You are a biomedical research assistant specialized in biology and medicine"},
                  { 'role': 'user', 'content': 
                  "Summary key findings into one paragraph using following information: " + str([ab,]) 
                  }]
                for ab in abstracts]
        
        #paper_temprature=[temprature[0] if x=="Yes" else temprature[1] for x in has_main_text]
        responses= asyncio.run(chat_llm(self.llm_index_model_name, messages))
        return responses
            
    def llm_output_to_df(self, llm_out, has_main_text):
        new_llm_out=[locate_json_string_body_from_string(xx) for xx in llm_out]
        df=[pd.DataFrame(xx) for xx in new_llm_out]
        df=[standarized_llm_df(xx) for xx in df]
        for i in range(len(df)):
            df[i]["wh_paper"]=i
            df[i]['tag']=list(range(df[i].shape[0]))
            df[i]['has_main_text']= has_main_text[i]
        return pd.concat(df, axis=0)
    
     
    def subset_maintext_async(self, df_abstract_output, main_texts, has_main_text):
        df_main_texts=pd.concat([ pd.DataFrame({'wh_paper': i, 'main_txt': main_texts[i]}) for i in range(len(main_texts)) if has_main_text[i]=="Yes"  ])
        new_main_texts=list(df_main_texts['main_txt'])
        
        main_text_embed =  asyncio.run(embed_llm(self.embed_model_name, new_main_texts ))
        
        df_abstract_output_sub1=df_abstract_output[df_abstract_output['has_main_text']=="Yes"].copy()
        df_abstract_output_sub2=df_abstract_output[df_abstract_output['has_main_text']=="No"].copy()
        
        queries_item=["".join(["what is the relationship or interaction between ", str(x) , " and ", str(y) ,"?"])
            for x, y in zip(list(df_abstract_output_sub1['source_entity']), list(df_abstract_output_sub1['target_entity']))]
        
        queries_embed=  asyncio.run(embed_llm(self.embed_model_name, queries_item))   
                      
        res=[]
        for cc in range(len(queries_embed)):                   
            which_paper=df_abstract_output_sub1.iloc[cc]["wh_paper"].item()
            wh_main_text= [i for i, val in enumerate(df_main_texts['wh_paper']) if val == which_paper]
            #wh_main_text=df_main_texts['wh_paper'].index(which_paper)
            select_main_text_embed=[main_text_embed[i] for i in wh_main_text]
            select_main_text      =[new_main_texts[i]  for i in wh_main_text]
            sims =  [(1 - cosine(queries_embed[cc], fb )).item() for fb in select_main_text_embed ]
            sim_item = [ x for x in sims if x > self.embedding_smilarity ]
            if len(sim_item)==0:
                new_text_list=[select_main_text[ sims.index(max(sims)) ] ]
            else:
                new_text_list=[select_main_text[sims.index(x)] for x in sim_item if x > self.embedding_smilarity]
            new_text= str(new_text_list)
            res.append(new_text)
        
        df_abstract_output_sub1['main_text']=res
        df_abstract_output_sub2['main_text']=[str([])]* df_abstract_output_sub2.shape[0]
        return pd.concat([df_abstract_output_sub1, df_abstract_output_sub2], axis=0)
    
    
    def refine_with_maintext_async(self, df_ab, temprature=0.7):
        df_ab_sub1=df_ab[df_ab['has_main_text']=="Yes"].copy()
        df_ab_sub2=df_ab[df_ab['has_main_text']=="No"].copy()
        user_query = [ "".join(["Extract the relationship or interaction between ", str(x) ," and ", str(y), \
                            " into one brief and short paragraph according to following information: ", str([z])]) 
                for x, y, z in zip(list(df_ab_sub1['source_entity']), list(df_ab_sub1['target_entity']), list(df_ab_sub1['main_text']))]
        
        messages=[ [ { 'role': 'system', 'content': system_prompt_main_text},
                           { 'role': 'user', 'content': qry} ]
                  for qry in user_query ]
                  
        current_temprature =[temprature]*len(messages)       
        responses= asyncio.run(chat_llm(self.llm_index_model_name, messages, current_temprature))
        
        df_ab_sub1['updated_relationship_description']=responses
        df_ab_sub2['updated_relationship_description']=list(df_ab_sub2['relationship_description'])
        df_ab = pd.concat([df_ab_sub1, df_ab_sub2], axis=0)
        df_ab=df_ab.drop(columns=['main_text'], axis=1) 
        return df_ab 
    
    def save_to_csv(self, df, file_name):
        if os.path.exists(file_name):
            df.to_csv(os.path.join(self.working_dir, file_name))
        else:
            df.to_csv(os.path.join(self.working_dir, file_name), mode="a")
        
    def insert_paper(self, documents):
        
        abstracts=[safe_unicode_decode(doc) for doc in list(documents['abstract'])]
        main_texts=list(documents['main_text'] )
        try:
            main_texts=[eval(txt) for txt in main_texts]
        except SyntaxError:
            pass
        except Exception as e:
            pass
                    
        main_texts=[txt if isinstance(txt, list) else [ safe_unicode_decode(x) for x in str(txt).split("\n") ]
                    for txt in main_texts ]
                    
        has_main_text=["No" if not main_texts[i] or len(str(main_texts[i])) < 30  else "Yes" for i in range(len(main_texts)) ]
        
        tmp_doc=documents.drop('main_text', axis=1)
        tmp_doc=tmp_doc.drop('abstract', axis=1)
        
        meta_info=tmp_doc.to_dict(orient='records')
        paper_id=list(documents['paper_id'])
        
        has_meta=self.client.query_points(collection_name=self.collection_name, query=None,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(key="paper_id", match=models.MatchAny(any=paper_id))]
                    ),
                    limit=len(paper_id) +100,
                    with_payload=["paper_id"]
                   ).points
        
        if has_meta:
            has_ids = [x.payload['paper_id'] for x in has_meta]
            has_index = [i for i in range(len(paper_id)) if paper_id[i] not in has_ids]
            meta_info=[meta_info[x] for x in has_index]
            abstracts=[abstract[x] for x in has_index]
            main_texts=[main_texts[x] for x in has_index]
            has_main_text=[has_main_text[x] for x in has_index]
            
        if len(main_texts)==0:
            logging.info("All paper exists")
            print("All paper exists")
            return None
        
        logging.info("Process abstract " )
        output_abstract = self.process_abstract_async(abstracts, has_main_text, temprature=[0.3, 0.7])
        #sum_abstract = self.summary_abstract_async(abstracts, temprature=0.8)
        
        
        logging.info("transform LLM output into DataFrome ")
        df_abstract = self.llm_output_to_df(output_abstract, has_main_text)
        
        self.save_to_csv(df_abstract, "df_abstract_v0.csv")
        
        logging.info("Extract related main text ")
        df_abstract = self.subset_maintext_async(df_abstract, main_texts, has_main_text)
        
        self.save_to_csv(df_abstract, "df_abstract_v1.csv")
        
        logging.info("Refine with related main text ")
        df_abstract = self.refine_with_maintext_async(df_abstract, temprature=0.7)

        logging.info("write to local file ")
        df_meta=pd.DataFrame(meta_info)
        df_meta['wh_paper']=list(range(len(meta_info)))
        if not "weight" in list(df_meta.columns):
            df_meta['weight']=-1
        
        self.save_to_csv(df_abstract, "df_abstract_v2.csv")
        self.save_to_csv(df_meta, "df_meta_v2.csv")
        
        logging.info("Generate meta information records ")
        input_df=pd.merge(df_meta, df_abstract,  on='wh_paper',how='inner' )
        input_df=input_df.drop(columns=['wh_paper'], axis=1)  
        paper_id=list(input_df["paper_id"])
        tag_id=list(input_df["tag"])
        input_df=input_df.drop(columns=['tag'], axis=1)
        if all([isinstance(x, int) for x in paper_id]):
            document_id=[ int(paper_id[i] * (tag_id[i]+1)) for i in range(len(tag_id))]
        else:
            document_id=[int(hash(paper_id[i]) * (tag_id[i]+1)) for i in range(len(tag_id))]
        
        self.save_to_csv(input_df, "input_df.csv")
        logging.info("Generate embed of relationship ")
        
        text_embedding=[embed(model=self.embed_model_name, input=txt).embeddings[0] 
                         if isinstance(txt, str) and len(txt) > 10  else [0.0] * self.embed_size   
                         for txt in list(input_df['updated_relationship_description'])  ]
                            
        #out=list()
        #for i,txt in enumerate( list(input_df['updated_relationship_description']) ):
        #    print(i)
        #    if isinstance(txt, str):
        #        continue
        #    out.append(embed(model=self.embed_model_name,input=safe_unicode_decode(txt)).embeddings[0])
        
        logging.info("Store data into vector database ")
        input_df['relationship_type']=[str(x) for x in list(input_df['relationship_type'])]
        input_df['source_entity']=[str(x) for x in list(input_df['source_entity'])]
        input_df['target_entity']=[str(x) for x in list(input_df['target_entity'])]
        input_df['source_entity_type']=[str(x) for x in list(input_df['source_entity_type'])]
        input_df['target_entity_type']=[str(x) for x in list(input_df['target_entity_type'])]
        input_dict=input_df.to_dict(orient='records')
        #print(input_dict[0])
        
        new_data=[PointStruct(id=document_id[i], vector=text_embedding[i], payload= input_dict[i]) 
                        for i in range(len(document_id))]
        
        
        res=self.client.upsert(collection_name=self.collection_name, wait=True, points=new_data)
        
        logging.info("Insert document into database, done ")
        
    def query(self, question: str, top_results:int=300, gene:list[str]=[], disease:list[str]=[], paper_id:list[int]=[], question_analysis=True, filter_importance= -1 ,temprature:float=0.75,  similarity_score:float=0.75):
        
        if (not gene) and (not disease) and (not paper_id) and question_analysis:
            question_query="Extract the offical HGNC symbols from following sentence: " + str([question])
            message_gene=[ { 'role': 'system', 'content': system_prompt_gene}, { 'role': 'user', 'content': question_query} ]
            #gene_text = chat(model=self.llm_query_model_name, messages=message_gene, options={"temperature": temprature})['message']['content']
            question_query="Extract the MESH disease term from following sentence: " + str([question])
            message_disease=[ { 'role': 'system', 'content': system_prompt_disease}, { 'role': 'user', 'content': question_query} ]
            #disease_text = chat(model=self.llm_query_model_name, messages=message_disease, options={"temperature": temprature})['message']['content']
            responses= asyncio.run(chat_llm(self.llm_query_model_name, [message_gene, message_disease], [temprature,temprature]))
            #print(responses[0])
            #print(responses[1])
            #print(remove_thinking(responses[0].strip()))
            #print(remove_thinking(responses[1].strip()))
            
            gene= eval(remove_thinking(responses[0].strip()))
            disease= eval(remove_thinking(responses[1].strip()))
            print("Query of Gene: "+ str(gene) + " and disease: "+ str(disease))
        
        data_df=pd.read_csv(self.working_dir+ "/" + "input_df.csv")
        if filter_importance > 0:
            ### calculate weight
            G = nx.from_pandas_edgelist(data_df, source='source_entity', target='target_entity')
            degrees=G.degree(weight="weight")

        tmp_list=list(data_df['source_entity_type']) + list(data_df['target_entity_type'])     
        node_all_types= list(pd.DataFrame({'A': tmp_list}).groupby('A').size().sort_values(ascending=False).keys())[0:40]
        process_query= "The question is: " + safe_unicode_decode(question) +"\n" +  "The label list is as following: " + str(node_all_types) 
        process_messages=[ { 'role': 'system', 'content': system_prompt_process_query}, { 'role': 'user', 'content': process_query} ]
        process_response = chat(model=self.llm_query_model_name, messages=process_messages, options={"temperature": 0.1})['message']['content']
        process_response=remove_thinking(process_response)
        #print(process_response)
        
        try:
            output_types=eval(process_response)
        except Exception as e:
            print("Sorry that something wrong with LLM output. Please re-query again")
            return null
        
        question_embed= embed(model=self.embed_model_name, input=safe_unicode_decode(question)).embeddings[0]
        filter_vector=gene + disease
        
        #print(filter_vector)
        if len(gene)!=0 and len(disease)!=0:
            filter=models.Filter(
                must=[ models.Filter(
                        should=[models.FieldCondition(
                                    key='source_entity',
                                    match=models.MatchAny(any = gene)
                                ),
                                models.FieldCondition(
                                    key='target_entity',
                                    match=models.MatchAny(any = gene)
                                ),
                            ]),
                        models.Filter(
                         should=[models.FieldCondition(
                                    key='source_entity',
                                    match=models.MatchAny(any = disease)
                                ),
                                models.FieldCondition(
                                    key='target_entity',
                                    match=models.MatchAny(any = disease)
                                ),
                            ]),
                        models.Filter(
                         should=[models.FieldCondition(
                                    key='source_entity_type',
                                    match=models.MatchAny(any = output_types)
                                ),
                                models.FieldCondition(
                                    key='target_entity_type',
                                    match=models.MatchAny(any = output_types)
                                ),
                            ]),
                            
                      ]        
            )
            output_query=self.client.query_points(collection_name=self.collection_name,
                query=question_embed, with_payload=True,
                limit=top_results,
                with_vectors=False,
                score_threshold=similarity_score,
                query_filter=filter
            ).points
        elif len(gene)!=0 or len(disease)!=0:
            filter=models.Filter(
                must=[ models.Filter(
                        should=[models.FieldCondition(
                                    key='source_entity',
                                    match=models.MatchAny(any = filter_vector)
                                ),
                                models.FieldCondition(
                                    key='target_entity',
                                    match=models.MatchAny(any = filter_vector)
                                ),
                            ]),
                        models.Filter(
                         should=[models.FieldCondition(
                                    key='source_entity_type',
                                    match=models.MatchAny(any = output_types)
                                ),
                                models.FieldCondition(
                                    key='target_entity_type',
                                    match=models.MatchAny(any = output_types)
                                ),
                            ]),
                            
                      ]        
            )
            output_query=self.client.query_points(collection_name=self.collection_name,
                query=question_embed, with_payload=True,
                limit=top_results,
                with_vectors=False,
                score_threshold=similarity_score,
                query_filter=filter
            ).points
        else:
            filter=models.Filter(
                         should=[models.FieldCondition(
                                    key='source_entity_type',
                                    match=models.MatchAny(any = output_types)
                                ),
                                models.FieldCondition(
                                    key='target_entity_type',
                                    match=models.MatchAny(any = output_types)
                                ),
                  ])
                           
            output_query=self.client.query_points(collection_name=self.collection_name,
                query=question_embed,
                with_vectors=False,
                score_threshold=similarity_score,
                limit=top_results,
                query_filter=filter
            ).points    
        
        #print(output_query)
        metadatas=[x.payload for x in output_query]
        similar_score=[x.score for x in output_query]
        if len(metadatas)==0:
            return "No related paper or record finding in current database!"
        
        df_metadatas = pd.DataFrame(metadatas)
        
        
        if filter_importance > -1:
            start_node=list(df_metadatas['source_entity'])
            end_node  =list(df_metadatas['target_entity'])
            edge_weight=[degrees[start_node[i]] + degrees[start_node[i]] for i in range(len(start_node))]
            min_edge_weight=[min(degrees[start_node[i]], degrees[start_node[i]]) for i in range(len(start_node))]
            df_metadatas['edge_weight']=edge_weight
            df_metadatas['min_edge_weight']=min_edge_weight
            importance_cutoff = np.quantile(min_edge_weight, [filter_importance])
            df_metadatas= df_metadatas[df_metadatas['min_edge_weight'] > importance_cutoff ]

        #df_metadatas.sort_values(by=['edge_weight','min_edge_weight'], ascending=False)

        match_metadatas=df_metadatas[['paper_id','journal','year',"weight"]].drop_duplicates(subset=['paper_id'])
        match_text= list(df_metadatas['updated_relationship_description'])
        paper_ids= list(df_metadatas['paper_id'])
        paper_weight=list(df_metadatas['weight'])
        input_text=[{'id': paper_ids[i], 'weight':paper_weight, 'text': match_text[i]} for i in range(len(match_text))]

        
        input_step=20
        messages=list()
        chunk_paper_ids=list()
        for i in range(0,len(input_text),input_step):
            user_query= "According to input json: " + str(match_text[i:min(i + input_step, len(match_text))]) +"\n\n  Please answer following question: " + question 
            messages.append([ { 'role': 'system', 'content': system_prompt_query}, { 'role': 'user', 'content': user_query} ])
            chunk_paper_ids.append(paper_ids[i:min(i + input_step, len(match_text))])
            
        current_temprature=[temprature] *len(messages)
        output_response= asyncio.run(chat_llm(self.llm_query_model_name, messages, current_temprature))


        merge_query= "For the question of " + str([question]) +"\n"+"LLM gives mutliple outputs using different resources. Merge and summarize them as an updated response using following LLM outputs: " +  str(output_response)
        response = chat(model=self.llm_query_model_name, messages=[{'role':'user', 'content': merge_query}], options={"temperature": temprature})
        llm_output= remove_thinking(response['message']['content'])
        output={"outcome": llm_output,'reference':match_metadatas, "chunked_outcome":pd.DataFrame({'outcome': output_response, 'paper_id': chunk_paper_ids})}

        #user_query= "According to input json: " + str(input_text) +"\n\n  Please answer following question: " + question 
        #messages=[ { 'role': 'system', 'content': system_prompt_query}, { 'role': 'user', 'content': user_query} ]
        #response = chat(model=self.llm_query_model_name, messages=messages, options={"temperature": temprature})
        #llm_output= remove_thinking(response['message']['content']) + "\n\n\n" + str(match_metadatas)
        
        return output


if __name__ == '__main__':
    import FastBioMedRAG
    import argparse
    parser = argparse.ArgumentParser(description="FastBioMedRAG is relationship-based Retrieval-Augmented Generation tool for deep analysis of biological papers.")
    parser.add_argument('--job', type=str, required=True, help="'either update' or 'query': to add new document or query", default='update')
    parser.add_argument('--document', type=str,  help="csv file name, it should have at least three columns: abstract,main_text,paper_id")
    parser.add_argument('--question', type=str,  help="The question for query")
    parser.add_argument('--working_dir', type=str,  help="The directory to store database", default='./test')
    parser.add_argument('--collection_name', type=str, help="The database name", default='paper')
    parser.add_argument('--llm_index_model_name', type=str,help="The ollama LLM model ", default='phi4:latest')
    parser.add_argument('--llm_query_model_name', type=str,help="The ollama LLM model ", default='gemma3:27b-it-qat')
    parser.add_argument('--embed_model_name', type=str, help="The ollama embedding model", default='phi4:latest')
    parser.add_argument('--vector_smilarity', type=float,  help="The embedding similiarity cutoff", default=0.75)
    parser.add_argument('--top_match', type=int, help="The record number returned for query", default=300)
    parser.add_argument('--temprature', type=float, help="The tempreture of LLM model", default=0.7)
    parser.add_argument('--question_analysis', type=bool, help="The tempreture of LLM model", default=True)

    args=parser.parse_args()
    
    rag=FastBioMedRAG.RAG(working_dir=args.working_dir, 
                   collection_name=args.collection_name, 
                   llm_index_model_name=args.llm_index_model_name, 
                   llm_query_model_name=args.llm_query_model_name, 
                   embed_model_name=args.embed_model_name, 
                   embedding_smilarity=args.vector_smilarity)
    if args.job=='update':
        input_text = pd.read_csv(args.document)
        
        #abstract=list(input_text['abstract'])
        #main_text=list(input_text['main_text'])
        
        #paper_id=list(input_text['paper_id'])
        #input_text=input_text.drop('abstract', axis=1)
        #input_text=input_text.drop('main_text', axis=1)
        #meta_info = input_text.to_dict(orient='records')
        #documents=[ {'abstract': abstract[i], 'main_text': main_text[i], 'meta_info':meta_info[i] } for i in range(abstract)]
        rag.insert_paper(input_text)
        print("Done")
    
    if args.job=='query':
        out=rag.query(args.question, top_results=args.top_match, similarity_score=args.vector_smilarity, temprature=args.temprature, question_analysis=args.question_analysis)
        print(out)
    
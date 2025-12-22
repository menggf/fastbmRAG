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
from tqdm import tqdm

from fastbmrag.prompt import system_prompt_abstract, system_prompt_main_text, system_prompt_query, system_prompt_gene, system_prompt_disease, system_prompt_process_query, system_prompt_query2
from fastbmrag.utils import chunking_by_word_size, safe_unicode_decode, locate_json_string_body_from_string, standarized_llm_df, chat_llm, embed_llm
from fastbmrag.utils import convert_response_to_json, remove_thinking

import logging
from typing import List, Dict, Any, Optional, Tuple, Union

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
                   embedding_similarity=0.8                                 # the retrieval cutoff of embedding similarity
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
                 working_dir: str = f"./FastBioMedRAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
                 collection_name: str = "paper",
                 llm_index_model_name: str = "phi4:latest",
                 llm_query_model_name: str = "gemma3:27b-it-qat",
                 embed_model_name: str = "mxbai-embed-large:latest",
                 embed_size: int = 1024,
                 embedding_similarity: float = 0.8,
                 llm_semaphore_limit: int = 2,
                 embed_semaphore_limit: int = 2,
                 backend: str = "ollama") -> None:
        self.working_dir: str = working_dir
        self.llm_index_model_name: str = llm_index_model_name
        self.llm_query_model_name: str = llm_query_model_name
        self.embed_size: int = embed_size
        
        self.embed_model_name: str = embed_model_name
        self.embedding_similarity: float = embedding_similarity
        self.collection_name: str = collection_name
        self.llm_semaphore_limit: int = llm_semaphore_limit
        self.embed_semaphore_limit: int = embed_semaphore_limit
        self.backend: str = backend
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.client: QdrantClient = QdrantClient(path=os.path.join(working_dir, collection_name))

        if not any( [self.collection_name == x.name for x in self.client.get_collections().collections] ):
            collection = self.client.create_collection(collection_name=self.collection_name,
                    vectors_config = models.VectorParams(size=embed_size, distance=Distance.COSINE)
                )
        
    
    
    def process_abstract_async(self, abstracts: List[str], has_main_text: List[str], temperature: Union[List[float], float] = [0.3, 0.7]) -> List[str]:
        """
        Process abstracts asynchronously to extract entity relationships.
        
        Args:
            abstracts (List[str]): List of paper abstracts to process.
            has_main_text (List[str]): List indicating if each paper has main text ("Yes"/"No").
            temperature (Union[List[float], float]): Temperature for LLM generation. 
                If list, first value used for papers with main text, second for others.
                If single value, used for all papers.
        
        Returns:
            List[str]: List of LLM responses containing entity relationships.
        """
        messages: List[List[Dict[str, str]]] = [
                [ { 'role': 'system', 'content': system_prompt_abstract},
                  { 'role': 'user', 'content': 
                  "Extract the relationship of entities according to following information: " + str([ab,]) 
                  }]
                for ab in abstracts]
        paper_temperature: List[float] = [temperature[0] if isinstance(temperature, list) and x=="Yes" else temperature[1] if isinstance(temperature, list) else temperature for x in has_main_text]
        
        responses: List[str] = asyncio.run(chat_llm(self.llm_index_model_name, messages, paper_temperature, semaphore_limit=self.llm_semaphore_limit, backend=self.backend))
        return responses
    
    def summary_abstract_async(self, abstracts: List[str], temperature: Union[List[float], float] = 0.8) -> List[str]:
        """
        Generate summaries of abstracts asynchronously.
        
        Args:
            abstracts (List[str]): List of paper abstracts to summarize.
            temperature (Union[List[float], float]): Temperature for LLM generation.
        
        Returns:
            List[str]: List of summarized abstracts.
        """
        messages: List[List[Dict[str, str]]] = [
                [ { 'role': 'system', 'content': "You are a biomedical research assistant specialized in biology and medicine"},
                  { 'role': 'user', 'content': 
                  "Summary key findings into one paragraph using following information: " + str([ab,]) 
                  }]
                for ab in abstracts]
        
        responses: List[str] = asyncio.run(chat_llm(self.llm_index_model_name, messages, temperature, semaphore_limit=self.llm_semaphore_limit, backend=self.backend))
        return responses
            
    def llm_output_to_df(self, llm_out: List[str], has_main_text: List[str]) -> pd.DataFrame:
        """
        Convert LLM output strings to a structured DataFrame.
        
        Args:
            llm_out (List[str]): List of LLM responses containing entity relationships.
            has_main_text (List[str]): List indicating if each paper has main text ("Yes"/"No").
        
        Returns:
            pd.DataFrame: Structured DataFrame with entity relationships.
        """
        new_llm_out: List[List[Dict[str, Any]]] = [locate_json_string_body_from_string(xx) for xx in llm_out]
        df: List[pd.DataFrame] = [pd.DataFrame(xx) for xx in new_llm_out]
        df: List[pd.DataFrame] = [standarized_llm_df(xx) for xx in df]
        for i in range(len(df)):
            df[i]["wh_paper"] = i
            df[i]['tag'] = list(range(df[i].shape[0]))
            df[i]['has_main_text'] = has_main_text[i]
        return pd.concat(df, axis=0)
    
     
    def subset_maintext_async(self, df_abstract_output: pd.DataFrame, main_texts: List[List[str]], has_main_text: List[str]) -> pd.DataFrame:
        """
        Extract relevant main text sections based on entity relationships.
        
        Args:
            df_abstract_output (pd.DataFrame): DataFrame with entity relationships from abstracts.
            main_texts (List[List[str]]): List of main text sections for each paper.
            has_main_text (List[str]): List indicating if each paper has main text ("Yes"/"No").
        
        Returns:
            pd.DataFrame: Updated DataFrame with relevant main text sections.
        """
        df_main_texts=pd.concat([ pd.DataFrame({'wh_paper': i, 'main_txt': main_texts[i]}) for i in range(len(main_texts)) if has_main_text[i]=="Yes"  ])
        new_main_texts=list(df_main_texts['main_txt'])
        
        main_text_embed =  asyncio.run(embed_llm(self.embed_model_name, new_main_texts, semaphore_limit=self.embed_semaphore_limit, backend=self.backend))
        
        df_abstract_output_sub1=df_abstract_output[df_abstract_output['has_main_text']=="Yes"].copy()
        df_abstract_output_sub2=df_abstract_output[df_abstract_output['has_main_text']=="No"].copy()
        
        queries_item=["".join(["what is the relationship or interaction between ", str(x) , " and ", str(y) ,"?"])
            for x, y in zip(list(df_abstract_output_sub1['source_entity']), list(df_abstract_output_sub1['target_entity']))]
        
        queries_embed=  asyncio.run(embed_llm(self.embed_model_name, queries_item, semaphore_limit=self.embed_semaphore_limit, backend=self.backend))
                      
        res=[]
        for cc in tqdm(range(len(queries_embed)), desc="Processing main text sections", unit="query"):                   
            which_paper=df_abstract_output_sub1.iloc[cc]["wh_paper"].item()
            wh_main_text= [i for i, val in enumerate(df_main_texts['wh_paper']) if val == which_paper]

            select_main_text_embed=[main_text_embed[i] for i in wh_main_text]
            select_main_text      =[new_main_texts[i]  for i in wh_main_text]
            sims =  [(1 - cosine(queries_embed[cc], fb )).item() for fb in select_main_text_embed ]
            sim_item = [ x for x in sims if x > self.embedding_similarity ]
            if len(sim_item)==0:
                new_text_list=[select_main_text[ sims.index(max(sims)) ] ]
            else:
                new_text_list=[select_main_text[sims.index(x)] for x in sim_item if x > self.embedding_similarity]
            new_text= str(new_text_list)
            res.append(new_text)
        
        df_abstract_output_sub1['main_text']=res
        df_abstract_output_sub2['main_text']=[str([])]* df_abstract_output_sub2.shape[0]
        return pd.concat([df_abstract_output_sub1, df_abstract_output_sub2], axis=0)
    
    
    def refine_with_maintext_async(self, df_ab: pd.DataFrame, temperature: Union[List[float], float] = 0.7) -> pd.DataFrame:
        """
        Refine entity relationships using relevant main text sections.
        
        Args:
            df_ab (pd.DataFrame): DataFrame with entity relationships and main text sections.
            temperature (Union[List[float], float]): Temperature for LLM generation.
        
        Returns:
            pd.DataFrame: Updated DataFrame with refined entity relationships.
        """
        df_ab_sub1=df_ab[df_ab['has_main_text']=="Yes"].copy()
        df_ab_sub2=df_ab[df_ab['has_main_text']=="No"].copy()
        user_query = [ "".join(["Extract the relationship or interaction between ", str(x) ," and ", str(y), \
                            " into one brief and short paragraph according to following information: ", str([z])]) 
                for x, y, z in zip(list(df_ab_sub1['source_entity']), list(df_ab_sub1['target_entity']), list(df_ab_sub1['main_text']))]
        
        messages=[ [ { 'role': 'system', 'content': system_prompt_main_text},
                           { 'role': 'user', 'content': qry} ]
                  for qry in user_query ]
                  
        current_temperature =[temperature]*len(messages)       
        responses= asyncio.run(chat_llm(self.llm_index_model_name, messages, current_temperature, semaphore_limit=self.llm_semaphore_limit, backend=self.backend))
        
        df_ab_sub1['updated_relationship_description']=responses
        df_ab_sub2['updated_relationship_description']=list(df_ab_sub2['relationship_description'])
        df_ab = pd.concat([df_ab_sub1, df_ab_sub2], axis=0)
        df_ab=df_ab.drop(columns=['main_text'], axis=1) 
        return df_ab 
    
    def save_to_csv(self, df: pd.DataFrame, file_name: str) -> None:
        file_path = os.path.join(self.working_dir, file_name)
        try:
            # Ensure working directory exists
            os.makedirs(self.working_dir, exist_ok=True)
            # Write dataframe to CSV
            df.to_csv(file_path, index=False)
            logging.info(f"Successfully saved data to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save data to {file_path}: {str(e)}")
            raise ValueError(f"Failed to save data to {file_path}: {str(e)}") from e
        
    def insert_paper(self, documents: pd.DataFrame) -> Optional[None]:
        """
        Insert papers into the vector database, building the knowledge graph.
        
        Args:
            documents (pd.DataFrame): DataFrame containing paper information with required columns:
                - abstract: Paper abstracts
                - main_text: Main text sections (list or string)
                - paper_id: Unique paper identifiers
        
        Raises:
            ValueError: If required columns are missing or contain null values.
        
        Returns:
            Optional[None]: None if successful, or raises an exception on failure.
        """
        # Validate input DataFrame structure
        required_columns = ['abstract', 'main_text', 'paper_id']
        for col in required_columns:
            if col not in documents.columns:
                raise ValueError(f"Missing required column: {col}")
            if documents[col].isnull().any():
                raise ValueError(f"Column {col} contains null values")
        
        abstracts=[safe_unicode_decode(doc) for doc in list(documents['abstract'])]
        main_texts=list(documents['main_text'] )
        
        # Process main_texts with proper validation
        main_texts = self._process_main_texts(main_texts)
        has_main_text=["No" if not main_texts[i] or len(" ".join(main_texts[i])) < 30 else "Yes" for i in range(len(main_texts)) ]
        
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
        output_abstract = self.process_abstract_async(abstracts, has_main_text, temperature=[0.3, 0.7])

        
        
        logging.info("transform LLM output into DataFrome ")
        df_abstract = self.llm_output_to_df(output_abstract, has_main_text)
        
        self.save_to_csv(df_abstract, "df_abstract_v0.csv")
        
        logging.info("Extract related main text ")
        df_abstract = self.subset_maintext_async(df_abstract, main_texts, has_main_text)
        
        self.save_to_csv(df_abstract, "df_abstract_v1.csv")
        
        logging.info("Refine with related main text ")
        df_abstract = self.refine_with_maintext_async(df_abstract, temperature=0.7)

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
        
        # Generate embeddings with progress indicator
        relationship_texts = list(input_df['updated_relationship_description'])
        text_embedding = []
        for txt in tqdm(relationship_texts, desc="Generating embeddings", unit="text"):
            if isinstance(txt, str) and len(txt) > 10:
                embedding = embed(model=self.embed_model_name, input=txt).embeddings[0]
            else:
                embedding = [0.0] * self.embed_size
            text_embedding.append(embedding)
                            

        
        logging.info("Store data into vector database ")
        input_df['relationship_type']=[str(x) for x in list(input_df['relationship_type'])]
        input_df['source_entity']=[str(x) for x in list(input_df['source_entity'])]
        input_df['target_entity']=[str(x) for x in list(input_df['target_entity'])]
        input_df['source_entity_type']=[str(x) for x in list(input_df['source_entity_type'])]
        input_df['target_entity_type']=[str(x) for x in list(input_df['target_entity_type'])]
        input_dict=input_df.to_dict(orient='records')

        
        new_data=[PointStruct(id=document_id[i], vector=text_embedding[i], payload= input_dict[i]) 
                        for i in range(len(document_id))]
        
        
        res=self.client.upsert(collection_name=self.collection_name, wait=True, points=new_data)
        
        logging.info("Insert document into database, done ")
        
    def _validate_query_parameters(self, question: str, top_results: int, gene: List[str], disease: List[str], 
                                   paper_id: List[Union[int, str]], question_analysis: bool, filter_importance: float, 
                                   temperature: float, similarity_score: float) -> None:
        """
        Validate query parameters to ensure they meet expected types and ranges.
        
        Args:
            question: Biomedical question to answer.
            top_results: Maximum number of results to return.
            gene: List of genes to filter by.
            disease: List of diseases to filter by.
            paper_id: List of paper IDs to filter by.
            question_analysis: Whether to analyze the question for entities.
            filter_importance: Importance cutoff for entity relationships.
            temperature: Temperature for LLM generation.
            similarity_score: Minimum similarity score for vector queries.
        
        Raises:
            ValueError: If any parameter fails validation.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
        
        if not isinstance(top_results, int) or top_results < 1:
            raise ValueError("top_results must be a positive integer")
        
        if not isinstance(gene, list):
            raise ValueError("gene must be a list")
        
        if not isinstance(disease, list):
            raise ValueError("disease must be a list")
        
        if not isinstance(paper_id, list):
            raise ValueError("paper_id must be a list")
        
        if not isinstance(question_analysis, bool):
            raise ValueError("question_analysis must be a boolean")
        
        if not isinstance(filter_importance, (int, float)) or filter_importance < -1 or filter_importance > 1:
            raise ValueError("filter_importance must be between -1 and 1")
        
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if not isinstance(similarity_score, (int, float)) or similarity_score < 0 or similarity_score > 1:
            raise ValueError("similarity_score must be between 0 and 1")
    
    def _sanitize_query_inputs(self, gene: List[str], disease: List[str], paper_id: List[Union[int, str]]) -> Tuple[List[str], List[str], List[Union[int, str]]]:
        """
        Sanitize query inputs to ensure they are in the correct format.
        
        Args:
            gene: List of genes to sanitize.
            disease: List of diseases to sanitize.
            paper_id: List of paper IDs to sanitize.
        
        Returns:
            Tuple of sanitized gene, disease, and paper_id lists.
        """
        # Sanitize gene inputs
        sanitized_gene = [safe_unicode_decode(g).strip() for g in gene if isinstance(g, str) and g.strip()]
        
        # Sanitize disease inputs
        sanitized_disease = [safe_unicode_decode(d).strip() for d in disease if isinstance(d, str) and d.strip()]
        
        # Sanitize paper_id inputs
        sanitized_paper_id = [pid for pid in paper_id if isinstance(pid, (int, str)) and str(pid).strip()]
        sanitized_paper_id = [int(pid) if isinstance(pid, str) and pid.isdigit() else pid for pid in sanitized_paper_id]
        
        return sanitized_gene, sanitized_disease, sanitized_paper_id

    def _build_query_filter(self, gene: List[str], disease: List[str], output_types: List[str]) -> models.Filter:
        """
        Build a query filter based on the provided gene, disease, and output types.
        
        Args:
            gene: List of genes to filter by.
            disease: List of diseases to filter by.
            output_types: List of entity types to filter by.
        
        Returns:
            A Qdrant Filter object configured with the appropriate conditions.
        """
        filter_conditions = []
        
        # Add entity type filter (always included)
        entity_type_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key='source_entity_type',
                    match=models.MatchAny(any=output_types)
                ),
                models.FieldCondition(
                    key='target_entity_type',
                    match=models.MatchAny(any=output_types)
                )
            ]
        )
        filter_conditions.append(entity_type_filter)
        
        # Add gene filter if provided
        if gene and disease:
            # Both gene and disease are provided - add separate conditions
            gene_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key='source_entity',
                        match=models.MatchAny(any=gene)
                    ),
                    models.FieldCondition(
                        key='target_entity',
                        match=models.MatchAny(any=gene)
                    )
                ]
            )
            disease_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key='source_entity',
                        match=models.MatchAny(any=disease)
                    ),
                    models.FieldCondition(
                        key='target_entity',
                        match=models.MatchAny(any=disease)
                    )
                ]
            )
            filter_conditions.extend([gene_filter, disease_filter])
        elif gene or disease:
            # Either gene or disease is provided - combine into single filter
            filter_vector = gene + disease
            combined_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key='source_entity',
                        match=models.MatchAny(any=filter_vector)
                    ),
                    models.FieldCondition(
                        key='target_entity',
                        match=models.MatchAny(any=filter_vector)
                    )
                ]
            )
            filter_conditions.append(combined_filter)
        
        # Build final filter - use must if multiple conditions, else just the single condition
        if len(filter_conditions) > 1:
            return models.Filter(must=filter_conditions)
        else:
            return filter_conditions[0]

    def _calculate_graph_weights(self, data_df: pd.DataFrame, df_metadatas: pd.DataFrame, filter_importance: float) -> pd.DataFrame:
        """
        Calculate graph-based weights for entity relationships and filter results.
        
        Args:
            data_df: Full dataframe with all entity relationships.
            df_metadatas: Filtered dataframe from vector query.
            filter_importance: Importance cutoff (0-1). -1 means no filtering.
            
        Returns:
            Filtered dataframe with calculated edge weights.
        """
        if filter_importance == -1:
            return df_metadatas
            
        # Create graph and calculate degrees if importance filtering is needed
        if filter_importance > 0:
            G = nx.from_pandas_edgelist(data_df, source='source_entity', target='target_entity')
            degrees = G.degree(weight="weight")
        
        # Calculate edge weights based on node degrees
        start_node = list(df_metadatas['source_entity'])
        end_node = list(df_metadatas['target_entity'])
        
        # Fix bug: use both start_node and end_node degrees
        edge_weight = []
        min_edge_weight = []
        for i in range(len(start_node)):
            start_degree = degrees.get(start_node[i], 0)
            end_degree = degrees.get(end_node[i], 0)
            edge_weight.append(start_degree + end_degree)
            min_edge_weight.append(min(start_degree, end_degree))
        
        df_metadatas['edge_weight'] = edge_weight
        df_metadatas['min_edge_weight'] = min_edge_weight
        
        # Apply filtering if importance threshold is set
        if filter_importance > 0:
            importance_cutoff = np.quantile(min_edge_weight, filter_importance)
            df_metadatas = df_metadatas[df_metadatas['min_edge_weight'] > importance_cutoff]
            
        return df_metadatas

    def _extract_entities_from_question(self, question: str, temperature: float) -> Tuple[List[str], List[str]]:
        """
        Extract gene and disease entities from a biomedical question using LLM.
        
        Args:
            question: Biomedical question to analyze.
            temperature: Temperature for LLM generation.
            
        Returns:
            Tuple containing extracted gene list and disease list.
        """
        question_query="Extract the offical HGNC symbols from following sentence: " + str([question])
        message_gene=[ { 'role': 'system', 'content': system_prompt_gene}, { 'role': 'user', 'content': question_query} ]

        question_query="Extract the MESH disease term from following sentence: " + str([question])
        message_disease=[ { 'role': 'system', 'content': system_prompt_disease}, { 'role': 'user', 'content': question_query} ]

        responses= asyncio.run(chat_llm(self.llm_query_model_name, [message_gene, message_disease], [temperature,temperature], semaphore_limit=self.llm_semaphore_limit, backend=self.backend))

        from ast import literal_eval
        gene= literal_eval(remove_thinking(responses[0].strip()))
        disease= literal_eval(remove_thinking(responses[1].strip()))
        print("Query of Gene: "+ str(gene) + " and disease: "+ str(disease))
        
        return gene, disease

    def _process_main_texts(self, main_texts: List[Union[str, List[str], Any]]) -> List[List[str]]:
        """
        Process main_text entries into a standardized format.
        
        Args:
            main_texts: List of main_text entries that can be strings, lists, or other formats.
            
        Returns:
            List of processed main_texts, where each entry is a list of non-empty string paragraphs.
        """
        processed_main_texts = []
        for txt in main_texts:
            try:
                if isinstance(txt, str):
                    # Try literal_eval first for structured text
                    from ast import literal_eval
                    evaluated = literal_eval(txt)
                    if isinstance(evaluated, list):
                        processed = [safe_unicode_decode(x) for x in evaluated]
                    else:
                        processed = [safe_unicode_decode(txt)]
                elif isinstance(txt, list):
                    processed = [safe_unicode_decode(x) for x in txt]
                else:
                    processed = [safe_unicode_decode(str(txt))]
                
                # Filter out empty strings
                processed = [x.strip() for x in processed if x.strip()]
                processed_main_texts.append(processed)
            except (SyntaxError, ValueError, TypeError) as e:
                # Fallback to basic string splitting if literal_eval fails
                try:
                    processed = [safe_unicode_decode(x).strip() for x in str(txt).split("\n") if safe_unicode_decode(x).strip()]
                    processed_main_texts.append(processed)
                except Exception:
                    processed_main_texts.append([])
        return processed_main_texts

    def query(self, question: str, top_results:int=300, gene:List[str]=[], disease:List[str]=[], paper_id:List[Union[int, str]]=[], question_analysis:bool=True, filter_importance:float= -1, temperature:float=0.75, similarity_score:float=0.75) -> Union[str, Dict[str, Any]]:
        """
        Query the knowledge graph to answer biomedical questions.
        
        Args:
            question (str): Biomedical question to answer.
            top_results (int): Maximum number of results to return.
            gene (List[str]): List of genes to filter by (HGNC symbols).
            disease (List[str]): List of diseases to filter by (MeSH terms).
            paper_id (List[Union[int, str]]): List of paper IDs to filter by.
            question_analysis (bool): Whether to automatically extract gene/disease entities from the question.
            filter_importance (float): Importance cutoff for entity relationships (0-1). -1 means no filtering.
            temperature (float): Temperature for LLM generation.
            similarity_score (float): Minimum similarity score for vector database queries.
        
        Returns:
            Union[str, Dict[str, Any]]: Either a string message if no results, or a dictionary with:
                - outcome: Final answer to the question
                - reference: DataFrame with reference paper metadata
                - chunked_outcome: DataFrame with intermediate LLM results
        """
        # Validate input parameters
        self._validate_query_parameters(question, top_results, gene, disease, paper_id, question_analysis, filter_importance, temperature, similarity_score)
        
        # Sanitize inputs
        gene, disease, paper_id = self._sanitize_query_inputs(gene, disease, paper_id)
        
        if (not gene) and (not disease) and (not paper_id) and question_analysis:
            gene, disease = self._extract_entities_from_question(question, temperature)
        
        data_df=pd.read_csv(self.working_dir+ "/" + "input_df.csv")

        tmp_list=list(data_df['source_entity_type']) + list(data_df['target_entity_type'])     
        node_all_types= list(pd.DataFrame({'A': tmp_list}).groupby('A').size().sort_values(ascending=False).keys())[0:40]
        process_query= "The question is: " + safe_unicode_decode(question) +"\n" +  "The label list is as following: " + str(node_all_types) 
        process_messages=[ { 'role': 'system', 'content': system_prompt_process_query}, { 'role': 'user', 'content': process_query} ]
        process_response = chat(model=self.llm_query_model_name, messages=process_messages, options={"temperature": 0.1})['message']['content']
        process_response=remove_thinking(process_response)
        #print(process_response)
        
        try:
            from ast import literal_eval
            output_types=literal_eval(process_response)
        except Exception as e:
            print("Sorry that something wrong with LLM output. Please re-query again")
            return None
        
        question_embed= embed(model=self.embed_model_name, input=safe_unicode_decode(question)).embeddings[0]
        
        # Build filter using the reusable method
        filter = self._build_query_filter(gene, disease, output_types)
        
        # Execute query with the built filter
        output_query = self.client.query_points(
            collection_name=self.collection_name,
            query=question_embed,
            with_payload=True,
            limit=top_results,
            with_vectors=False,
            score_threshold=similarity_score,
            query_filter=filter
        ).points    
        
        #print(output_query)
        metadatas=[x.payload for x in output_query]
        similar_score=[x.score for x in output_query]
        if len(metadatas)==0:
            return "No related paper or record finding in current database!"
        
        df_metadatas = pd.DataFrame(metadatas)
        
        # Calculate graph weights and filter results
        df_metadatas = self._calculate_graph_weights(data_df, df_metadatas, filter_importance)

        #df_metadatas.sort_values(by=['edge_weight','min_edge_weight'], ascending=False)

        match_metadatas=df_metadatas[['paper_id','journal','year',"weight"]].drop_duplicates(subset=['paper_id'])
        match_text= list(df_metadatas['updated_relationship_description'])
        paper_ids= list(df_metadatas['paper_id'])
        paper_weight=list(df_metadatas['weight'])
        input_text=[{'id': paper_ids[i], 'weight':paper_weight, 'text': match_text[i]} for i in range(len(match_text))]

        
        input_step=20
        messages=list()
        chunk_paper_ids=list()
        for i in tqdm(range(0,len(input_text),input_step), desc="Preparing query chunks", unit="chunk"):
            user_query= "According to input json: " + str(match_text[i:min(i + input_step, len(match_text))]) +"\n\n  Please answer following question: " + question 
            messages.append([ { 'role': 'system', 'content': system_prompt_query}, { 'role': 'user', 'content': user_query} ])
            chunk_paper_ids.append(paper_ids[i:min(i + input_step, len(match_text))])
            
        current_temperature=[temperature] *len(messages)
        output_response= asyncio.run(chat_llm(self.llm_query_model_name, messages, current_temperature, semaphore_limit=self.llm_semaphore_limit, backend=self.backend))


        merge_query= "For the question of " + str([question]) +"\n"+"LLM gives mutliple outputs using different resources. Merge and summarize them as an updated response using following LLM outputs: " +  str(output_response)
        response = chat(model=self.llm_query_model_name, messages=[{'role':'user', 'content': merge_query}], options={"temperature": temperature})
        llm_output= remove_thinking(response['message']['content'])
        output={"outcome": llm_output,'reference':match_metadatas, "chunked_outcome":pd.DataFrame({'outcome': output_response, 'paper_id': chunk_paper_ids})}

        #user_query= "According to input json: " + str(input_text) +"\n\n  Please answer following question: " + question 
        #messages=[ { 'role': 'system', 'content': system_prompt_query}, { 'role': 'user', 'content': user_query} ]
        #response = chat(model=self.llm_query_model_name, messages=messages, options={"temperature": temperature})
        #llm_output= remove_thinking(response['message']['content']) + "\n\n\n" + str(match_metadatas)
        
        return output


if __name__ == '__main__':
    import fastbmrag.fastbmrag as fastbmrag
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
    parser.add_argument('--embedding_similarity', type=float,  help="The embedding similarity cutoff", default=0.75)
    parser.add_argument('--top_match', type=int, help="The record number returned for query", default=300)
    parser.add_argument('--temperature', type=float, help="The temperature of LLM model", default=0.7)
    parser.add_argument('--question_analysis', type=bool, help="Whether to analyze the question to extract entities", default=True)

    args=parser.parse_args()
    
    rag=fastbmrag.RAG(working_dir=args.working_dir, 
                   collection_name=args.collection_name, 
                   llm_index_model_name=args.llm_index_model_name, 
                   llm_query_model_name=args.llm_query_model_name, 
                   embed_model_name=args.embed_model_name, 
                   embedding_similarity=args.embedding_similarity)
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
        out=rag.query(args.question, top_results=args.top_match, similarity_score=args.embedding_similarity, temperature=args.temperature, question_analysis=args.question_analysis)
        print(out)
    
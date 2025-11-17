# fastbmRAG
FastbmRAG, an fast lightweight graph-based RAG optimized for biomedical literature. 

Utilizing well organized structure of biomedical papers, fastbmRAG divides the construction of knowledge graph into two stages:
* Build a draft knowledge graph using only abstracts of scientific papers
* Refine the nodes and edges of knowledge graph using main texts 

Our evaluations demonstrate that fastbmRAG is over 10x faster than existing graph-RAG tools and achieve superior coverage and accuracy to existing knowledge. 

Overall, fastbmRAG provides a fast solution for quickly understanding, summarizing, and answering questions about biomedical literature on a large scale.

## Install and build conda envirenment

fastbmRAG is implemented in Python 3.11 or higher. 

Current version only support ollama LLM models. Please install ollama by visiting:

https://ollama.com/download

The ollama models should be installed before usage with a command like:
```
ollama run gemma3
```

Build a conda envirenment for neccessary dependences:

```
git clone https://github.com/menggf/fastbmRAG.git
cd fastbmRAG
conda env create -f environment.yml
conda activate fastbmrag
```


## Usage

There are two modes: 'update' and 'query', which are used to build local collection and query local collection, respectively.

### Build local collection

The 'update' mode is used to create an new collection or add new documents to an existing collection of vector database. The documents should be a text file in CSV format. 

It should have at least three columns: 'abstract', 'main_text' and 'paper_id'. If there are more columns, they are used for additional information. Each element of the 'main_text' column should be either a list of strings in the format '[str1, str2]' or a string separated by '\n'. 'paper_id' is a unique ID for each paper. If a paper ID exists in the collection, the corresponding paper will be ignored. 

To update the collection, use the following command:

```
python main.py --job update --document examples/demo1.csv
			   --collection_name collection_name 
			   --working_dir directory_path
               --llm_update_model_name llm_model_name
               --embed_model_name embedding_model_name
```

Here, ‘collection_name’ and ‘working_dir’ specify the collection name and directory to store collection. llm_model_name and embedding_model_name are the ollama model names. 

### Query the collection 
It is used to query the collection with questions.

```
python main.py --job query --collection_name collection_name 
	           --working_dir directory_path 
	           --question 'your question'
               --llm_query_model_name llm_model_name
               --embed_model_name embedding_model_name

```

### Related publication

Guofeng Meng et.al., fastbmRAG: A Fast Graph-Based RAG Framework for Efficient Processing of Large-Scale Biomedical Literature. ArXiv, 2025, DOI:10.48550/arXiv.2511.10014



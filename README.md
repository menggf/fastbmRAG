# fastbmRAG
FastbmRAG, an fast lightweight graph-based RAG  optimized for biomedical literature. 

Utilizing well organized structure of biomedical papers, fastbmRAG divides the construction of knowledge graph into two stages, first drafting graphs using abstracts; and second, refining them using main texts guided by vector-based entity linking, which minimizes redundancy and computational load. 

Our evaluations demonstrate that fastbmRAG is over 10x faster than existing graph-RAG tools and achieve superior coverage and accuracy to input knowledge. FastbmRAG provides a fast solution for quickly understanding, summarizing, and answering questions about biomedical literature on a large scale.

## Build a conda envirenment

fastbmRAG is implemented in Python 3.11 or higher. 

Build a conda envirenment for neccessary dependences:

```
git clone https://github.com/menggf/fastbmRAG.git
cd fastbmRAG
conda env create -f environment.yml
conda activate fastbmrag
```


## Usage

There are two modes: 'update' and 'query'. 

### Build local collection of vector database

The 'update' mode is used to create an new collection or add new documents to an existing collection of vector database. The documents should be a text file in CSV format. 

It should have at least three columns: 'abstract', 'main_text' and 'paper_id'. If there are more columns, they are used for additional information. Each element of the 'main_text' column should be either a list of strings in the format '[str1, str2]' or a string separated by '$\setminus$n'. 'paper_id' is a unique ID for each paper. If a paper ID exists in the collection, the corresponding paper will be ignored. To update the collection, use the following command:

```
python main.py --job update --document examples/demo1.csv
			   --collection_name collection\_name 
			   --working_dir directory\_path
```

Here, ‘collection\_name’ and ‘working\_dir’ specify the collection name and directory to store collection.

### Query the collection 
It is used to query the collection with questions.

```
python main.py --job query --collection\_name collection\_name 
	           --working\_dir directory\_path 
	           --question 'your question'
```


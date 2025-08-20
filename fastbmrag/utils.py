import json
import os
import re
import asyncio
import textract
from ollama import AsyncClient

def locate_json_string_body_from_string(content: str) -> str | None:
    """
    Locate the JSON string body from a string
    This code is borrowed from LightRag
    https://github.com/HKUDS/LightRAG/tree/main
    """
    try:
        maybe_json_str = re.search("json\\n*\\[.*\\]\\n", content, re.DOTALL)
        #print(maybe_json_str)
        if maybe_json_str is not None:
            maybe_json_str = maybe_json_str.group(0)
            maybe_json_str = maybe_json_str.replace("\\n", "")
            maybe_json_str = maybe_json_str.replace("\n", "")
            #maybe_json_str = maybe_json_str.replace("'", '"')
            maybe_json_str = maybe_json_str.replace("json", "")
            # json.loads(maybe_json_str) # don't check here, cannot validate schema after all
            return json.loads(maybe_json_str)
    except Exception:
        pass
        return None

def safe_unicode_decode(content):
    '''
    This code is borrowed from LightRag
    https://github.com/HKUDS/LightRAG/tree/main
    '''
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")
    
    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))
    
    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content
    )
    new_decoded_content=re.sub(r"\n","",decoded_content)
    return new_decoded_content
    
def convert_response_to_json(response: str) -> dict[str, str]:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except Exception:
        print("Failed to parse JSON: " + json_str)
        return None

    

def chunking_by_word_size(
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_word_size: int = 128,
    max_word_size: int = 1024,
) -> list[str]:
    '''
    split the text into chunks if long text is used
    This code is borrowed from LightRag
    https://github.com/HKUDS/LightRAG/tree/main
    '''
    results: list[str] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                new_chunks.append( chunk)
        else:
            for chunk in raw_chunks:
                if len(chunk) > max_word_size:
                    for start in range(
                        0, len(chunk), max_word_size - overlap_word_size
                    ):
                        chunk_content = chunk[start : start + overlap_word_size]
                        new_chunks.append( chunk_content)
                else:
                    new_chunks.append( chunk)
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(chunk.strip())
    else:
        for index, start in enumerate(
            range(0, len(content), max_word_size - overlap_word_size)
        ):
            chunk_content = content[start : start + max_word_size]
            results.append(chunk_content)
    return results

def standarized_llm_df(df):
    '''
    if df includes non-standard gene name or disease name, a transformation is need
    Need to be updated
    '''
    return df
    
#async def ollama_chat_task_old(model, content, temprature=0.7):
#    return await AsyncClient().chat(model=model, messages=content, options={"temperature": temprature})
    
#async def chat_ollama_old(model, content, temprature=0.7):
#    tasks = [ollama_chat_task(model, cc, temprature) for cc in content]
#    responses = await asyncio.gather(*tasks)
#    return [response['message']['content'] for response in responses]

async def nanovllm_chat_task(semaphore, model, content, temprature=0.7):
    async with semaphore:
        return await AsyncClient().chat(model=model, messages=content, options={"temperature": temprature})
        
async def ollama_chat_task(semaphore, model, content, temprature=0.7):
    async with semaphore:
        return await AsyncClient().chat(model=model, messages=content, options={"temperature": temprature})
    
async def chat_llm(model, content, temprature):
    semaphore=asyncio.Semaphore(2)
    tasks = [ollama_chat_task(semaphore, model, content[i], temprature[i]) for i in range(len(content))]
    responses = await asyncio.gather(*tasks)
    return [remove_thinking(response['message']['content']) for response in responses]


    
async def ollama_embed_task(semaphore, model, content):
    async with semaphore:
        return await AsyncClient().embed(model=model, input=content)
        
async def embed_llm(model, content):
    semaphore=asyncio.Semaphore(2)
    tasks = [ollama_embed_task(semaphore, model, cc) for cc in content]
    responses = await asyncio.gather(*tasks)
    return [response.embeddings[0]  for response in responses]

#async def ollama_embed_task_old(model, content):
#    return await AsyncClient().embed(model=model, input=content)
        
#async def embed_ollama_old(model, content):
#    tasks = [ollama_embed_task(model, cc) for cc in content]
#    responses = await asyncio.gather(*tasks)
#    return [response.embeddings[0]  for response in responses]


def read_doc_by_word_number(input_file, max_word_size, overlap_word_size):
    input_text = textract.process(input_file).decode("utf8")
    input_text=re.sub(r'\n', ' ', input_text, count=0, flags=0)
    out=chunking_by_word_size(input_text, overlap_word_size=overlap_word_size, max_word_size=max_word_size)
    return {'abstract': out[0], 'main_text': out[1:], 'meta_info':{'paper_id': input_file, 'weight': 0}}


def remove_thinking(input_string):
    input_string=re.sub(r'<thinking>.*?</thinking>', '', input_string, flags=re.DOTALL)
    input_string=re.sub(r'<think>.*?</think>', '', input_string, flags=re.DOTALL)
    input_string=re.sub(r'\s*</start_of_turn>\s*', '', input_string, flags=re.DOTALL)
    input_string=re.sub(r'\s*</end_of_turn>\s*', '', input_string, flags=re.DOTALL).strip()
    return input_string


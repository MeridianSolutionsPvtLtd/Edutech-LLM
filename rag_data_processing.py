import zipfile
import os
import re
import sys
import shutil
import itertools
import requests
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from openai import AzureOpenAI
from nltk.tokenize import sent_tokenize
import json
import ast
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, UploadFile, File, HTTPException


client = AzureOpenAI(
  api_key = "f619d2d04b4f44d28708e4c391039d01",
  api_version = "2024-02-01",
  azure_endpoint = "https://openainstance001.openai.azure.com/"
)


##############################    ********* CREATE - QUIZ *********   ##############################


# Split Content into chunks
def chunks_string(text, tokens):
    # Initialize variables
    segments = []
    len_sum = 0
    k = 0

    # Split the text into sentences
    raw_list = sent_tokenize(text)

    # Iterate the Sentences one-by-one
    for i in range(len(raw_list)):

      # Split that sentence into tokens
      x1 = len(raw_list[i].split())

      # Cummulative length of tokens till this sentence
      len_sum = len_sum + x1

      k = k + 1

      # If no. of tokens > threshold
      if len_sum > tokens:

        ### Logic for finding how many sentences need to be repeat in current segment ###

        # Will be used for first segment only
        if i-(k+1) < 0:
            j = 0

        # Will be used for next  all segments
        else:
          j = i-(k+1)
          if len(" ".join(raw_list[j: i+1]).split()) > tokens:
            j = i-k

        # Append list of sentences to each segment
        segments.append(" ".join(raw_list[j: i]))

        # Set variables = 0
        len_sum = 0
        k = 0

      # If it is last iteration
      if i == len(raw_list)-1:
        if i-(k+1) < 0:
          j = 0

        else:
          j = i-(k+1)
          if len(" ".join(raw_list[j: i+1]).split()) > tokens:
            j = i-k

          # Append list of sentences to each segment
          segments.append(" ".join(raw_list[j: i+1]))

    return segments


def read_and_split_pdf(file_path, start_page=1, chunk_size=200, word_limit=14000):
    # Read PDF and split content into chunks starting from a specific page
    reader = PdfReader(file_path)
    content_chunks = []
    total_words = 0

    for page_num, page in enumerate(reader.pages[start_page - 1:], start=start_page):
        page_content = page.extract_text() or ''
        words = page_content.split()
        
        # Check if adding this page's content will exceed the word limit
        if total_words + len(words) > word_limit:
            # Truncate the content of this page if necessary
            remaining_words = word_limit - total_words
            truncated_content = " ".join(words[:remaining_words])
            content_chunks.append((page_num, file_path, truncated_content.strip()))
            print(f"Reached word limit. Truncated at Page {page_num} from {file_path}.")
            break
        else:
            # Add entire page content
            content_chunks.extend([(page_num, file_path, chunk.strip()) for chunk in chunks_string(page_content, chunk_size) if len(chunk.split()) > 2])
            total_words += len(words)
            print(f"Reading Page {page_num} from {file_path}...")  # Debugging page number

    return content_chunks


def truncate_with_buffer(content, word_limit, buffer=200):
    # Truncate content to a specified word limit, with a buffer to avoid cutting off sentences
    words = content.split()
    if len(words) <= word_limit:
        return content
    else:
        # Take initial word limit plus buffer
        truncated_content = " ".join(words[:word_limit + buffer])
        # Find the end of the last complete sentence
        end_position = max(truncated_content.rfind('.'), truncated_content.rfind('!'), truncated_content.rfind('?'))
        if end_position != -1:
            truncated_content = truncated_content[:end_position + 1]
        else:
            truncated_content = " ".join(words[:word_limit])
        return truncated_content
    

def process_small_pdf(pdf_path):
    # Process the entire content of a small PDF
    pdf_content = ""
    for chunk_tuple in read_and_split_pdf(pdf_path, word_limit=14000):
        pdf_content += chunk_tuple[2] + " "
    return pdf_content.strip()


def process_large_pdf(pdf_path, start_page):
    # Process content of a large PDF starting from a specific page
    chunks = read_and_split_pdf(pdf_path, start_page=start_page, word_limit=14000)
    return " ".join(chunk_tuple[2] for chunk_tuple in chunks).strip()


def combine_and_truncate_content(content_list, word_limit=14000, buffer=200):
    # Combine content from all PDFs and apply truncation with a buffer
    combined_content = " ".join(content_list)
    print("Combined Content Length (before truncation):", len(combined_content.split()))
    final_content = truncate_with_buffer(combined_content, word_limit, buffer)
    print("Final Content Length (after truncation):", len(final_content.split()))
    return final_content


def download_pdf(url: str, folder_path: str, filename: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        pdf_file_path = os.path.join(folder_path, filename)
        with open(pdf_file_path, "wb") as f:
            f.write(response.content)
        return pdf_file_path
    else:
        raise HTTPException(status_code=400, detail=f"Failed to download the PDF from the URL: {url}")


##############################    ********* CREATE - EMBEDDINGS *********   ##############################


# print (generate_embeddings(text_chunks[1]))
def generate_embeddings(texts, model="text-embedding-3-small"):
    return client.embeddings.create(input=[texts], model=model).data[0].embedding


# Function to read PDF file content and split into chunks
def read_and_split_pdf_for_embedding(file_path, file_name, chunk_size=200):
    reader = PdfReader(file_path)
    content_chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_content = page.extract_text() or ''
        # Split content into chunks based on word count
        chunks = chunks_string(page_content, chunk_size)
        content_chunks.extend([(page_num,file_name, chunk.strip()) for chunk in chunks if len(chunk.split()) > 2])
    return content_chunks


####################################    ********* CHATBOT *********   ####################################


def extract_array_of_embedding_from_file(file_name):
    print("extract_array_of_embedding_from_file")
    df = pd.read_csv(file_name)
    embedding_list_final = []
    embedding_list = df.embedding.apply(ast.literal_eval)
    for temp_element in embedding_list:
        embedding_list_final.append(temp_element)
    embedding_array = np.array(embedding_list_final)
    return embedding_array, df


def query_array(query, model="text-embedding-3-small"):
    print("query_array")
    data = client.embeddings.create(input=[query], model=model).data[0].embedding
    query_array = np.array(data)
    query_array = query_array.reshape(1, -1)
    return query_array


def get_text_cosine_similarity(query_array, db_array, top_k, dataframe):
    print("get_text_cosine_similarity")
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(query_array, db_array)
    cosine_sim = cosine_sim.flatten()
    
    # Get indices of top K most similar entries
    top_indices = np.argsort(cosine_sim)[-top_k:][::-1]
    
    # Retrieve top entries from the dataframe
    top_df = dataframe.iloc[top_indices]
    text_list = top_df["text"].to_list()
    
    
    return text_list


def extract_content_based_on_query(query, top_k, folder_name):
    print("extract_content_based_on_query")

    # Construct the folder path
    folder_path = "Embeddings"

    # File path
    file_path = os.path.join(folder_path, f"{folder_name}_embedding.csv")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}.")
    
    # Extract array of embedding from the file
    db_array, dataframe = extract_array_of_embedding_from_file(file_path)

    # Query the array and get results
    array_query = query_array(query)
    resulted_text = get_text_cosine_similarity(array_query, db_array, top_k, dataframe)
    return resulted_text





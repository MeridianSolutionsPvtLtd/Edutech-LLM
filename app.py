from proctor import (verify_face, register_face, detect_faces, submit_exam)
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import *
from pydantic import BaseModel
import os
from rag_data_processing import *
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import List, Optional
from model import generate_question_store
import requests 
import tempfile
from azure.storage.blob import BlobServiceClient, BlobClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

origins = [
    "*"
]

app.add_middleware( 
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your Azure Blob Storage details
CONNECTION_STRING=os.getenv('BLOB_CONNECTION_STRING')
CONTAINER_NAME=os.getenv('CONTAINER_NAME_EMB')

blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

##############################    ********* CREATE - QUIZ *********   ##############################

class CreateQuizRequest(BaseModel):
    pdf_urls: List[str]  # List of URLs
    question_type: str
    easy_questions: int
    medium_questions: int
    hard_questions: int
    language: str = 'english'

@app.post("/create-quiz")
async def trigger_task(request: CreateQuizRequest):
    try:
        print("Starting .....")

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_folder:
            print(f"Temporary folder created: {temp_folder}")

            # Download PDFs
            for i, url in enumerate(request.pdf_urls):
                filename = f"downloaded_file_{i + 1}.pdf"
                download_pdf(url, temp_folder, filename)

            files = os.listdir(temp_folder)
            pdf_files = [f for f in files]
            word_limit_per_pdf = 14000 // len(pdf_files)
            small_pdf_threshold = 20  # Define a threshold for small PDFs based on page count

            total_chunks = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(temp_folder, pdf_file)
                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)
                print(f"Processing {pdf_file} with {total_pages} pages...")

                if total_pages <= small_pdf_threshold:
                    # Handle small PDFs
                    total_chunks.append(process_small_pdf(pdf_path))
                else:
                    # Handle large PDFs: Skip introductory pages
                    start_page = 5  # Example: Start after 4 introductory pages
                    pdf_content = process_large_pdf(pdf_path, start_page)
                    total_chunks.append(pdf_content)
                    print(f"Started reading from page {start_page} for {pdf_file}")

            # Combine and truncate content
            final_content = combine_and_truncate_content(total_chunks)
            content_list = chunks_string(final_content, 3500) if len(final_content.split()) > 3500 else [final_content]

            print("Content ready for question generation.")

            # Generate questions
            question_list = []
            if request.easy_questions > 0:
                questions = generate_question_store(content_list, request.easy_questions, 'easy', request.question_type, request.language)
                # print("Questionsssssss :", questions)
                question_list.extend(questions)
                print("added easy questions...")
            if request.medium_questions > 0:
                questions = generate_question_store(content_list, request.medium_questions, 'medium', request.question_type, request.language)
                question_list.extend(questions)
                print("added medium questions...")
            if request.hard_questions > 0:
                questions = generate_question_store(content_list, request.hard_questions, 'hard', request.question_type, request.language)
                question_list.extend(questions)
                print("added hard questions...")

            # Check if no questions were found
            if not question_list:
                raise HTTPException(status_code=404, detail="No questions found for the given input.")

            print("Returning questions ...")
            return {"questions": question_list, "status": "success", "message": "Quiz creation process started."}

    except Exception as e:
        print("Error:", str(e))
        return {"status": "failure", "message": str(e)}

##############################    ********* GENERATE - FEEDBACK *********   ##############################

# Define the request body model
class FeedbackRequest(BaseModel):
    question: str
    correct_answer: str
    user_answer: str
    question_type: int  # 1 for descriptive, 2 for numerical
    difficulty: str
    language: str = 'english'

# Define the response model
class FeedbackResponse(BaseModel):
    feedback: str
    score: str

@app.post("/feedback", response_model=FeedbackResponse)
async def get_feedback(request: FeedbackRequest):
    try:
        # Initialize variables
        feedback = ""
        score = "0"

        # Handle descriptive questions
        if request.question_type == 1:
            feedback, score = desc_marking(request.question, request.user_answer, request.correct_answer, request.difficulty, request.language)
            print(f"Descriptive Feedback: {feedback}")
            print(f"Descriptive Score: {score}")

        # Handle numerical questions
        elif request.question_type == 2:
            feedback, score = numerical_marking(request.question, request.user_answer, request.correct_answer, request.difficulty, request.language)
            print(f"Numerical Feedback: {feedback}")
            print(f"Numerical Score: {score}")

        else:
            raise HTTPException(status_code=400, detail="Invalid question type provided.")

        return FeedbackResponse(feedback=feedback, score=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
##############################    ********* CREATE - EMBEDDINGS *********   ##############################

class EmbeddingRequest(BaseModel):
    pdf_links: List[str]
    folder_name: str
    existing_embedding_links: Optional[List[str]] = []


def upload_blob_from_file(file_name: str, file_content: BytesIO, container_name: str):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
    blob_client.upload_blob(file_content, overwrite=True)


def delete_blob(container_name: str, prefix: str):
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blobs_to_delete = [blob.name for blob in container_client.list_blobs() if blob.name.startswith(prefix)]

        if blobs_to_delete:
            for blob_name in blobs_to_delete:
                blob_client = container_client.get_blob_client(blob_name)
                blob_client.delete_blob()
                print(f"Deleted blob: {blob_name}")
        else:
            print(f"No blobs found with prefix: {prefix}")

    except Exception as e:
        print(f"Error deleting blobs with prefix {prefix}: {str(e)}")

@app.post("/create-embeddings")
async def create_embeddings(request: EmbeddingRequest):

    print(f"Debug: folder_name - {request.folder_name}")

    # Check if there are existing embeddings to delete
    if request.existing_embedding_links:
        # Extract the prefix from each blob URL
        prefixes = set()
        for link in request.existing_embedding_links:
            blob_name = link.split(CONTAINER_NAME + "/")[1]  # Extract blob path
            prefix = blob_name.rsplit('/', 1)[0]  # Remove the file name to get the folder prefix
            prefixes.add(prefix)

        # Delete blobs for each prefix
        for prefix in prefixes:
            delete_blob(CONTAINER_NAME, prefix)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_folder:
        print(f"Temporary folder created: {temp_folder}")

        # Download PDFs into the temporary directory
        for i, url in enumerate(request.pdf_links):
            filename = f"downloaded_file_{i + 1}.pdf"
            download_pdf(url, temp_folder, filename)

        pdf_files = [f for f in os.listdir(temp_folder) if f.lower().endswith('.pdf')]
        total_chunks = []
        embedding_list = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(temp_folder, pdf_file)
            print(f"Reading {pdf_file}...")
            chunks = read_and_split_pdf_for_embedding(pdf_path, pdf_file)
            total_chunks += chunks

        print("File read and chunks generated.")

        # Generate embeddings
        for chunk in total_chunks:
            embedding = generate_embeddings(chunk[2])
            embedding_list.append(embedding)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(total_chunks, columns=['page_no', 'file_name', 'text'])
        df['embedding'] = embedding_list
        csv_file_path = os.path.join(temp_folder, f"{request.folder_name}_embedding.csv")
        df.to_csv(csv_file_path, index=False)
        
        # Upload the CSV file to Azure Blob Storage
        with open(csv_file_path, 'rb') as file:
            upload_blob_from_file(f"{request.folder_name}_embedding.csv", BytesIO(file.read()), CONTAINER_NAME)
    
    # Return response with links to newly created embeddings
    new_embedding_link = f"https://{blob_service_client.account_name}.blob.core.windows.net/{CONTAINER_NAME}/{request.folder_name}_embedding.csv"
    return {"Embedding_links": new_embedding_link}

####################################    ********* CHATBOT *********   ####################################

EMBEDDINGS_DIR = os.path.join(os.getcwd(), "Embeddings")

# Define request models using BaseModel
class InitialChatbotRequest(BaseModel):
    foldername: str
    blob_url: str

class QueryChatbotRequest(BaseModel):
    foldername: str
    query: str

@app.post("/initial-chatbot")
def initial_chatbot(request: InitialChatbotRequest):
    print("Starting ....")

    # Create the folder path as 'Embeddings'
    folder_path = "Embeddings"
    os.makedirs(folder_path, exist_ok=True)
    
    # Extract the file name from the blob URL
    file_path = os.path.join(folder_path, f"{request.foldername}_embedding.csv")
    print("File Path:", file_path)
    
    # Check if the file already exists
    if os.path.exists(file_path):
        print("File already exists. Using the existing file.")
    else:
        # Download the CSV file from the provided blob URL if the file doesn't exist
        response = requests.get(request.blob_url)
        response.raise_for_status()  # Ensure the request was successful
        
        # Write the content of the CSV file to the local file
        with open(file_path, "wb") as file:
            file.write(response.content)
    
    return {"message": "File is ready for use."}

@app.post("/query-chatbot")
def query_chatbot(request: QueryChatbotRequest):

    print("Starting ....")
    folder_path = "Embeddings"
    print("Folder path:", folder_path)

    file_path = os.path.join(folder_path, f"{request.foldername}_embedding.csv")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Embedding file not found. Please run initialize Chatbot first.")
    
    print("History Initialized ...")
    # Initialize history as an empty string
    history = " "
    
    # Correct the query and get the language
    print("Correcting Query...")
    language_response = language_correct_query(request.query, history)
    # query_string = language_response["Modified Content"]

    # Print the language_response for debugging
    print("Language Response:", language_response)
    
    query_string = language_response['Modified Content']
    if query_string is None:
        raise HTTPException(status_code=500, detail="Modified Content not found in language response.")
    
    # Extract content and citations based on the query
    print("Extracting content based on the query...")
    content_list = extract_content_based_on_query(query_string, 10, request.foldername)
    content = " ".join(content_list)
    
    # Get the response from the query
    print("Generating response...")
    response = get_response_from_query(query_string, content, history, language_response["Language"].strip().lower())
    
    # Prepare the output response
    print("Returning Response ...")
    output_response = {"bot_answer": response["bot_answer"]}
    return output_response

###########################################################################################################
# ---------------------------------------------------------------------------------------------------------
# Proctor

@app.get("/")
async def root():
    return {"message": "Welcome to the Face Detection API"}

# Face verification route
@app.post("/verify_face")
async def verify_face_route(image: UploadFile = File(...), camera_no: int = Form(...)):
    return await verify_face(image, camera_no)

# Face registration route
@app.post("/register_face")
async def register_face_route(username: str = Form(...), image: UploadFile = File(...)):
    return await register_face(username, image)

# Face detection route for proctoring
@app.post("/proctoring/detect-faces")
async def detect_faces_route(image: UploadFile = File(...)):
    return await detect_faces(image)

@app.post("/verify_face_during_exam")
async def verify_face_during_exam(image: UploadFile = File(...), camera_no: int = Form(...)):
    try:
        # Call the helper function to verify the face
        verification_result = await verify_face(image, camera_no)
 
        if verification_result["message"] == "No matching face found.":
            return verification_result['message']
        return verification_result["userid"]
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

# Submit exam route
class ExamSubmission(BaseModel):
    username: str
    quiz_id: str
    fullscreen_exit_count: int
    deviation_count: int
    submission_status: str
    noise_count:int

@app.post("/submit_exam")
async def submit_exam_route(submission: ExamSubmission):
    return await submit_exam(submission)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

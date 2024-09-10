import os
import uuid
import io
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.storage.blob import BlobServiceClient
from pymongo import MongoClient
from PIL import Image, ImageDraw
from datetime import datetime
from fastapi import UploadFile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Face API credentials
ENDPOINT = os.getenv('AZURE_FACE_API_ENDPOINT')
SUBSCRIPTION_KEY = os.getenv('AZURE_FACE_API_SUBSCRIPTION_KEY')
FACELIST_ID = os.getenv('AZURE_FACE_API_FACELIST_ID')

# Azure Blob Storage credentials
BLOB_CONNECTION_STRING = os.getenv('BLOB_CONNECTION_STRING')
CONTAINER_NAME = os.getenv('CONTAINER_NAME_FACE')

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI')
client = MongoClient(MONGODB_URI)
db = client['test']
collection = db['Azure_Face_API']

# Initialize FaceClient and BlobServiceClient
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY))
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)


async def verify_face(image: UploadFile, camera_no: int):
    image_path = f"temp_image_{uuid.uuid4()}.jpg"
    try:
        with open(image_path, "wb") as buffer:
            buffer.write(image.file.read())

        detected_faces = face_client.face.detect_with_stream(open(image_path, 'rb'))
        if not detected_faces:
            return {"message": "No face detected in the image."}

        face_id = detected_faces[0].face_id
        similar_faces = face_client.face.find_similar(face_id=face_id, face_list_id=FACELIST_ID)

        if similar_faces:
            matched_face_id = similar_faces[0].persisted_face_id
            user_info = collection.find_one({"persisted_faceid": matched_face_id})

            if user_info:
                current_time = datetime.now()
                collection.update_one(
                    {"persisted_faceid": matched_face_id},
                    {"$push": {"verification_logs": {"camera_no": camera_no, "timestamp": current_time}}}
                )
                return {"message": f"Verification successful for {user_info['username']}.", "userid": user_info['username']}
        return {"message": "No matching face found."}
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

async def register_face(username: str, image: UploadFile):
    image_path = f"{str(uuid.uuid4())}.jpg"
    try:
        existing_user = collection.find_one({"username": username})
        if existing_user:
            return {"message": "User already exists."}

        with open(image_path, "wb") as buffer:
            buffer.write(image.file.read())

        face_id = detect_face(image_path)
        persisted_face_id = add_face_to_facelist(image_path)
        azure_blob_path = upload_image_to_blob(image_path)
        insert_user_to_db(username, azure_blob_path, persisted_face_id)

        return {"message": "Face registered successfully."}
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

async def detect_faces(image: UploadFile):
    image_data = await image.read()
    image_stream = io.BytesIO(image_data)
    detected_faces = face_client.face.detect_with_stream(image=image_stream, detection_model='detection_03')
    image_stream.seek(0)
    image = Image.open(image_stream)
    draw = ImageDraw.Draw(image)

    for face in detected_faces:
        rect = face.face_rectangle
        left = rect.left
        top = rect.top
        right = left + rect.width
        bottom = top + rect.height
        draw.rectangle([left, top, right, bottom], outline="red", width=5)

    output_filename = f"detected_faces_image.jpg"
    image.save(output_filename, format="JPEG")
    return {"face_count": len(detected_faces), "saved_image": output_filename}

async def submit_exam(submission):
    existing_record = db['User_Deviations'].find_one({
        "username": submission.username,
        "quiz_id": submission.quiz_id
    })

    if existing_record:
        return {"message": "Record already exists for this username and quiz ID."}

    doc = {
        "username": submission.username,
        "quiz_id": submission.quiz_id,
        "fullscreen_exit_count": submission.fullscreen_exit_count,
        "deviation_count": submission.deviation_count,
        "timestamp": datetime.now(),
        "submission_status": submission.submission_status,
        "noise_Count":submission.noise_count
    }

    print(doc)
    db['User_Deviations'].insert_one(doc)
    return {"message": "Exam submitted successfully!"}

def detect_face(image_path):
    with open(image_path, 'rb') as image:
        detected_faces = face_client.face.detect_with_stream(image)
        if detected_faces:
            return detected_faces[0].face_id
        raise Exception("No face detected.")

def add_face_to_facelist(image_path):
    with open(image_path, 'rb') as image:
        added_face = face_client.face_list.add_face_from_stream(FACELIST_ID, image)
        return added_face.persisted_face_id

def upload_image_to_blob(image_path):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=image_path)
    with open(image_path, "rb") as data:
        blob_client.upload_blob(data)
    return f"https://{CONTAINER_NAME}.blob.core.windows.net/{image_path}"

def insert_user_to_db(username, blob_image_url, persisted_face_id):
    collection.insert_one({
        "username": username,
        "azure_image_url": blob_image_url,
        "persisted_faceid": persisted_face_id,
        "verification_logs": []
    })

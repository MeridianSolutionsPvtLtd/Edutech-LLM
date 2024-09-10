from PyPDF2 import PdfReader
import nltk
from fastapi import FastAPI, UploadFile, File, HTTPException
from nltk.tokenize import sent_tokenize, word_tokenize
from openai import AzureOpenAI
import time
import random
import string
import re
from rag_data_processing import *
import json


# Initialize NLTK
nltk.download('punkt')
nltk.download('punkt_tab')


# Initialize AzureOpenAI client
chat_client = AzureOpenAI(
    azure_endpoint="https://openainstance001.openai.azure.com/",  # Replace with your Azure OpenAI endpoint
    api_key="f619d2d04b4f44d28708e4c391039d01",  # Replace with your API key
    api_version="2024-02-15-preview"
)   


##############################    ********* CREATE - QUIZ *********   ##############################


def get_mcq_question(content, num_questions=10, level='medium', language='english'):
    prompt = f"""
    Generate exactly {num_questions} different MCQ questions based on the content provided. Each question should have 3 incorrect options and 1 correct option.
    
    Content: {content}

    Language: {language.capitalize()}
    
    Difficulty Level: {level.capitalize()}
    
    Instructions:
    - Generate only {num_questions} questions. Do not generate more or fewer questions in any case whatsoever.
    - The questions and answers must be in {language.capitalize()}. Ensure the questions should be clear and require the respondent to recall or recognize facts, terms, or concepts related to the content provided.
    - Ensure questions are appropriate for the specified difficulty level: 
      * Easy: Questions should cover basic facts and simple concepts.
      * Medium: Questions should require some understanding and the application of concepts.
      * Hard: Questions should challenge the respondent’s deeper understanding and ability to synthesize information.
    - Each question should have exactly 4 options (1 correct and 3 incorrect).
    - Questions should be neutral and objective, avoiding leading or judgmental phrasing.
    - Provide a correct answer and a concise explanation for each question.
    - Generate questions with respect to language provided. 
    - Generate the questions evenly across different sections of the content to ensure a diverse and comprehensive set of questions. Don't stick to single portion for generating all questions.
    - Tags must be in ENGLISH LANGUAGE ONLY for Questions, Options, Correct Answer and Explanation.

    Generate each MCQ according to the following format:
    Question 1: [Your question here]
    1) [Choice 1]
    2) [Choice 2]
    3) [Choice 3]
    4) [Choice 4]
    Correct Answer: [Correct choice, e.g., 1) [Choice 1]]
    Explanation: [Concise explanation that explains the answer]
    """

    message = [
        {"role": "system", "content": f"You are an AI assistant that helps to generate questions and answers from the content."},
        {"role": "user", "content": prompt}
    ]

    response = chat_client.chat.completions.create(
        model="gpt4",  # Replace with your deployment model name
        messages=message,
        temperature=0.7,
        max_tokens=3000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    response_text = response.choices[0].message.content
    # print(response_text)

    mcq_list = []
    questions = (response_text.strip().split("Question ") or 
             response_text.strip().split("السؤال ") or
             response_text.strip().split("Pytanie ") or
             response_text.strip().split("问题 ") or
             response_text.strip().split("प्रश्न "))


    for q in questions[1:]:
        lines = q.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        

        # Extracting the question text
        question_text = lines[0].split(":", 1)[1].strip()
        
        # Extracting the options
        options = [line.split(") ", 1)[1].strip() for line in lines[1:5]]
        

        # Finding the correct answer
        correct_answer_line = [
            line for line in lines if line.startswith("الإجابة الصحيحة")  # Arabic
            or line.startswith("Correct Answer")  # English
            or line.startswith("सही उत्तर")  # Hindi
            or line.startswith("Prawidłowa odpowiedź")  # Polish
            or line.startswith("正确答案")  # Standard Chinese
        ][0]
        correct_answer = correct_answer_line.split(":", 1)[1].strip()[0]

        # Extracting the explanation
        explanation_line = [
            line for line in lines if line.startswith("التوضيح") or line.startswith("التفسير") # Arabic
            or line.startswith("Explanation")  # English
            or line.startswith("व्याख्या")  # Hindi
            or line.startswith("Wyjaśnienie")  # Polish
            or line.startswith("解释")  # Standard Chinese
        ][0]
        explanation = explanation_line.split(": ", 1)[1].strip()
            

        # Append to mcq_list with the correct identifier for MCQs
        mcq_list.append(["1", question_text, options, correct_answer, explanation])


    return mcq_list


def get_descriptive_question(content, num_questions=5, level='medium', language='english'):
    prompt = f"""
    Generate exactly {num_questions} descriptive questions based on the content provided. Each question should have a long, detailed answer, along with an explanation of how the answer was derived.
    
    Content: {content}

    Language: {language.capitalize()}
    
    Difficulty Level: {level.capitalize()}
    
    Instructions:
    - Generate only {num_questions} questions. Do not generate more or fewer questions in any case whatsoever.
    - The questions and answers must be in {language.capitalize()}. Ensure the questions are clear and require the respondent to recall or recognize facts, terms, or concepts related to the content provided above.
    - Provide strictly correct and detailed descriptive answers.
    - Include an explanation of how the answer was derived for each question.
    - The questions should be appropriate for the target audience, written in a neutral and objective manner.
    - Ensure questions are appropriate for the specified difficulty level:
      * Easy: Simple concepts with straightforward answers and explanations.
      * Medium: Questions requiring some analysis and multi-step answers with explanations.
      * Hard: Complex questions that require deep understanding and comprehensive answers with detailed explanations.
    - Avoid asking leading questions or those that require inferences or judgments.
    - Generate a correct answer and an explanation for each question.
    - Follow the format carefully (i.e., Question, Correct Answer, Explanation) as given below for each question.
    - Generate all content (questions, answers, explanations) in {language.capitalize()}.
    - Tags must be in ENGLISH LANGUAGE ONLY for Questions, Correct Answer and Explanation.
    
    Generate the response according to the following format:
    Question 1: [Your question here]
    Correct Answer: [Answer]
    Explanation: [Explanation of how the answer was derived]
    """

    message = [
        {"role": "system", "content": f"You are an AI assistant that generates accurate and well-structured questions and answers from the content."},
        {"role": "user", "content": prompt}
    ]

    response = chat_client.chat.completions.create(
        model="gpt4",  # Replace with your deployment model name
        messages=message,
        temperature=0.7,
        max_tokens=3000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    response_text = response.choices[0].message.content
    # print("Response_Text:", response_text)

    desc_list = []
    questions = (response_text.strip().split("Question ") or 
             response_text.strip().split("السؤال ") or
             response_text.strip().split("Pytanie ") or
             response_text.strip().split("问题 ") or
             response_text.strip().split("प्रश्न "))

    for q in questions[1:]:
        lines = q.strip().split("\n")
        question_text = lines[0].split(":", 1)[1].strip()
        correct_answer_line = [
            line for line in lines if line.startswith("الإجابة الصحيحة")  # Arabic
            or line.startswith("Correct Answer")  # English
            or line.startswith("सही उत्तर")  # Hindi
            or line.startswith("Prawidłowa odpowiedź")  # Polish
            or line.startswith("正确答案")  # Standard Chinese
        ][0]
        correct_answer = correct_answer_line.split(": ")[1].strip()
        explanation_line = [
            line for line in lines if line.startswith("التوضيح") or line.startswith("التفسير")  # Arabic
            or line.startswith("Explanation")  # English
            or line.startswith("व्याख्या")  # Hindi
            or line.startswith("Wyjaśnienie")  # Polish
            or line.startswith("解释")  # Standard Chinese
        ][0]
        explanation = explanation_line.split(": ", 1)[1].strip()
        desc_list.append(["2", question_text, correct_answer, explanation])

        # print("DESC:", desc_list)

    return desc_list


def get_numerical_question(content, num_questions=10, level='medium', language='english'):
    prompt = f"""
    Generate exactly {num_questions} different numerical type questions based on the content below. 
    Provide a correct numerical answer for each question.
    
    Content: {content}

    Language: {language.capitalize()}
    
    Difficulty Level: {level.capitalize()}
    
    Instructions:
    - Generate only {num_questions} questions. Do not generate more or fewer questions in any case whatsoever.
    - The questions must be in {language.capitalize()}. Ensure the questions should be clear and require the respondent to perform calculations or analyze numerical data related to the content.
    - Ensure questions are appropriate for the specified difficulty level:
      * Easy: Simple calculations with clear steps.
      * Medium: Multi-step problems requiring moderate calculations.
      * Hard: Complex calculations that require advanced understanding.
    - Provide the correct numerical answer and a brief explanation of how it was derived.
    - Generate all content (questions, answers, explanations) in {language.capitalize()}.
    - Tags must be in ENGLISH LANGUAGE ONLY for Questions, Correct Answer and Explanation.

    Format:
    Question 1: [Your question here]
    Correct Answer: [Numerical answer here]
    Explanation: [Explanation of how the answer was derived]
    """

    message = [
        {"role": "system", "content": f"You are an AI assistant that generates accurate and well-structured questions and answers."},
        {"role": "user", "content": prompt}
    ]

    response = chat_client.chat.completions.create(
        model="gpt4",  # Replace with your deployment model name
        messages=message,
        temperature=0.7 ,
        max_tokens=3000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )


    response_text = response.choices[0].message.content

    numerical_list = []
    questions = (response_text.strip().split("Question ") or 
             response_text.strip().split("السؤال ") or
             response_text.strip().split("Pytanie ") or
             response_text.strip().split("问题 ") or
             response_text.strip().split("प्रश्न "))

    for q in questions[1:]:
        lines = q.strip().split("\n")
        question_number = "3"
        question_text = lines[0].split(":", 1)[1].strip()
        correct_answer_line = [
            line for line in lines if line.startswith("الإجابة الصحيحة")  # Arabic
            or line.startswith("Correct Answer")  # English
            or line.startswith("सही उत्तर")  # Hindi
            or line.startswith("Prawidłowa odpowiedź")  # Polish
            or line.startswith("正确答案")  # Standard Chinese
        ][0]
        correct_answer = correct_answer_line.split(": ", 1)[1].strip()
        explanation_line = [
            line for line in lines if line.startswith("التوضيح") or line.startswith("التفسير")  # Arabic
            or line.startswith("Explanation")  # English
            or line.startswith("व्याख्या")  # Hindi
            or line.startswith("Wyjaśnienie")  # Polish
            or line.startswith("解释")  # Standard Chinese
        ][0]
        explanation = explanation_line.split(": ", 1)[1].strip()
        numerical_list.append([question_number, question_text, correct_answer, explanation])

    return numerical_list


def generate_question_store(content_list, num_questions=15, difficulty="medium", question_type="mcq", language='english'):
    
    print("Generating questions...")
    
    # Combine content if necessary
    combined_content = " ".join(content_list)
    
    # Dynamically call the appropriate function based on question_type
    if question_type == "mcq":
        question_generator = get_mcq_question
    elif question_type == "descriptive":
        question_generator = get_descriptive_question
    elif question_type == "numerical":
        question_generator = get_numerical_question  # Assuming you have this function
    else:
        raise ValueError(f"Unsupported question_type: {question_type}")

    print("Function picked for generation ..")

    # Generate the exact number of questions
    question_list = question_generator(combined_content, num_questions=num_questions, level=difficulty, language=language)
    # print("Model questions:", question_list)

    print("Questions generated...")
    # print(question_list)

    formatted_questions = []
    for question in question_list:
        print("Starting ...")
        if question_type == "mcq" and question[0] == "1":
            print("MCQ")
            formatted_questions.append({
                "Questions": question[1],
                "Options": [question[2][0], question[2][1], question[2][2], question[2][3]],
                "Answer": question[3],
                "Difficulty": difficulty,
                "Explanation": question[4]
            })
            print("Done")
        elif question_type == "descriptive" and question[0] == "2":
            print("Descriptive")
            formatted_questions.append({
                "Questions": question[1],
                "Answer": question[2],
                "Difficulty": difficulty,
                "Explanation": question[3]
            })
            print("Done")
        elif question_type == "numerical" and question[0] == "3":
            print("Numerical")
            formatted_questions.append({
                "Questions": question[1],
                "Answer": question[2],  # Assuming numerical questions have a correct answer stored similarly
                "Difficulty": difficulty,
                "Explanation": question[3]
            })
            print("Done")
        else:
            print(f"Unsupported question type or format: {question_type} and {question[0]}")
    
    print("Questions insertion process completed.")
    return formatted_questions  # Return the list of formatted questions


##############################    ********* GENERATE - FEEDBACK *********   ##############################


def desc_marking(question, student_response, correct_answer, difficulty, language='english'):
    prompt = f"""
    Your task is to act as a feedback system where you need to evaluate and give feedback for the student's response to the question provided below.

    Question: {question}

    Student's Response: {student_response}

    Correct Answer: {correct_answer}

    Language: {language.capitalize()}

    Difficulty Level: {difficulty.capitalize()}

    Instructions:
    - Evaluate the student's response and provide a score between 0 to 100 based on correctness and relevance, considering the difficulty level.
    - For easier questions, you may be more lenient in grading, but still ensure that key concepts are covered.
    - For medium difficulty questions, ensure the response includes more detailed understanding and context.
    - For hard questions, expect comprehensive and in-depth responses, and be strict about the accuracy and relevance of the content.
    - Correct answer is provided just for reference do not mark on the basis of sentence to sentence matching. 
    - You will always give feedback for the student answer with the question. 
    - Feedback should be given in a manner where students can help themselves to improve. 
    - It should give 0 for the spam responses. 
    - There should not be any instructions in the response. 
    - Just for the reference if the score is greater than or equal to 50, it is considered as pass.
    - Generate feedback in {language.capitalize()}.

    Generate feedback according to the below format:

    Feedback: [Your feedback here]
    Score: [Your score here]
    """

    message = [
        {"role": "system", "content": "You are an AI assistant designed to provide feedback and grading for descriptive questions."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = chat_client.chat.completions.create(
            model="gpt4",  # Ensure you're using the correct model name
            messages=message,
            temperature=0.7,
            max_tokens=300,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract feedback and score from the response content
        response_content = response.choices[0].message.content.strip()

        feedback_match = re.search(r'Feedback\s*:\s*(.*)', response_content)
        score_match = re.search(r'Score\s*:\s*(\d+)', response_content)

        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback found."
        score = score_match.group(1).strip() if score_match else "0"
        
        return feedback, score
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error generating feedback.", "0"


def numerical_marking(question, student_answer, correct_answer, difficulty, language='english'):
    prompt = f"""
    Your task is to evaluate and provide feedback for a student’s numerical answer to the following question.

    Question: {question}

    Student's Answer: {student_answer}

    Correct Answer: {correct_answer}

    Language: {language.capitalize()}

    Difficulty Level: {difficulty.capitalize()}

    Instructions:
    - Assess the student’s numerical answer in relation to the correct answer.
    - Provide a score between 0 and 100 based on the accuracy of the student’s response.
    - Offer feedback that explains why the answer is correct or incorrect, and how the student can improve.
    - Focus on the correctness and methodology, not exact phrasing.
    - Provide 0 for responses deemed as spam or irrelevant.
    - Generate feedback in {language.capitalize()}.

    Generate feedback according to the below format:

    Feedback : [Feedback]
    Score: [Score]
    """

    message = [
        {"role": "system", "content": "You are an AI assistant designed to provide feedback and grading for numerical questions."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = chat_client.chat.completions.create(
            model="gpt4",  # Replace with your deployment model name
            messages=message,
            temperature=0.7,
            max_tokens=300,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )
            
        # Extract feedback and score from the response content
        response_content = response.choices[0].message.content.strip()

        feedback_match = re.search(r'Feedback\s*:\s*(.*)', response_content)
        score_match = re.search(r'Score\s*:\s*(\d+)', response_content)

        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback found."
        score = score_match.group(1).strip() if score_match else "0"
        
        return feedback, score  
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error generating feedback.", "0"


####################################    ********* CHATBOT *********   ####################################


def language_correct_query(query, history):
    message = [
        {"role": "system", "content": "You are an AI assistant that helps to identify and extract the language, fixes the typing error and change the any language into English language content by understanding the user query."},
        {"role": "user", "content": f"""Your task is to identify and extract the language of the query string, fix any typing errors, and change the language of input content into English if it is in another language. Give the response always in JSON format only.\n\nInput Content: {query}\n\nHistory: {history}\n\nImportant instructions: \n1. Your task is to identify the language of content.\n2. Generate the modified content by fixing the typing error and translating to English if necessary.\n\nKey Entities for the JSON response:\n1. Language\n2. Modified Content\n\nExtracted JSON Response:"""}
    ]

    response = chat_client.chat.completions.create(
        model="gpt4",
        messages=message,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    
    json_response = json.loads(response.choices[0].message.content)
    return json_response


def get_response_from_query(query, content, history, language):
    prompt_template = f"""
    Your task is to follow the chain of thought method to first extract an accurate answer for the given user query, chat history, and provided input content. Then change the language of the response into {language}. Provide the response in JSON format only with 'bot_answer' and 'scope' as keys.

    Input Content: {content}

    User Query: {query}

    Chat History: {history}

    Important Points:
    1. The answer should be relevant to the input text.
    2. Answer complexity should match the input content.
    3. If input content is missing, direct the user to provide content.
    4. Answers should be safe and appropriate. If not, give instructions to the user.
    5. If the user query is out of scope, set the 'scope' key to False.

    Extracted JSON response:
    """

    message = [
        {"role": "system", "content": f"You are an AI assistant that helps to answer the questions from the given content in {language} language."},
        {"role": "user", "content": prompt_template}
    ]

    response = chat_client.chat.completions.create(
        model="gpt4",
        messages=message,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    
    json_response = json.loads(response.choices[0].message.content)
    print(json_response)
    return json_response


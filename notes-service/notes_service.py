import gc, traceback, os
import ujson as json
import orjson
import asyncio
from typing import Union, Dict, Any, List
from llama_index.core import (
    VectorStoreIndex,
    Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.llms.openai import OpenAI
from openai import AsyncOpenAI
from quart import Quart, request, jsonify, Response, g
from google.cloud import secretmanager
from google.auth import default
from quart_cors import cors
import time, logging
from aiohttp import ClientSession, ClientTimeout
from collections import defaultdict
from collections import Counter
from google.cloud import logging as cloud_logging
import logging
from contextlib import asynccontextmanager
import psutil
from fuzzywuzzy import fuzz

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Google Cloud Logging
cloud_logging_client = cloud_logging.Client()
cloud_logger = cloud_logging_client.logger('notes-service-log')

def log_message(severity, message):
    """Log messages to both console and Google Cloud Logging."""
    logger.log(getattr(logging, severity), message)
    cloud_logger.log_text(message, severity=severity)

log_message('INFO', 'Starting Notes Service')

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini-2024-07-18"
MAX_RETRIES = 5

app = Quart(__name__)
app = cors(app, allow_origin="*")

@asynccontextmanager
async def get_pinecone_index(pc_client: Pinecone, index_name: str):
    index = pc_client.Index(index_name)
    try:
        yield index
    finally:
        pass  # No cleanup needed for Pinecone

@asynccontextmanager
async def get_vector_store(pinecone_index):
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    try:
        yield vector_store
    finally:
        pass  # Add cleanup if necessary

@asynccontextmanager
async def get_vector_index(vector_store):
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    try:
        yield vector_index
    finally:
        pass  

@asynccontextmanager
async def get_vector_retriever(vector_index, similarity_top_k=8):
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=similarity_top_k)
    try:
        yield vector_retriever
    finally:
        pass  

@asynccontextmanager
async def get_openai_client(api_key):
    client = AsyncOpenAI(api_key=api_key)
    try:
        yield client
    finally:
        await client.close()

def log_memory_usage(request_id):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    log_message('INFO', f"[{request_id}] Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

async def update_token_count(counter, model, tokens):
    counter[model] += tokens
    return counter  

async def get_token_usage(counter):
    model_mapping = {
        "gpt-3.5-turbo-0125": "gpt_3_5_turbo_0125",
        "gpt-4o-mini-2024-07-18": "gpt_4o_mini_2024_07_18",
        "gpt-4o-2024-05-13": "gpt_4o_2024_05_13",
        "gpt-4-turbo-2024-04-09": "gpt_4_turbo_2024_04_09",
        "gpt-4-0613": "gpt_4_0613",
        "text-embedding-3-small": "text_embedding_3_small"
    }
    
    all_models = {v: 0 for v in model_mapping.values()}
    
    for model, count in counter.items():
        if model in model_mapping:
            all_models[model_mapping[model]] = count
        else:
            logger.warning(f"Unknown model in counter: {model}")
    return all_models
        
credentials, project_id = default()
log_message('INFO', f"Authenticated with project ID: {project_id}")
# if credentials is not None:
#     # logging.debug(f"Authenticated account: {credentials.service_account_email}")
#     log_message('DEBUG', f"Authenticated account: {credentials.service_account_email}")
    
# else:
#     log_message('DEBUG', "No credentials found.")
#     raise ValueError("Credentials not found")

def access_secret_version(project_id, secret_id, version_id):
    client = secretmanager.SecretManagerServiceClient(credentials=credentials)
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    log_message('DEBUG', f"Accessing secret: {name}")
    response = client.access_secret_version(name=name)
    payload = response.payload.data.decode("UTF-8")
    log_message('INFO', f"Successfully retrieved secret: {secret_id}")
    return payload


openai_secret_id = "openai-api-key"
openai_secret_version = "latest"
pinecone_secret_id = "pinecone-api-key"
pinecone_secret_version = "latest"

openai_api_key = access_secret_version(project_id, openai_secret_id, openai_secret_version)
pinecone_api_key = access_secret_version(project_id, pinecone_secret_id, pinecone_secret_version)
client = AsyncOpenAI(api_key=openai_api_key)

pc = Pinecone(api_key=pinecone_api_key)

Settings.llm = OpenAI()
embed_model = OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-3-small")
Settings.embed_model = embed_model

async def get_vector_store(index_name):
    pinecone_index = pc.Index(index_name)
    return PineconeVectorStore(pinecone_index=pinecone_index)

async def fetch_openai_response(client, model: str, messages: List[Dict[str, str]], token_counter: Counter, request_id: str) -> Dict[str, Any]:
    try:
        log_message('DEBUG', f"[{request_id}] Fetching OpenAI response with model: {model}")
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        await update_token_count(token_counter, response.model, response.usage.total_tokens)
        
        content = response.choices[0].message.content
        log_message('DEBUG', f"[{request_id}] Received response from OpenAI for model {model}")
        return orjson.loads(content)
    
    except Exception as e:
        log_message('ERROR', f"[{request_id}] Error fetching OpenAI response: {str(e)}")
        return {"error": str(e)}

async def query_technical_index(query_text: str, index_name: str, pc_client: Pinecone, openai_api_key: str, token_counter: Counter, request_id: str) -> str:
    log_message('INFO', f"[{request_id}] Starting query_technical_index with query: {query_text}")
    log_memory_usage(request_id)
    
    try:
        pinecone_index = pc_client.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=15)

        async with get_openai_client(openai_api_key) as async_client:
            raw_nodes = await asyncio.to_thread(vector_retriever.retrieve, query_text)
            # print("Raw Nodes:\n",raw_nodes)
            text_results, equation_results, image_results = categorize_nodes(raw_nodes)            
            
            # start0 = time.time()
            text_context, equation_context, image_context = await asyncio.gather(
                generate_text_context(text_results),
                generate_equation_context(equation_results),
                generate_image_context(image_results)
            )
            # end0 = time.time()
            # elapsed0 = end0 - start0
            # print(f"Elapsed time generate context for individual data types {elapsed0:.7f} seconds")
            
            # start1 = time.time()
            text_messages, equation_messages, image_messages = await asyncio.gather(
                prepare_text_messages(query_text, text_context),
                prepare_equation_messages(query_text, equation_context),
                prepare_image_messages(query_text, image_context)
            )
            # end1 = time.time()
            # elapsed1 = end1 - start1
            # print(f"Elapsed time prepare individual data types {elapsed1:.7f} seconds")

            log_message('DEBUG', f"[{request_id}] Fetching OpenAI responses")
            
            # start2 = time.time()
            text_response, equation_response, image_response = await asyncio.gather(
                fetch_openai_response(async_client, 'gpt-4o', text_messages, token_counter, request_id),
                fetch_openai_response(async_client, 'gpt-4o-mini-2024-07-18', equation_messages, token_counter, request_id),
                fetch_openai_response(async_client, 'gpt-4o-mini-2024-07-18', image_messages, token_counter, request_id)
            )
            # end2 = time.time()
            # elapsed2 = end2 - start2
            # print(f"Elapsed time for fetching openai resposnes for each data types {elapsed2:.7f} seconds")            
            log_message('DEBUG', f"[{request_id}] OpenAI responses received")

            log_message('DEBUG', f"[{request_id}] Combining responses")

            combined_response = await combine_responses(async_client, text_response, equation_response, image_response, token_counter, request_id)
            log_message('INFO', f"[{request_id}] Query_technical_index completed successfully")

            return orjson.dumps(combined_response).decode('utf-8')

    except Exception as e:
        log_message('ERROR', f"[{request_id}] Error in query_technical_index: {str(e)}")
        log_message('ERROR', f"[{request_id}] Traceback: {traceback.format_exc()}")
        return orjson.dumps({"error": str(e)}).decode('utf-8')
    finally:
        log_message('DEBUG', f"[{request_id}] Cleaning up resources")
        del vector_store, vector_index, vector_retriever
        gc.collect()
        log_memory_usage(request_id)

def categorize_nodes(raw_nodes):
    text_results, equation_results, image_results = [], [], []
    # print("RAW NODES:<<<<<\n",raw_nodes)

    for node in raw_nodes:
        content_type = node.metadata.get('content_type')
        if content_type == 'text':
            text_results.append(node)
            if content_type == 'page_path':
                text_results.append(node)
        elif content_type == 'equation':
            equation_results.append(node)
            if content_type == 'page_path':
                equation_results.append(node)            
        elif content_type == 'image':
            image_results.append(node)
            if content_type == 'page_path':
                image_results.append(node)    

    # print("TEXT RESULT>>>>>>>:\n",text_results)        
    # print("EQUATION RESULT>>>>>>>:\n",equation_results)        
    # print("IMAGE RESULT>>>>>>>:\n",image_results)        
    return text_results[:3], equation_results[:3], image_results[:1]

async def generate_text_context(text_results):
    # print("Final Text Result:\n",text_results)
    return "\n\n".join([f"Top Text {i+1} from Lecture {text.metadata.get('lecture_number')}\n"
                        f"Description: {text.text or 'No description available'}\n" 
                        f"Document Link: {text.metadata.get('page_path')}\n"
                        for i, text in enumerate(text_results)])

async def generate_equation_context(equation_results):
    # print("Final Equation Result:\n",equation_results)
    return "\n\n".join([f"Top Equation {i+1} from Lecture {equation.metadata.get('lecture_number')}\n"
                        f"Path: {equation.metadata.get('equation_path')}\n"
                        f"Description: {equation.text or 'No description available'}\n" 
                        f"Document Link: {equation.metadata.get('page_path')}\n"
                        for i, equation in enumerate(equation_results)])

async def generate_image_context(image_results):
    # print("Final Image Result:\n",image_results)
    return "\n\n".join([f"Top Image {i+1} from Lecture {image.metadata.get('lecture_number')}\n"
                        f"Path: {image.metadata.get('image_path')}\n"
                        f"Description: {image.text or 'No description available'}\n" 
                        f"Document Link: {image.metadata.get('page_path')}\n"
                        for i, image in enumerate(image_results)])

async def prepare_text_messages(query_text, text_context):
    return [
        {"role": "system", "content": "You are a helpful assistant for a fluid dynamics course."},
        {"role": "user", "content": f"""
        Your tasks are the following:
        0. Read the user's question and understand what they are asking. Criteria Point - remember it is their first time learning about these concepts, we must provide simple and informative responses - `{query_text}`. 
        1. Then read the following context that is returned as being the most relevant from this course's lecture slide index database: `{text_context}`.
        2. The user's question MUST be directly answered ONLY using this provided Text context. If the question can be answered, populate the following JSON with can_answer field as true and relevant info, remembering the Criteria Point.
        3. If the user's question cannot be answered based solely on the provided context, populate the following JSON with can_answer field as false, and the text field as "Sorry, insert_the_single_topic_ONLY doesn't seem to be present within Professor Amirfazli's lectures. Please inquire about topics covered in MECH3202.".
        4. Populate the JSON "can_answer" key with: true if answerable
        {{
            "can_answer": true/false,
            "Text": "Your answer or the apology message",
            "reference": "Insert lecture number and page here."
            "page_path": "`insert_URL_link_to_lecture_slide_here`"
        }}
        7. IMPORTANT: Do NOT use any external knowledge. Base your response entirely on the provided Text context.
        """
    }]

async def prepare_equation_messages(query_text, equation_context):
    # print("EQUATION CONTEXT:\n", equation_context)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
        Your tasks are the following:
        0. Read the user's question and understand what they are asking: `{query_text}`
        Criteria Points: 
        - Remember it is their first time learning about these concepts, we must provide simple and informative response explaining HOW the equation can be used for their question!
        - The selected equation(s) should ONLY be passed to the user if it helps them further grasp the concept behind their question, if it just seems related and doesn't actually help DO NOT return it!
        - If there are multiple equations for the user that you gauge should be included as it will help them understand the concept behind their question, you may include them.
        1. Then read the following context that is returned as being the most relevant equations from this course's lecture slide index database: `{equation_context}`.
        2. The user's question MUST directly answered ONLY using this provided equations context. If the question can be answered, populate the following JSON, remembering the Criteria Points.
        3. If the no equations provided can help the user understand their question better, simply return empty fields.                        
        {{
            "Equation1": "Equation in LaTeX JSON math format ONLY.",
            "EquationDescription1": "Equation description for the user.",
            "equation_link1": "`insert_URL_link_to_lecture_slide_here`" 
            ... *additional equation/equationdescription/equationlink set here IF NECESSARY/APPROPRIATE ONLY, THIS DOES NOT MEAN TO ALWAYS INCLUDE ADDITIONAL EQUATIONS!!!* ...       
        }}
            """}
        ]

async def prepare_image_messages(query_text, image_context):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
        Your tasks are the following:
        0. Read the user's question and understand what they are asking: `{query_text}`
        Criteria Points: 
        - Remember it is their first time learning about these concepts, we must provide simple and informative responses.
        - The selected equation(s) should ONLY be passed to the user if it helps them further grasp the concept behind their question, if it just seems related and doesn't actually help DO NOT return it!
        - If there are multiple equations for the user that you gauge should be included as it will help them understand the concept behind their question, you may include them.                
        1. Then read the following context that is returned as being the most relevant from this course's lecture slide index database: `{image_context}`.
        2. The user's question MUST directly answered ONLY using this provided image context. If the question can be answered, populate the following JSON, remembering the Criteria Points.
        3. If the no images provided can help the user understand their question better, simply return empty fields.        
        {{
            "Image1": "Public Image URL Path.",
            "ImageDescription1": "Image description for the user.",
            "image_link1": "`insert_URL_link_to_lecture_slide_here`"        
        }}
            """}
        ]

async def combine_responses(
    async_client,
    text_response: Union[Dict[str, Any], str],
    equation_response: Union[Dict[str, Any], str],
    image_response: Union[Dict[str, Any], str],
    token_counter: Counter,
    request_id: str
) -> Dict[str, Any]:
    log_message('DEBUG', f"[{request_id}] Combining responses")
    try:
        start = time.time()
        # Ensure responses are dictionaries
        text_response = json.loads(text_response) if isinstance(text_response, str) else text_response
        equation_response = json.loads(equation_response) if isinstance(equation_response, str) else equation_response
        image_response = json.loads(image_response) if isinstance(image_response, str) else image_response

        combined_response = {
            "Text": "",
            "reference": "",
            "text_link": ""
        }

        can_answer = text_response.get('can_answer', False)
        combined_response['can_answer'] = can_answer

        if can_answer:
            combined_response['Text'] = text_response.get('Text', '')
            combined_response['reference'] = text_response.get('reference', '')
            combined_response['text_link'] = text_response.get('page_path', '')

            # Handle multiple equations
            equation_index = 1
            while f'Equation{equation_index}' in equation_response:
                combined_response[f'Equation{equation_index}'] = equation_response.get(f'Equation{equation_index}', '')
                combined_response[f'EquationDescription{equation_index}'] = equation_response.get(f'EquationDescription{equation_index}', '')
                combined_response[f'equation_link{equation_index}'] = equation_response.get(f'equation_link{equation_index}', '')
                equation_index += 1

            # Handle multiple images
            image_index = 1
            while f'Image{image_index}' in image_response:
                combined_response[f'Image{image_index}'] = image_response.get(f'Image{image_index}', '')
                combined_response[f'ImageDescription{image_index}'] = image_response.get(f'ImageDescription{image_index}', '')
                combined_response[f'image_link{image_index}'] = image_response.get(f'image_link{image_index}', '')
                image_index += 1
        else:
            combined_response['Text'] = text_response.get('Text', '')

        # Update token counter (if needed)
        # await update_token_count(token_counter, "combine_responses", 0)  # Uncomment and adjust if needed
        end = time.time()
        elapsed = end - start
        print(f"Elapsed time {elapsed:.7f} seconds")
        return combined_response

    except Exception as e:
        log_message('ERROR', f"[{request_id}] Error in combine_responses: {str(e)}")
        return {"error": str(e)}

def decode_unicode_escapes(obj):
    if isinstance(obj, str):
        return obj.encode('utf-8').decode('unicode_escape')
    elif isinstance(obj, dict):
        return {k: decode_unicode_escapes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_unicode_escapes(i) for i in obj]
    else:
        return obj

async def query_syllabus_index(query_text, index_name, token_counter: Counter, request_id: str):
    log_message('INFO', f"[{request_id}] Starting query_syllabus_index with query: {query_text}")
    log_memory_usage(request_id)
    try:
        # Preprocess the query
        query_lower = query_text.lower()
        # print("Query Lower:\n",query_lower)
        # print("Processed Query:\n",processed_query)
        
        # log_message('DEBUG', f"[{request_id}] Processed query: {processed_query}")

        # log_message('DEBUG', f"[{request_id}] Initializing Pinecone index: {index_name}")
        pinecone_index = pc.Index(index_name)
        # log_message('DEBUG', f"[{request_id}] Creating PineconeVectorStore")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)     
        
        # log_message('DEBUG', f"[{request_id}] Creating VectorStoreIndex")
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        # log_message('DEBUG', f"[{request_id}] Creating VectorIndexRetriever")
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=5
        )
        
        # log_message('DEBUG', f"[{request_id}] Retrieving raw nodes")
        raw_nodes = vector_retriever.retrieve(query_lower)
        # log_message('DEBUG', f"[{request_id}] Retrieved {len(raw_nodes)} raw nodes")
        
        text_results = []
        for node in raw_nodes:
            text_results.append(node.text)
            
        top_texts = text_results[:3] if text_results else []
        # print(f"[{request_id}] Top 5 text results extracted \n {top_texts}")
        
        log_message('DEBUG', f"[{request_id}] Initializing AsyncOpenAI client")
        client = AsyncOpenAI(api_key=openai_api_key)
        
        log_message('DEBUG', f"[{request_id}] Sending request to OpenAI")
        completion_response = await client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""
            Tasks:
            1. You must provide a precise and helpful answer to the student's question about the course syllabus, here is the question: `{query_lower}`.
            2. You MUST ONLY use the content returned in the following syllabus context: `{top_texts}`.
            3. If the answer is not present, simply return an apology and to email prof/TA is urgent. DO NOT tell them to refer to a page in this case, leave it blank!
        
            4. Populate the following JSON. 

            Response Format:
            {{
                "Text": "Your answer here."
                "Reference": "Please refer to `insert_page_number_here.lower()` on the official syllabus."
            }}
            """}
        ], 
            temperature=0,
            max_tokens=500
        )        
        
        log_message('DEBUG', f"[{request_id}] OpenAI response received")
        await update_token_count(token_counter, completion_response.model, completion_response.usage.total_tokens)
        
        content = completion_response.choices[0].message.content.strip()
        content = content.strip('```json').strip('```')
        try:
            log_message('DEBUG', f"[{request_id}] Parsing JSON response")
            response_json = json.loads(content)
            log_message('INFO', f"[{request_id}] Query_syllabus_index completed successfully")
            return jsonify(response_json)
        except json.JSONDecodeError as e:
            log_message('ERROR', f"[{request_id}] Failed to parse JSON response: {e}")
            return jsonify({"error": "Failed to parse JSON response"})
    except Exception as e:
        log_message('ERROR', f"[{request_id}] Error in query_syllabus_index: {str(e)}")
        log_message('ERROR', f"[{request_id}] Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)})
    finally:
        log_message('DEBUG', f"[{request_id}] Cleaning up resources")
        gc.collect()
        log_memory_usage(request_id)
        
@app.route('/query_technical', methods=['POST'])
async def handle_technical_query():
    try:
        data = await request.get_json()
        request_id = data.get('request_id', 'N/A')
        user_query = data.get('query', '')
        log_message('INFO', f"[{request_id}] User Query: {user_query}")

        token_counter = Counter()  
        index_name = "index-technical-final"            
        
        start = time.time()
        log_message('INFO', f"[{request_id}] Querying Technical Index...")
        response_json_str = await query_technical_index(user_query, index_name, pc, openai_api_key, token_counter, request_id)
        log_message('INFO', f"[{request_id}] Finished Querying Technical Index...")
        end = time.time()
        elapsed = end - start
        
        response_content = orjson.loads(response_json_str)
        token_usage = await get_token_usage(token_counter)

        combined_response = {
            "response": response_content,
            "token_usage": token_usage,
            "elapsed_time": f"{elapsed:.2f} seconds"
        }
        
        combined_response_json = orjson.dumps(combined_response).decode('utf-8')
        log_message('INFO', f"[{request_id}] Combined Response Constructed...")
        
        gc.collect()        
        return Response(combined_response_json, content_type='application/json')
    except Exception as e:
        log_message('ERROR', f"Error in handle_technical_query: {str(e)}")
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500
    
@app.route('/query_syllabus', methods=['POST'])
async def handle_syllabus_query():
    data = await request.get_json()
    request_id = data.get('request_id', 'N/A')
    user_query = data.get('query', '')
    token_counter = Counter()  

    index_name = "index-syllabus-final"    
    try:
        start = time.time()

        print(f"[{request_id}] DEBUG: Querying Syllabus Index...")
        response = await query_syllabus_index(user_query, index_name, token_counter, request_id)
        print(f"[{request_id}] DEBUG: Finishing Querying Syllabus Index...")

        end = time.time()
        elapsed_time = end - start
        
        if isinstance(response, Response):
            response_content = await response.get_json()
        else:
            response_content = response

        pdf_link = "https://drive.google.com/file/d/16hX4GAuynjescPwrBgmpUgyEGhgOJp4R/view?usp=drive_link"

        response_content['Reference'] = {
            'pdf_link': pdf_link
        }
        print("Response content:\n", response_content)
        token_usage = await get_token_usage(token_counter)
            
        combined_response = {
            "response": response_content,
            "token_usage": token_usage,
            "elapsed_time": f"{elapsed_time:.2f} seconds"            
        }
                
        logging.debug(f"Response for syllabus query: {combined_response}")
        print(f"[{request_id}] DEBUG: Combined Response Constructed...")

        gc.collect()        
        return jsonify(combined_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug', methods=['GET'])
async def debug():
    return jsonify({"status": "OK", "service": "service_name"}), 200

@app.route('/health', methods=['GET'])
async def health_check():
    return jsonify({"status": "healthy", "service": "service_name"}), 200

@app.before_serving
async def startup():
    app.aiohttp_session = ClientSession(timeout=ClientTimeout(total=10))

@app.after_serving
async def shutdown():
    await app.aiohttp_session.close()   

@app.before_request
async def before_request():
    g.request_data = {}  

@app.after_request
async def after_request(response):
    g.request_data.clear()
    gc.collect()
    
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081, debug=False)
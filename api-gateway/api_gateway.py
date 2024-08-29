from quart import Quart, request, jsonify, send_from_directory
from quart_cors import cors
from quart_rate_limiter import RateLimiter, rate_limit
from datetime import timedelta
import aiohttp
import asyncio
import ujson as json  
import time
from openai import AsyncOpenAI
import os
import logging
from google.cloud import secretmanager
from pinecone import Pinecone
import hashlib, psutil
from google.cloud.logging.handlers import CloudLoggingHandler
from circuitbreaker import circuit
from functools import wraps
from jose import jwt
from decimal import Decimal
from google.cloud import firestore
from datetime import datetime
import firebase_admin
from firebase_admin import auth, credentials
import os, logging, threading
from google.cloud import secretmanager, firestore, logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.auth import default as google_auth_default
from aiohttp import ClientSession, TCPConnector
from contextvars import ContextVar
import traceback

MAX_CONNECTIONS = 50
KEEPALIVE_TIMEOUT = 120

session_var = ContextVar('aiohttp_session', default=None)

def is_local_development():
    return os.environ.get('ENVIRONMENT') == 'local'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

is_cloud_environment = os.environ.get('GOOGLE_CLOUD_PROJECT') is not None

cloud_logging_client = cloud_logging.Client()
cloud_logger = cloud_logging_client.logger('api-gateway-log')

if is_cloud_environment:
    client = cloud_logging.Client()
    handler = CloudLoggingHandler(client)
else:
    handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

TOTAL_CREDITS = 100
REFRESH_DAYS = 90
COST_PER_QUERY = 0.02

app = Quart(__name__)
app = cors(app, allow_origin="*")

def log_message(severity, message):
    """Log messages to both console and Google Cloud Logging."""
    logger.log(getattr(logging, severity), message)
    cloud_logger.log_text(message, severity=severity)

rate_limiter = RateLimiter(app)

google_credentials, project_id = google_auth_default()
if google_credentials is None:
    raise ValueError("Google credentials not found")

def access_secret_version(secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient(credentials=google_credentials)
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")

openai_api_key = access_secret_version("openai-api-key")
pinecone_api_key = access_secret_version("pinecone-api-key")

client = AsyncOpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index_name = "profsnotes-cache"
pinecone_index = pc.Index(index_name)

if not firebase_admin._apps:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {
        'projectId': 'profsnotes',
    })

logger.info(f"Firebase Admin SDK initialized successfully for project: {project_id}")

db = None

def get_firestore_client():
    global db
    if db is None:
        db = firestore.Client()
    return db

JWT_SECRET_KEY = access_secret_version("jwt-secret-key")
JWT_ALGORITHM = "HS256"

async def get_aiohttp_session():
    session = session_var.get()
    if session is None:
        connector = TCPConnector(limit=MAX_CONNECTIONS, keepalive_timeout=KEEPALIVE_TIMEOUT)
        session = ClientSession(connector=connector)
        session_var.set(session)
    return session

@app.before_serving
async def startup():
    logger.info("Running startup tasks...")
    try:
        pinecone_index.describe_index_stats()
        logger.info("Pinecone index check passed")
        
        await get_aiohttp_session()
        logger.info("Aiohttp session initialized")
    except Exception as e:
        logger.error(f"Startup tasks failed: {str(e)}")
    logger.info("Startup tasks completed")

@app.after_serving
async def shutdown():
    logger.info("Running shutdown tasks...")
    session = session_var.get()
    if session:
        await session.close()
        logger.info("Aiohttp session closed")
    logger.info("Shutdown tasks completed")

async def initialize_services():
    services = [
        ('notes', 'https://notes-service-dot-profsnotes.uk.r.appspot.com/health'),
        # ('notes', 'http://localhost:8081/health'),

        ('wikipedia', 'https://wikipedia-service-dot-profsnotes.uk.r.appspot.com/health'),
        # ('notes', 'http://localhost:8083/health'),

        ('youtube', 'https://youtube-service-dot-profsnotes.uk.r.appspot.com/health')
        # ('notes', 'http://localhost:8082/health'),

    ]
    
    results = await asyncio.gather(*[warmup_service(name, url) for name, url in services])
    
    for service, result in zip(services, results):
        if 'error' in result:
            logger.error(f"Failed to warm up {service[0]}: {result['error']}")
        else:
            logger.info(f"Successfully warmed up {service[0]}")

    logger.info("Service warm-up completed")

def background_init():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(initialize_services())

def create_or_initialize_user(uid):
    db = get_firestore_client()
    user_ref = db.collection('users').document(uid)
    user_doc = user_ref.get()

    if not user_doc.exists:
        user_ref.set({
            'credits': TOTAL_CREDITS,
            'last_refresh': datetime.utcnow()
        })
    return user_ref

def check_and_refresh_credits(user_ref):
    user_data = user_ref.get().to_dict()
    last_refresh = user_data.get('last_refresh').replace(tzinfo=None)
    current_time = datetime.utcnow()

    if current_time - last_refresh >= timedelta(days=REFRESH_DAYS):
        user_ref.update({
            'credits': TOTAL_CREDITS,
            'last_refresh': current_time
        })
        return TOTAL_CREDITS
    return user_data.get('credits')

def create_jwt_token(uid: str):
    payload = {
        "sub": uid,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {str(e)}")
    
def error_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            return {"error": f"An error occurred in {func.__name__}"}
    return wrapper

@error_handler
async def generate_embedding(text):
    response = await client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    if len(embedding) != 1536:
        raise ValueError(f"Unexpected embedding length: {len(embedding)}")
    return embedding

@error_handler
async def get_cached_response(query_embedding, similarity_threshold=0.87):
    results = pinecone_index.query(vector=query_embedding, top_k=1, include_metadata=True)
    if results['matches'] and results['matches'][0]['score'] >= similarity_threshold:
        cache_key = f"aggregated_response:{results['matches'][0]['id']}"
        db = get_firestore_client()
        doc_ref = db.collection('query_cache').document(cache_key)
        doc = doc_ref.get()
        if doc.exists:
            cached_data = doc.to_dict()
            return cached_data['core_response'] 
    return None

@error_handler
async def cache_response(query, embedding, response):
    pinecone_index.upsert(vectors=[(query, embedding, {'query': query})])
    cache_key = f"aggregated_response:{query}"
    db = get_firestore_client()
    doc_ref = db.collection('query_cache').document(cache_key)
    
    core_response = {
        'notes_response': {k: v for k, v in response['notes_response'].items() if k != 'elapsed_time' and k != 'token_usage'},
        'wikipedia_response': {k: v for k, v in response['wikipedia_response'].items() if k != 'elapsed_time' and k != 'token_usage'},
        'youtube_response': {k: v for k, v in response['youtube_response'].items() if k != 'elapsed_time' and k != 'token_usage'}
    }
    
    doc_ref.set({
        'core_response': core_response
    })

@circuit(failure_threshold=5, recovery_timeout=20)
@rate_limit(30, timedelta(seconds=60))  
async def get_query_direction(user_query):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Respond in JSON format."},
            {"role": "user", "content": f"""
            Your tasks are the following:
            0. Read the user's query: {user_query} 
            1. Identify the direction of the user's query. Inspect whether the query is:
               - A technical inquiry on Fluid Dynamics/Mechanics
               - A syllabus-based query about the course itself (labs, syllabus, outline, plan, TAs (teaching assistants), textbooks, tutorials, this sort of information etc.)
               - An unrelated or inappropriate query
            2. Populate the following JSON with one of these values: "technical", "syllabus", or "unrelated":
            {{
                "Direction": "Value"
            }}
            """}
        ],
        temperature=0,
        max_tokens=15,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    print("Direction>>>>>>:\n",result['Direction'])
    return result['Direction']

def parse_token_usage(response_data):
    total_token_usage = {
        "gpt_3_5_turbo_0125_token_count": 0,
        "gpt_4o_mini_2024_07_18_token_count": 0,
        "gpt_4o_2024_05_13_token_count": 0,
        "gpt_4_turbo_2024_04_09_token_count": 0,
        "gpt_4_0613_token_count": 0,
        "text_embedding_3_small_token_count": 0
    }
    
    model_name_mapping = {
        "gpt_4o_mini_2024_07_18": "gpt_4o_mini_2024_07_18_token_count",        
        "gpt_3_5_turbo_0125": "gpt_3_5_turbo_0125_token_count",
        "gpt_4o_2024_05_13": "gpt_4o_2024_05_13_token_count",
        "gpt_4_turbo_2024_04_09": "gpt_4_turbo_2024_04_09_token_count",
        "gpt_4_0613": "gpt_4_0613_token_count",
        "text_embedding_3_small": "text_embedding_3_small_token_count"
    }
    
    def get_model_key(model):
        for key in model_name_mapping:
            if key in model:
                return model_name_mapping[key]
        return None

    def process_token_usage(usage_data):
        if isinstance(usage_data, dict):
            for model, count in usage_data.items():
                model_key = get_model_key(model)
                if model_key:
                    total_token_usage[model_key] += count
                else:
                    logger.warning(f"Unknown model in token usage: {model}")

    if isinstance(response_data, dict):
        for service_name, service_response in response_data.items():
            if isinstance(service_response, dict):
                if 'token_usage' in service_response:
                    process_token_usage(service_response['token_usage'])
                elif 'results' in service_response and isinstance(service_response['results'], dict):
                    if 'token_usage' in service_response['results']:
                        process_token_usage(service_response['results']['token_usage'])
                else:
                    logger.warning(f"No token usage found in {service_name} response")
            else:
                logger.warning(f"Unexpected response structure for {service_name}")
    
    return total_token_usage

def calculate_cost(total_usage):
    cost_per_million = {
        "gpt_4o_mini_2024_07_18": 0.600,
        "gpt_3_5_turbo_0125": 1.50,
        "gpt_4o_2024_05_13": 15.00,
        "gpt_4_turbo_2024_04_09": 30.00,
        "gpt_4_0613": 60.00,
        "text_embedding_3_small": 0.02,        
    }
    
    total_cost = 0
    for model, count in total_usage.items():
        model_name = model.replace("_token_count", "")
        if model_name in cost_per_million:
            cost = (count / 1_000_000) * cost_per_million[model_name] * 1.37
            total_cost += cost
    print("DEBUG:>>>>>>>>> Total cost:\n",total_cost)
    
    credit_cost = (total_cost / COST_PER_QUERY) * (TOTAL_CREDITS / 100)
    if all(count == 0 for count in total_usage.values()):
        logger.warning("All token counts are zero. Setting minimum cost.")
        return 1 
    return max(1, round(credit_cost))

async def handle_unrelated_query(query):
    error_message = f"I apologize, but this question: '{query}', appears to be inappropriate or unrelated to Fluid Dynamics/Mechanics or the course content. Please try again with a relevant question."
    
    response_data = {
        'notes_response': {
            'response': {
                'Text': error_message
            }
        },
        'wikipedia_response': {
            'results': {}
        },
        'youtube_response': {
            'results': []
        },
        'credits_used': 0,  
        'remaining_credits': None,  
        'is_cached': False
    }

    return response_data

async def fetch_service_response(service_name, url, payload, request_id):
    start_time = time.time()
    try:
        log_message('INFO', f"[{request_id}] Attempting to call {service_name} service at {url}")
        # log_message('DEBUG', f"[{request_id}] Payload for {service_name}: {payload}")
        
        session = await get_aiohttp_session()
        payload['request_id'] = request_id 
        headers = {
            'Content-Type': 'application/json',
            'X-Request-ID': request_id
        }
        
        async with session.post(url, json=payload, headers=headers, timeout=60) as response:
            log_message('INFO', f"[{request_id}] Received response from {service_name} service with status {response.status}")
            
            response_text = await response.text()
            # log_message('DEBUG', f"[{request_id}] Raw response from {service_name}: {response_text[:10]}...")  # Log first 500 characters
            
            response.raise_for_status()
            result = await response.json()
            
            duration = time.time() - start_time
            log_message('INFO', f"[{request_id}] {service_name} service call completed in {duration:.2f} seconds")
            return {"success": True, "data": result}
    except aiohttp.ClientResponseError as e:
        duration = time.time() - start_time
        log_message('ERROR', f"[{request_id}] Client error when calling {service_name} service after {duration:.2f} seconds: {e.status}, message='{e.message}', url={e.request_info.url}")
        log_message('ERROR', f"[{request_id}] Response headers: {e.headers}")
        return {"success": False, "error": f"Failed to fetch response from {service_name}: {str(e)}"}
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        log_message('ERROR', f"[{request_id}] Timeout error when calling {service_name} service after {duration:.2f} seconds")
        return {"success": False, "error": f"Service {service_name} timed out"}
    except Exception as e:
        duration = time.time() - start_time
        log_message('ERROR', f"[{request_id}] Unexpected error when fetching response from {service_name} after {duration:.2f} seconds")
        log_message('ERROR', f"[{request_id}] Error details: {str(e)}")
        log_message('ERROR', f"[{request_id}] Traceback: {traceback.format_exc()}")
        return {"success": False, "error": f"Unexpected error from {service_name}: {str(e)}"}

@app.route('/query', methods=['POST'])
@error_handler
@rate_limit(30, timedelta(seconds=60))  
async def handle_query():
    request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    logger.info(f"[{request_id}] Handling new query request")
    
    initial_token_usage = {
        "gpt_3_5_turbo_0125_token_count": 0,
        "gpt_4_0125_preview_token_count": 0,
        "gpt_4_0613_token_count": 0,
        "text_embedding_3_small_token_count": 0
    }
    total_token_usage = initial_token_usage.copy()
    initial_cost = 0 
    overall_start_time = time.time()
    
    data = await request.json
    if data is None:
        logger.error(f"[{request_id}] Request body is empty or not valid JSON")
        return jsonify({'error': 'Invalid request body'}), 400
        
    query = data.get('query')
    logger.info(f"[{request_id}] Handling new query: `{query}` request")

    token = data.get('token')    

    if not query:
        logger.error(f"[{request_id}] Query is missing in the request")
        return jsonify({'error': 'Query is required'}), 400

    if not token:
        logger.error(f"[{request_id}] Token is missing in the request")
        return jsonify({'error': 'Token is required'}), 400

    if is_local_development():
        if token == "local-test-token":
            uid = 'local-test-user'
            logger.info(f"[{request_id}] Local development mode: using test user ID")
        else:
            logger.error(f"[{request_id}] Invalid token for local testing")
            return jsonify({'error': 'Invalid token for local testing'}), 401
    else:
        uid = verify_jwt_token(token)
        if not uid:
            logger.error(f"[{request_id}] Invalid or expired token")
            return jsonify({'error': 'Invalid or expired token'}), 401

    logger.info(f"[{request_id}] Starting query processing for user {uid}")
    
    embedding_start_time = time.time()
    normalized_query, query_embedding = await normalize_and_embed_query(query)
    logger.info(f"[{request_id}] Normalized query: {normalized_query[:10]}")
    
    embedding_time = time.time() - embedding_start_time
    logger.info(f"[{request_id}] Query embedding generated in {embedding_time:.2f} seconds")
    
    if not query_embedding:
        logger.error(f"[{request_id}] Failed to generate embedding")
        return jsonify({'error': 'Failed to generate embedding'}), 400
    
    cache_check_start_time = time.time()
    cached_response = await get_cached_response(query_embedding)
    cache_check_time = time.time() - cache_check_start_time
    logger.info(f"[{request_id}] Cache check completed in {cache_check_time:.2f} seconds")
    
    if cached_response:
        logger.info(f"[{request_id}] Cache hit for query: {normalized_query[:10]}...")
        
        elapsed_time = time.time() - overall_start_time
        cached_response['elapsed_time'] = f"{elapsed_time:.2f} seconds"
        cached_response['token_usage'] = parse_token_usage(cached_response)

        db = get_firestore_client()
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()
        current_credits = user_doc.to_dict().get('credits', 0)

        cached_response['remaining_credits'] = current_credits
        cached_response['is_cached'] = True

        overall_time = time.time() - overall_start_time
        logger.info(f"[{request_id}] Total processing time (cache hit): {overall_time:.2f} seconds")
        return jsonify(cached_response)
    
    logger.info(f"[{request_id}] Cache miss, processing query")
    payload = {'query': normalized_query, 'embedding': query_embedding, 'request_id': request_id}
    
    direction = await get_query_direction(normalized_query)
    logger.info(f"[{request_id}] Query direction: {direction}")
    service_call_start_time = time.time()

    if direction == 'syllabus':
        logger.info(f"[{request_id}] Fetching syllabus response")
        notes_response = await fetch_service_response('notes', 'https://notes-service-dot-profsnotes.uk.r.appspot.com/query_syllabus', payload, request_id)
        # notes_response = await fetch_service_response('notes', 'http://localhost:8081/query_syllabus', payload, request_id)
        
        if not notes_response['success']:
            logger.error(f"[{request_id}] Failed to fetch syllabus response")
            return handle_network_error()

        logger.info(f"[{request_id}] Successfully fetched syllabus response")
        response_data = {
            'notes_response': notes_response['data'],
            'wikipedia_response': {},
            'youtube_response': {}
        }

    elif direction == 'technical':
        logger.info(f"[{request_id}] Fetching technical response")
        responses = await asyncio.gather(
            fetch_service_response('notes', 'https://notes-service-dot-profsnotes.uk.r.appspot.com/query_technical', payload, request_id),
            # fetch_service_response('notes', 'http://localhost:8081/query_technical', payload, request_id),

            fetch_service_response('wikipedia', 'https://wikipedia-service-dot-profsnotes.uk.r.appspot.com/query', payload, request_id),
            # fetch_service_response('wikipedia', 'http://localhost:8082/query', payload, request_id),

            fetch_service_response('youtube', 'https://youtube-service-dot-profsnotes.uk.r.appspot.com/query', payload, request_id)
            # fetch_service_response('youtube', 'http://localhost:8083/query', payload, request_id)

        )

        if not any(response['success'] for response in responses):
            logger.error(f"[{request_id}] All services failed for technical query")
            return handle_network_error()

        logger.info(f"[{request_id}] Services response status: Notes: {responses[0]['success']}, Wikipedia: {responses[1]['success']}, YouTube: {responses[2]['success']}")
        
        notes_response = responses[0]['data'] if responses[0]['success'] else {}
        
        # Check if can_answer is False in the notes_response
        if not notes_response.get('response', {}).get('can_answer', True):
            logger.info(f"[{request_id}] Notes service cannot answer. Returning empty Wikipedia and YouTube responses.")
            response_data = {
                'notes_response': notes_response,
                'wikipedia_response': {},
                'youtube_response': {}
            }
        else:
            response_data = {
                'notes_response': notes_response,
                'wikipedia_response': responses[1]['data'] if responses[1]['success'] else {},
                'youtube_response': responses[2]['data'] if responses[2]['success'] else {}
            }

        updated_token_usage = total_token_usage.copy()
        for service_name, service_response in response_data.items():
            if 'token_usage' in service_response:
                service_token_usage = service_response['token_usage']
                for model, count in service_token_usage.items():
                    model_key = f"{model}_token_count"
                    if model_key in updated_token_usage:
                        updated_token_usage[model_key] += count
                    else:
                        logger.warning(f"[{request_id}] Unexpected model in token usage: {model}")

        total_token_usage = updated_token_usage

    elif direction == 'unrelated':
        logger.info(f"[{request_id}] Handling unrelated or inappropriate query")
        response_data = await handle_unrelated_query(normalized_query)
    else:
        logger.error(f"[{request_id}] Unknown query direction: {direction}")
        return jsonify({'error': 'Unable to process query'}), 400

    try:
        total_token_usage = parse_token_usage(response_data)
        logger.info(f"[{request_id}] Total token usage after all services: {total_token_usage}")

    except Exception as e:
        logger.error(f"[{request_id}] Error parsing token usage: {str(e)}")
        total_token_usage = initial_token_usage

    try:
        cost = calculate_cost(total_token_usage)
        logger.info(f"[{request_id}] Calculated cost: {cost}")
    except Exception as e:
        logger.error(f"[{request_id}] Error calculating cost: {str(e)}")
        cost = initial_cost

    if direction != 'unrelated':
        if all('error' not in response for response in response_data.values()):
            logger.info(f"[{request_id}] Caching response and updating suggestions")
            await cache_response(normalized_query, query_embedding, response_data)
            await update_suggestions_cache(normalized_query)
        else:
            logger.warning(f"[{request_id}] Not caching response due to errors in service responses")

    success, remaining_credits = await deduct_user_credits(uid, cost)
    if not success:
        logger.error(f"[{request_id}] Insufficient credits for user {uid}")
        return jsonify({'error': 'Insufficient credits'}), 403
    
    logger.info(f"[{request_id}] Credits deducted successfully. Remaining credits: {remaining_credits}")
    
    response_data['credits_used'] = cost
    response_data['remaining_credits'] = remaining_credits
    response_data['is_cached'] = False

    overall_time = time.time() - overall_start_time
    logger.info(f"[{request_id}] Total processing time: {overall_time:.2f} seconds")
    return jsonify(response_data)

async def deduct_user_credits(uid, cost):
    if is_local_development():
        logger.info(f"Local development mode: simulating credit deduction for user {uid}")
        return True, 100  
        
    db = get_firestore_client()
    user_ref = db.collection('users').document(uid)
    
    @firestore.transactional
    def update_credits(transaction):
        user_doc = user_ref.get(transaction=transaction)
        current_credits = user_doc.to_dict().get('credits', 0)
        
        new_credits = Decimal(current_credits) - Decimal(cost)
        
        if new_credits < 0:
            return False, current_credits
        
        transaction.update(user_ref, {'credits': float(new_credits)})
        return True, new_credits
    
    transaction = db.transaction()
    success, remaining_credits = update_credits(transaction)
    return success, float(remaining_credits)

def handle_network_error():
    return jsonify({
        'error': 'Service unavailable',
        'message': 'Network error. Please refresh the page and try again.',
        'credits_used': 0,
        'remaining_credits': None,
        'is_cached': False
    }), 503

async def normalize_and_embed_query(query):
    normalized_query = query.lower().strip()
    embedding = await generate_embedding(normalized_query)
    return normalized_query, embedding

async def update_suggestions_cache(query):
    db = get_firestore_client()
    doc_ref = db.collection('query_suggestions').document('suggestions')
    doc = doc_ref.get()
    if doc.exists:
        all_queries = doc.to_dict().get('queries', [])
    else:
        all_queries = []
    if query not in all_queries:
        all_queries.append(query)
        all_queries = all_queries[-100:]  
        doc_ref.set({'queries': all_queries})


@app.route('/suggestions', methods=['POST'])
@error_handler
async def get_suggestions():
    data = await request.json
    query = data.get('query', '').lower()
    
    if not query:
        return jsonify([])

    db = get_firestore_client()
    doc_ref = db.collection('query_suggestions').document('suggestions')
    doc = doc_ref.get()
    if doc.exists:
        all_queries = doc.to_dict().get('queries', [])
        suggestions = [q for q in all_queries if q.lower().startswith(query)][:5]
    else:
        suggestions = []
    
    return jsonify(suggestions)

@app.route('/flush_cache', methods=['POST'])
@error_handler
async def flush_cache():
    try:
        db = get_firestore_client()
        
        docs = db.collection('query_cache').limit(500).stream()
        deleted_queries = 0
        for doc in docs:
            doc.reference.delete()
            deleted_queries += 1
        
        suggestions_doc = db.collection('query_suggestions').document('suggestions')
        suggestions_doc.set({'queries': []})
        
        pinecone_index = pc.Index(index_name)
        pinecone_index.delete(delete_all=True)
        
        logger.info(f"Firestore caches flushed successfully. Deleted {deleted_queries} query documents and cleared suggestions.")
        logger.info("Pinecone index flushed successfully")

        return jsonify({'message': 'All caches and Pinecone index flushed successfully'}), 200
    except Exception as e:
        logger.exception(f"Error in flush_cache: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/user_credits', methods=['GET'])
@error_handler
async def get_user_credits():
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        logger.error("No Bearer token in Authorization header")
        return jsonify({'error': 'Bearer token is required'}), 401

    token = auth_header.split('Bearer ')[-1]
    uid = verify_jwt_token(token)
    if not uid:
        logger.error(f"Invalid token: {token[:10]}...")
        return jsonify({'error': 'Invalid or expired token'}), 401

    logger.info(f"Fetching credits for user: {uid}")
    db = get_firestore_client()
    user_ref = db.collection('users').document(uid)
    user_doc = user_ref.get()

    if not user_doc.exists:
        logger.error(f"User document not found for UID: {uid}")
        return jsonify({'error': 'User not found'}), 404

    credits = user_doc.to_dict().get('credits', 0)
    logger.info(f"Credits for user {uid}: {credits}")
    return jsonify({'credits': credits})

@app.route('/refresh_credits', methods=['POST'])
@error_handler
async def refresh_credits():
    data = await request.json
    uid = data.get('uid')
    if not uid:
        return jsonify({'error': 'UID is required'}), 400

    db = get_firestore_client()
    user_ref = db.collection('users').document(uid)
    user_doc = user_ref.get()

    if not user_doc.exists:
        return jsonify({'error': 'User not found'}), 404

    current_credits = check_and_refresh_credits(user_ref)
    return jsonify({'message': 'Credits checked/refreshed successfully', 'current_credits': current_credits}), 200

@app.route('/login', methods=['POST'])
@error_handler
async def login():
    data = await request.json
    id_token = data.get('idToken')
    if not id_token:
        logger.error("No ID token provided in request")
        return jsonify({'error': 'ID token is required'}), 400
    
    try:
        logger.debug(f"Attempting to verify token: {id_token[:10]}...")  
        decoded_token = auth.verify_id_token(id_token)
        logger.info(f"Token verified successfully. UID: {decoded_token['uid']}")
        
        uid = decoded_token['uid']
        token = create_jwt_token(uid)
        
        db = firestore.Client()
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()

        if not user_doc.exists:
            user_ref.set({
                'credits': TOTAL_CREDITS,
                'last_refresh': datetime.utcnow(),
                'last_query_cached': False
            })
            credits = TOTAL_CREDITS
            last_query_cached = False
        else:
            user_data = user_doc.to_dict()
            credits = user_data.get('credits', TOTAL_CREDITS)
            last_query_cached = user_data.get('last_query_cached', False)
    
        threading.Thread(target=background_init).start()
        
        logger.info(f"JWT created for user {uid}")
        return jsonify({
            'token': token,
            'credits': credits,
            'last_query_cached': last_query_cached
        })
    except auth.InvalidIdTokenError as e:
        logger.error(f"Invalid ID token: {str(e)}")
        return jsonify({'error': 'Invalid ID token'}), 401
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Login failed'}), 500

async def warmup_service(service_name, url):
    try:
        session = await get_aiohttp_session()
        async with session.get(url, timeout=10) as response:
            logger.info(f"Warm-up response from {service_name} service with status {response.status}")
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Connection error during warm-up of {service_name} service: {str(e)}")
        return {"error": f"Failed to connect to {service_name}: {str(e)}"}
    except aiohttp.ClientResponseError as e:
        logger.error(f"HTTP error during warm-up of {service_name} service: {e.status}, {e.message}")
        return {"error": f"HTTP error from {service_name}: {e.status}, {e.message}"}
    except asyncio.TimeoutError:
        logger.error(f"Timeout during warm-up of {service_name} service")
        return {"error": f"Timeout while warming up {service_name}"}
    except Exception as e:
        logger.error(f"Unexpected error during warm-up of {service_name} service: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Failed to warm up {service_name}: {str(e)}"}

@app.route('/_ah/warmup')
async def warmup():
    logger.info("Warm-up request received")
    try:
        await asyncio.gather(
            fetch_service_response('notes', 'https://notes-service-dot-profsnotes.uk.r.appspot.com/debug', {}),
            fetch_service_response('wikipedia', 'https://wikipedia-service-dot-profsnotes.uk.r.appspot.com/debug', {}),
            fetch_service_response('youtube', 'https://youtube-service-dot-profsnotes.uk.r.appspot.com/debug', {})
        )
        return 'OK', 200
    except Exception as e:
        logger.error(f"Warm-up request failed: {str(e)}")
        return 'Error', 500

@app.route('/')
async def serve_index():
    return await send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
async def serve_static(path):
    return await send_from_directory('static', path)

@app.route('/health', methods=['GET'])
async def health_check():
    try:
        # Firestore check
        db = get_firestore_client()
        db.collection('health_check').document('test').get()
        logger.info("Firestore check passed")

        # Pinecone check
        pinecone_index.describe_index_stats()
        logger.info("Pinecone index check passed")

        # OpenAI check
        await client.models.list()
        logger.info("OpenAI client check passed")

        # Aiohttp session check
        session = await get_aiohttp_session()
        if session.closed:
            raise Exception("Aiohttp session is closed")
        
        connector = session.connector
        
        # Safely get connection information
        try:
            num_connections = sum(len(conns) for conns in connector._conns.values())
        except AttributeError:
            num_connections = "Unable to determine"
        
        try:
            connection_limit = connector.limit
        except AttributeError:
            connection_limit = "Unable to determine"

        logger.info(f"Aiohttp session check passed. Active connections: {num_connections}, Limit: {connection_limit}")

        return jsonify({
            'status': 'healthy',
            'firestore': 'OK',
            'pinecone': 'OK',
            'openai': 'OK',
            'aiohttp_session': {
                'status': 'OK',
                'num_connections': num_connections,
                'connection_limit': connection_limit
            }
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'error_type': type(e).__name__,
            'error_details': traceback.format_exc()
        }), 500
    
@app.errorhandler(429)
async def ratelimit_handler(e):
    return jsonify(error="ratelimit exceeded", message=str(e.description)), 429

@app.errorhandler(500)
async def internal_server_error(e):
    logger.error(f"Internal Server Error: {str(e)}")
    return jsonify(error="Internal Server Error", message="An unexpected error occurred"), 500

@app.before_request
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    app.logger.info(f"Memory usage before request: {mem:.2f} MB")

@app.after_request
def log_memory_usage_after(response):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    app.logger.info(f"Memory usage after request: {mem:.2f} MB")
    return response

if __name__ == '__main__':
    os.environ['ENVIRONMENT'] = 'local'  # Set the environment to local
    logger.info("Starting the application in local development mode")
    app.run(debug=True, host='0.0.0.0', port=8080)
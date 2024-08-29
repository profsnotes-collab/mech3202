import re
import ujson as json
import orjson
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled
from openai import AsyncOpenAI
from quart import Quart, request, jsonify
from quart_cors import cors
import time
import logging
from google.cloud import secretmanager
from google.auth import default
from collections import Counter
from googleapiclient.errors import HttpError
from requests.auth import HTTPProxyAuth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Quart(__name__)
app = cors(app, allow_origin="*")

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
if credentials is not None:
    print(f"Authenticated account: {credentials.service_account_email}")
else:
    print("No credentials found.")
    raise ValueError("Credentials not found")

def access_secret_version(project_id, secret_id, version_id):
    # logger.info(f"Accessing secret: {secret_id}")
    client = secretmanager.SecretManagerServiceClient(credentials=credentials)
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    # logger.info(f"Accessing secret version: {name}")
    response = client.access_secret_version(name=name)
    # logger.info(f"Secret accessed successfully: {secret_id}")
    return response.payload.data.decode("UTF-8")

openai_secret_id = "openai-api-key"
openai_secret_version = "latest"
secret_youtube_id = "youtube-api-key-v2"
youtube_secret_version = "latest"

openai_api_key = access_secret_version(project_id, openai_secret_id, openai_secret_version)

# if openai_api_key:
#     print(f"OpenAI API Key successfully retrieved: {openai_api_key[:4]} ...") 
# else:
#     # print("OpenAI API Key not found. Please check the environment variable.")
#     raise ValueError("OpenAI API Key not found")

client = AsyncOpenAI(api_key=openai_api_key)

# print(f"SECRET_YOUTUBE_ID: {secret_youtube_id}")

youtube_api_key = access_secret_version(project_id, secret_youtube_id, youtube_secret_version)

# if youtube_api_key:
#     print(f"YouTube API Key successfully retrieved: {youtube_api_key[:4]} ...")
# else:
#     print("YouTube API Key not found. Please check the environment variable.")
#     raise ValueError("YouTube API Key not found")

def sanitize_filename(filename):
    return re.sub(r'[\/:*?"<>|]', '_', filename)

async def get_topic_from_query(query, token_counter):
    logger.info(f"Extracting topic from query: {query}")
    response = await client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specializing in fluid dynamics and mechanics."},
            {"role": "user", "content": f"""
                Your task is to extract the main topic related to fluid dynamics/mechanics from the following query:
                Query: `{query}`
                --------------
                1. The topic should be 1-3 words long, focusing on the core concept in fluid dynamics/mechanics.
                2. Return your response as a JSON object with a single key "Text" whose value is the extracted topic.
            """}
        ],
        temperature=0,
        max_tokens=50,
        response_format={"type": "json_object"}
    )

    logger.debug(f"Updating token count: model={response.model}, tokens={response.usage.total_tokens}")
    await update_token_count(token_counter, response.model, response.usage.total_tokens)

    content = response.choices[0].message.content
    topic_json = json.loads(content)
    topic = topic_json.get("Text", "").strip()
    logger.info(f"Extracted topic: {topic}")    
    return topic

async def create_timestamp_link(video_id, start):
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"

async def refined_youtube_search(query, max_results=3, request_id='N/A'):
    logger.info(f"[{request_id}] Starting YouTube search for query: {query}")
    try:
        search_query = f"{query} in fluid dynamics"
        params = {
            'part': 'snippet,id',
            'q': search_query,
            'type': 'video',
            'relevanceLanguage': 'en',
            'maxResults': max_results,
            'videoDuration': 'medium',
            'key': youtube_api_key
        }
        async with app.aiohttp_session.get('https://www.googleapis.com/youtube/v3/search', params=params) as resp:
            response = await resp.json()
        logger.info(f"[{request_id}] YouTube API request successful")
        return response
    except aiohttp.ClientError as e:
        logger.error(f"[{request_id}] aiohttp ClientError in YouTube API request: {str(e)}")
        raise
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Timeout error in YouTube API request")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in YouTube API request: {str(e)}")
        raise

async def filter_videos(videos, query, token_counter, request_id='N/A'):
    logger.info(f"[{request_id}] Filtering videos for query: {query}")
    filtered_videos = []
    for video in videos:
        title = video['snippet']['title'].lower()
        description = video['snippet']['description'].lower()
        channel_name = video['snippet']['channelTitle']
        print("Channel name:\n",channel_name)
        keywords = query.lower().split()
        if any(keyword in title or keyword in description for keyword in keywords):
            relevance_prompt = f"""
            Determine if this video is relevant to both fluid dynamics/mechanics AND the user's query.
            User's query: '{query}'.
            -------------------------- 
            For the following video:
            Title: `{title}`
            Description: `{description}`
            --------------------------
            Remember, the video MUST BE relevant to fluid dynamics/mechanics.
            Answer ONLY with either 'Relevant' or 'Not' for irrelevant videos only.
            """
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": relevance_prompt}
                ],
                temperature=0,
                max_tokens=10
            )

            await update_token_count(token_counter, response.model, response.usage.total_tokens)

            if "Relevant" in response.choices[0].message.content or "relevant" in response.choices[0].message.content:
                filtered_videos.append(video)
                logger.debug(f"[{request_id}] Video {video['id']['videoId']} considered relevant")
    
    logger.info(f"[{request_id}] Filtered {len(filtered_videos)} relevant videos")
    return filtered_videos[:3]

async def analyze_transcript(transcript_text, query, token_counter, video_title, channel_name, request_id='N/A'):
    logger.info(f"[{request_id}] Analyzing transcript for query: {query}")
    # print(f"Transcript Text: f{transcript_text}")
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""
                Your tasks include:
                0. Analyze the following transcript and extract the most relevant information related to answering the user's question: '{query}'. 
                1. Populate the following JSON using the attached video transcript, also using the video title for the title field: `{video_title}` and the channel name for the description field: `{channel_name}`.
                2. Under the description 
                Provide the timestamp where this information can be found in JSON format. The JSON schema should include only the most relevant piece of information with the following structure: {{
                    "relevant_information":
                        {{
                            "title": "Video Title: insert here",
                            "description": "Video Creator: insert here",
                            "text": "Timestamped Video Explains: A very concise explanation, one sentence explanation of what the particular timestamped portion explains.",
                            "start_time": "number",
                            "end_time": "number",
                        }}
                }}: {transcript_text}"""}
            ],
            temperature=0.5,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        await update_token_count(token_counter, response.model, response.usage.total_tokens)
        
        content = response.choices[0].message.content
        logger.debug(f"[{request_id}] Transcript analysis result: {content}")
        
        try:
            return orjson.loads(content)
        except orjson.JSONDecodeError as e:
            logger.error(f"[{request_id}] Error parsing JSON from OpenAI response: {str(e)}")
            logger.debug(f"[{request_id}] Raw content: {content}")
            return None
        
    except Exception as e:
        logger.error(f"[{request_id}] Error analyzing transcript: {str(e)}", exc_info=True)
        return None

def get_smartproxy_credentials():
    return {
        'username': 'spjdrdpt6t',
        'password': 'd+BaKv57c1A3gZwdlf',
        'host': 'gate.smartproxy.com',
        'port': '7000'
    }

def get_proxy_url():
    smartproxy = get_smartproxy_credentials()
    return f'http://{smartproxy["username"]}:{smartproxy["password"]}@{smartproxy["host"]}:{smartproxy["port"]}'

async def fetch_transcripts_and_analyze(videos, query, token_counter, request_id='N/A'):
    logger.info(f"[{request_id}] Fetching and analyzing transcripts for {len(videos)} videos")
    
    async def process_video(item):
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        channel_name = item['snippet']['channelTitle'] 
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        logger.info(f"[{request_id}] Processing video: {video_id} - {video_title}")

        try:
            proxy_url = get_proxy_url()
            logger.info(f"[{request_id}] Using proxy: {proxy_url}")
            
            proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            
            transcript_list = await asyncio.to_thread(
                YouTubeTranscriptApi.get_transcript, 
                video_id, 
                languages=['en'], 
                proxies=proxies
            )

            if not transcript_list:
                logger.warning(f"[{request_id}] Empty transcript for video {video_id}")
                return None, None
            
            transcript_text = "\n".join([f"{t['start']}s - {t['start'] + t['duration']}s: {t['text']}" for t in transcript_list])
            logger.debug(f"[{request_id}] Transcript fetched for video {video_id}")

            analysis = await analyze_transcript(transcript_text, query, token_counter, video_title, channel_name, request_id)
            if not analysis:
                logger.warning(f"[{request_id}] Analysis failed for video {video_id}")
                return None, None
            
            logger.info(f"[{request_id}] Analysis completed for video {video_id}")
            return analysis, item

        except (NoTranscriptFound, TranscriptsDisabled) as e:
            logger.error(f"[{request_id}] Transcript not available for video {video_id}: {str(e)}")
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error for video {video_id}: {str(e)}", exc_info=True)
        
        return None, None

    results = await asyncio.gather(*[process_video(item) for item in videos])
    valid_results = [(analysis, item) for analysis, item in results if analysis is not None]
    logger.info(f"[{request_id}] Processed and analyzed {len(valid_results)} video transcripts successfully out of {len(videos)} total videos")
    return valid_results

@app.route('/query', methods=['POST'])
async def search_videos():   
    data = await request.get_json()

    request_id = data.get('request_id', 'N/A')
    logger.info(f"[{request_id}] Received query request")

    token_counter = Counter()

    query = data.get("query")    
    start_time = time.time()
    if not query:
        logger.error(f"[{request_id}] Query is missing")
        return jsonify({"error": "Query is required"}), 400

    logger.info(f"[{request_id}] Processing query: {query[:50]}...")

    try:
        topic = await get_topic_from_query(query, token_counter)
        logger.info(f"[{request_id}] Extracted topic: {topic}")

        response = await refined_youtube_search(topic, max_results=3, request_id=request_id)

        if 'items' in response and response['items']:
            videos = await filter_videos(response['items'], query, token_counter, request_id)
            analyses_and_items = await fetch_transcripts_and_analyze(videos, query, token_counter, request_id)

            results = []
            for analysis, item in analyses_and_items:
                video_id = item['id']['videoId']
                if analysis and 'relevant_information' in analysis:
                    info = analysis['relevant_information']
                    timestamp_link = await create_timestamp_link(video_id, info["start_time"])
                    results.append({
                        "video_url": f"https://www.youtube.com/watch?v={video_id}",
                        "title": info['title'],
                        "description": info['description'],
                        "text": info['text'],
                        "timestamp": timestamp_link
                    })
                    logger.debug(f"[{request_id}] Added result for video {video_id}")
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            token_usage = await get_token_usage(token_counter)

            logger.info(f"[{request_id}] Query processed successfully. Elapsed time: {elapsed_time:.2f}s")
            return jsonify({
                "results": results,
                "token_usage": token_usage,
                "elapsed_time": elapsed_time
            })
        else:
            logger.warning(f"[{request_id}] No videos found for the query")
            return jsonify({"message": "No videos found for the query."})

    except Exception as e:
        logger.error(f"[{request_id}] Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500

@app.before_serving
async def startup():
    app.aiohttp_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

@app.after_serving
async def shutdown():
    await app.aiohttp_session.close()

@app.route('/debug', methods=['GET'])
async def debug():
    return jsonify({"status": "OK", "service": "service_name"}), 200

@app.route('/health', methods=['GET'])
async def health_check():
    return jsonify({"status": "healthy", "service": "service_name"}), 200  

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8082)

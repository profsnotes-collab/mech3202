import re
import ujson as json
import asyncio
import aiohttp
import time
from quart import Quart, request, jsonify, current_app
from quart_cors import cors
from openai import AsyncOpenAI
import time
from google.cloud import secretmanager
from google.auth import default
import traceback
from google.cloud import logging
from aiohttp import ClientSession, ClientTimeout
from collections import defaultdict
from collections import Counter
import base64
import io
from PIL import Image

logging_client = logging.Client()
logger = logging_client.logger('wikipedia-service-profiling')

app = Quart(__name__)
app = cors(app, allow_origin="*")

async def update_token_count(counter, model, tokens):
    model_key = model.replace('-', '_')
    if model_key not in counter:
        print(f"Adding new model to counter: {model_key}")
        counter[model_key] = 0
    counter[model_key] += tokens
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
        elif model in all_models:
            all_models[model] = count
        else:
            print(f"Unknown model in counter: {model}")
            all_models[model] = count  
    return all_models

credentials, project_id = default()
if credentials is not None:
    print(f"Authenticated account: {credentials.service_account_email}")
else:
    print("No credentials found.")
    raise ValueError("Credentials not found")

# def log_message(severity, message):
#     logger.log_text(message, severity=severity)
#     print(f"{severity}: {message}")

# def log_memory_usage(message):
#     try:
#         process = psutil.Process(os.getpid())
#         mem_info = process.memory_info()
#         mem_usage = mem_info.rss / 1024 / 1024  # Convert to MB
#         log_entry = f"{message}: Memory usage: {mem_usage:.2f} MB"
#         logger.log_text(log_entry)
#     except Exception as e:
#         logger.log_text(f"Error logging memory usage: {str(e)}")

def access_secret_version(project_id, secret_id, version_id):
    client = secretmanager.SecretManagerServiceClient(credentials=credentials)
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    # print(f"NAME>> {name}")
    response = client.access_secret_version(name=name)
    payload = response.payload.data.decode("UTF-8")
    return payload

openai_secret_id = "openai-api-key"
openai_secret_version = "latest"

openai_api_key = access_secret_version(project_id, openai_secret_id, openai_secret_version)

# if openai_api_key:
#     print(f"OpenAI API Key successfully retrieved: {openai_api_key[:4]}...")  
# else:
#     print("OpenAI API Key not found. Please check the environment variable.")

client = AsyncOpenAI(api_key=openai_api_key)
    
async def get_topic_from_query(query, token_counter):
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a precise topic extractor. Your task is to identify the main topic from a given query, ensuring it's relevant to fluid dynamics or mechanics."},
                {"role": "user", "content": f"""
                Analyze the following query and perform these tasks:
                1. Identify and extract the principal main topic related to fluid dynamics or mechanics. There must be only one main topic.
                2. Return only the main topic with no additional text, symbols, or decorators.
                3. Be as quick and efficient as possible.
                 
                Query: {query}
                """}
            ],
            temperature=0,
            max_tokens=20,
            n=1,
            stop=["\n", ".", ","]
        )
        
        topic = response.choices[0].message.content.strip()        
      
        print(f"DEBUG: Topic extracted: '{topic}'")
        await update_token_count(token_counter, response.model, response.usage.total_tokens)
        return topic
    except Exception as e:
        print(f"ERROR: Failed to extract topic: {str(e)}")
        return None
    finally:
        end_time = time.time()
        print(f"DEBUG: Topic extraction time: {end_time - start_time:.3f} seconds")

async def is_disambiguation_page(title, session):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "titles": title,
        "format": "json"
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        pages = data['query']['pages']
        # print("DEBUG: Disambiguation Pages:\n", pages)
        page = next(iter(pages.values()))
        # print("DEBUG: Disambiguation Page:\n", page)
        return 'disambiguation' in page.get('pageprops', {})

async def search_wikipedia(topic, query, session, token_counter):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{topic} \"fluid dynamics\"",
        "srwhat": "text",
        "format": "json",
        "srlimit": "max",
        "srsort": "relevance",
        "srnamespace": "0",  
    } 
    async with session.get(url, params=params) as response:
        data = await response.json()
        # print(f"DEBUG: Raw API response: {data}")
        if 'query' in data and 'search' in data['query']:
            search_results = data['query']['search']
            if search_results:
                titles = [result['title'] for result in search_results if 'title' in result]
                # print(f"DEBUG: Titles before ranking: {titles}...")  # Show first 5 titles
                ranked_titles = await rank_articles(query, titles, token_counter)   
                
                for title in ranked_titles:
                    # print(f"DEBUG: Checking title: {title}")
                    if not await is_disambiguation_page(title, session):
                        # print(f"DEBUG: Selected Wikipedia article: '{title}'")
                        text, image_info = await fetch_article_content(session, title)
                        if text:
                            return title, text, image_info
        
        print("DEBUG: No suitable Wikipedia article found")
        return None, None, None

async def rank_articles(query, titles, token_counter):
    prompt = f"""Rank the following Wikipedia article titles based on their relevance to ANSWERING the user's question: '{query}'. 

Tasks:
0. Guage whether a more general wiki article would be more relevant or a more niche wiki article would be more appropriate to get information from.
1. Populate the following JSON sorted by what you gauge as the MOST relevant top THREE articles to the user's query!
2. Be as quick and efficient as possible.
{{
  "title1": "Most Relevant Title",
  "title2": "Second Most Relevant Title",
  "title3": "Third Most Relevant Title"
}}

Titles to rank:
{chr(10).join(f"- {title}" for title in titles)}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",

            messages=[
                {"role": "system", "content": "You are an AI that ranks Wikipedia articles based on their relevance to Fluid Dynamics/Mechanics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )        
        await update_token_count(token_counter, response.model, response.usage.total_tokens)
        content = response.choices[0].message.content
        # print(f"DEBUG: Raw AI response for ranking: {content}")        
        try:
            parsed_content = json.loads(content)
            ranked_titles = list(parsed_content.values())
            if not ranked_titles:
                print("DEBUG: No titles ranked by AI, falling back to original order")
                ranked_titles = titles
        except json.JSONDecodeError as json_error:
            print(f"DEBUG: JSON parsing error: {str(json_error)}")
            ranked_titles = titles          
    except Exception as e:
        print(f"DEBUG: Error in ranking articles: {str(e)}")
        ranked_titles = titles  
    print(f"DEBUG: Final ranked titles: {ranked_titles[:3]}...")  
    return ranked_titles

async def fetch_and_process_wikipedia_content(session, title, query):
    # Fetch article content
    text, image_info = await fetch_article_content(session, title)
    
    # Create a dictionary to hold all the data
    article_data = {
        "title": title,
        "full_text": text,
        "images": image_info
    }
    
    # Generate summary
    summary = await generate_summary(query, article_data["full_text"])
    article_data["summary"] = summary
    
    # Rank images
    ranked_images = await rank_images_by_relevance(query, article_data["images"], token_counter=None)
    article_data["ranked_images"] = ranked_images
    
    return article_data

async def fetch_article_content(session, title):
    print(f"DEBUG: Fetching content for article: '{title}'")
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|images",
        "imlimit": "max",
        "explaintext": "true",
        "exintro": "false",
    }
    
    async with session.get(url, params=params) as response:
        data = await response.json()
        pages = data.get("query", {}).get("pages", {})
        
        if not pages:
            print(f"DEBUG: No content found for article: '{title}'")
            return None, None

        page_id = next(iter(pages))
        page_data = pages[page_id]
        
        text = page_data.get('extract', '')
        if text:
            print(f"DEBUG: Article text processing ...")
        else:
            print(f"DEBUG: No text content found for article: '{title}'")

        image_titles = [img['title'] for img in page_data.get('images', []) if img['title'].lower().endswith(('.png', '.jpeg', '.gif', '.svg'))]
        print("DEBUG: Fetching image info...")
        image_info = await fetch_image_info(session, image_titles)

        # print(f"\nDEBUG: Article '{title}' - Total valid images found: {len(image_info)}")
        
        # # Print the first three images for debugging
        # print("DEBUG: First valid images:")
        # for i, img in enumerate(image_info, 1):
        #     print(f"Image {i}:")
        #     print(f"  Title: {img['title']}")
        #     print(f"  URL: {img['url']}...")
        #     print(f"  Description: {img.get('description', 'No description')[:50]}...")
        #     print(f"  Caption: {img.get('caption', 'No caption')[:50]}...")
        #     print()

        return text, image_info

async def fetch_image_info(session, image_titles):
    # print("Image Titles;\n",image_titles)
    EXCLUDED_IMAGE_KEYWORDS = {
        'logo', 'icon', 'symbol', 'banner', 'button',
        'wiki', 'wikipedia', 'commons', 'creative commons',
        'interface', 'ui', 'user interface',
        'placeholder', 'default', 'blank',
        'disambig', 'disambiguation',
        'edit', 'delete', 'flag'
    }    
    async def fetch_single_image(title):
        if any(keyword.lower() in title.lower() for keyword in EXCLUDED_IMAGE_KEYWORDS):
            # print(f"DEBUG: Skipping excluded image: {title}")
            return None

        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
        }
        async with session.get(url, params=params) as response:
            data = await response.json()
            pages = data.get("query", {}).get("pages", {})
            if pages:
                page = next(iter(pages.values()))
                if 'imageinfo' in page:
                    info = page['imageinfo'][0]
                    description = info.get('extmetadata', {}).get('ImageDescription', {}).get('value', '')
                    caption = info.get('extmetadata', {}).get('ObjectName', {}).get('value', '')
                    # print(f"Title of Image: {title}\n")
                    # print(f"URL of Image: {url}\n")
                    # print(f"Description of Image: {description}\n")
                    # print(f"Caption of Image: {caption}\n")

                    return {
                        'title': title,
                        'url': info['url'],
                        'description': description,
                        'caption': caption
                    }
        return None

    tasks = [fetch_single_image(title) for title in image_titles]
    results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]

async def generate_summary(query, text, max_tokens=200, temperature=0.5, token_counter=None):
    max_text_length = 1000
    truncated_text = text[:max_text_length]

    response = await client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a skilled summarizer specializing in Wikipedia content. Your goal is to craft concise, informative summaries that directly answer the user's query. Be as quick and efficient as possible!!"},
            {"role": "user", "content": f"Question: {query}\n\nWikipedia Text:\n\n{truncated_text}\n\nSummary:"}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    if token_counter is not None:
        await update_token_count(token_counter, response.model, response.usage.total_tokens)
    summary = response.choices[0].message.content.strip()
    summary = ' '.join(summary.split())
    return summary

async def rank_images_by_relevance(query, image_titles_urls, max_images=3, token_counter=None):
    if not image_titles_urls:
        print("DEBUG: No images to rank")
        return {"status": "no_images", "message": "No images found to rank"}

    descriptions = {image['url']: image.get('description', 'No description') for image in image_titles_urls}

    prompt = f"""
Your tasks:
1. Logically rank the following images based on their relevance to the user's query: '{query}'.
2. Consider the image information passed to you.
3. Only return the TOP TWO most relevant images, DO NOT INCLUDE IMAGES THAT MAKE NO SENSE IN HELPING THE USER GRASP FOUNDATIONAL KNOWLEDGE OF THE CONCEPT BEHIND THEIR QUESTION!!  
JSON format:
{{
  "ranked_images": [
    {{
      "title": "Image Title",
      "url": "Image URL",
    }},
    ...
  ]
}}

Images to rank:
"""
    for image in image_titles_urls:
        prompt += f"- Title: {image.get('title', 'No title')}\n  URL: {image['url']}\n\n"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        parsed_content = json.loads(content)

        if 'ranked_images' in parsed_content and isinstance(parsed_content['ranked_images'], list):
            ranked_images = parsed_content['ranked_images']

            # Append descriptions to the ranked images
            for image in ranked_images:
                image['description'] = descriptions.get(image['url'], 'No description')

            # print("Ranked images:\n", ranked_images)
            if token_counter is not None:
                await update_token_count(token_counter, response.model, response.usage.total_tokens)
            return {"status": "success", "images": ranked_images}
        else:
            raise ValueError("Unexpected JSON structure: 'ranked_images' key not found or not a list")

    except Exception as e:
        print(f"DEBUG: Error in ranking images: {str(e)}")
        return {"status": "error", "message": f"An error occurred while ranking images: {str(e)}"}

def is_plain_text(text):
    return not bool(re.search(r'<[^>]+>', text))


async def get_raster_image_url(session, svg_url):
    print(f"DEBUG: Attempting to get raster version for: {svg_url}")
    
    # Extract the file path from the URL
    match = re.search(r'/wikipedia/commons/(.+)$', svg_url)
    if not match:
        print(f"DEBUG: Invalid SVG URL format: {svg_url}")
        return svg_url

    file_path = match.group(1)
    file_name = file_path.split('/')[-1]
    
    # Construct the thumbnail URL
    thumb_url = f"https://upload.wikimedia.org/wikipedia/commons/thumb/{file_path}/500px-{file_name}.png"
    print(f"DEBUG: Constructed thumbnail URL: {thumb_url}")
    
    # Check if the thumbnail exists
    async with session.head(thumb_url) as response:
        if response.status == 200:
            print(f"DEBUG: Raster version found: {thumb_url}")
            return thumb_url
    
    print(f"DEBUG: Raster version not found, returning original URL: {svg_url}")
    return svg_url

async def get_image_description(client, image_url, image_description, query, token_counter=None):
    # print("Image URL:\n", image_url)
    # print("Image description:\n", image_description)      
    async with aiohttp.ClientSession() as session:
        original_url = image_url
        if image_url.lower().endswith('.svg'):
            # print(f"DEBUG: Fetching raster version for SVG: {image_url}")
            image_url = await get_raster_image_url(session, image_url)
            # print(f"DEBUG: Raster URL: {image_url}")

        print(f"DEBUG: Fetching image from URL: {image_url}")
        async with session.get(image_url) as response:
            if response.status == 200:
                image_data = await response.read()
                # print(f"DEBUG: Image data length: {len(image_data)} bytes")
                # print(f"DEBUG: First few bytes: {image_data[:20].hex()}")
            else:
                print(f"DEBUG: Failed to fetch image: HTTP {response.status}")
                return f"Failed to fetch image: HTTP {response.status}"

    try:
        image = Image.open(io.BytesIO(image_data))
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"DEBUG: Error processing image: {str(e)}")
        return f"Error processing image: {str(e)}"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"""
                        Generate a description of the image passed here, using the image description: `{image_description}`, as a contextual basis to explain to the user how this image can help them grasp a more concrete grounding/application of the concept behind their question: `{query}`. 
                        Absolutely NO referencing of this generated description or how it should help the user, should be included!  
                        Limit your answer to around 3 sentences. Avoid any unicode/ascii/hex.
                        """},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=150
        )
        # print("GPT 4 V Response:\n", response)
        if token_counter is not None:
            await update_token_count(token_counter, response.model, response.usage.total_tokens)        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"DEBUG: Error in GPT-4 Vision API call: {str(e)}")
        return f"Error processing image: {str(e)}"

async def process_query(query, session, token_counter):
    async with aiohttp.ClientSession() as session:
        try:
            main_topic = await get_topic_from_query(query, token_counter)
            if not main_topic:
                return {"error": "Failed to process the query."}

            search_result = await search_wikipedia(main_topic, query, session, token_counter)
            
            if isinstance(search_result, dict) and "error" in search_result:
                return search_result

            article_title, text, image_info = search_result

            if not text:
                return {"error": "Failed to fetch Wikipedia text."}

            print("DEBUG: Generating summary...")
            summary_task = asyncio.create_task(generate_summary(query, text, token_counter=token_counter))
            print("DEBUG: Ranking images...")
            ranked_images_task = asyncio.create_task(rank_images_by_relevance(query, image_info, token_counter=token_counter))
            summary, ranked_images = await asyncio.gather(summary_task, ranked_images_task)
            print("DEBUG: Generating GPT 4V Descriptions...")

            top_three_images = []
            if isinstance(ranked_images, dict) and ranked_images.get("status") == "success":
                image_tasks = []
                for image_info in ranked_images.get("images", [])[:3]:
                    image_url = image_info.get('url', '')
                    image_description = image_info.get('description', '')
                    
                    # Check if the image is SVG and get raster version if necessary
                    if image_url.lower().endswith('.svg'):
                        raster_url = await get_raster_image_url(session, image_url)
                    else:
                        raster_url = image_url
                    
                    if raster_url and image_description:
                        task = get_image_description(client, raster_url, image_description, query, token_counter)
                        image_tasks.append((task, image_url))  # Store original URL for SVG files
                
                descriptions = await asyncio.gather(*(task for task, _ in image_tasks))

                for i, ((_, original_url), image_info) in enumerate(zip(image_tasks, ranked_images.get("images", [])[:3])):
                    top_three_images.append({
                        "Title": image_info.get('title', 'No title'),
                        "URL": original_url,  # Use the original URL, which may be SVG
                        "Description": descriptions[i] if descriptions[i] is not None else "No description available."
                    })
            
            if not top_three_images:
                top_three_images.append({
                    "Title": "No Relevant Images",
                    "URL": "",
                    "Description": f"No relevant images found on the Wikipedia page for '{article_title}'."
                })

            result = {
                "Summary": summary,
                "Images": top_three_images
            }
            return result
        except Exception as e:
            print(f"DEBUG: Error in process_query: {str(e)}")
            print(f"DEBUG: Error traceback: {traceback.format_exc()}")
            return {"error": f"An error occurred while processing the query: {str(e)}"}
        
@app.route('/query', methods=['POST'])
async def handle_query():
    start_time = time.perf_counter()
    # log_memory_usage("Start of handle_technical_query")
    # current_app.logger.debug("Entered handle_query function.")
    token_counter = Counter() 

    try:
        data = await request.get_json()

        request_id = data.get('request_id', 'N/A')

        query = data.get('query', '')
        print(f"[{request_id}] DEBUG: Query Retrieved: {query}...")
        # current_app.logger.debug(f"Received query: '{query}'")

        if not query:
            # current_app.logger.debug("No query provided.")
            return jsonify({"error": "No query provided."}), 400

        print(f"[{request_id}] DEBUG: Processing ...")
        result = await process_query(query, app.aiohttp_session, token_counter)
        print(f"[{request_id}] DEBUG: Processing Compeleted...")

        if isinstance(result, dict) and "error" in result:
            # current_app.logger.debug(f"Error returned from process_query: {result['error']}")
            return jsonify(result), 400

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        token_usage = await get_token_usage(token_counter)

        response = {
            "results": result,
            "token_usage": token_usage,
            "elapsed_time": {
                "total_time": elapsed_time,
            }
        }
        print(f"[{request_id}] DEBUG: Combined Result Compeleted...")
        
        # log_memory_usage("Ending handle_technical_query")
        return jsonify(response)

    except Exception as e:
        current_app.logger.error(f"Error in handle_query: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing the query."}), 500

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

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8083)
    
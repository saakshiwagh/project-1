import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
import asyncio
import logging
import base64
import traceback
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.68
MAX_RESULTS = 10
MAX_CONTEXT_CHUNKS = 4
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI app (single initialization)
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate API key at startup
if not API_KEY:
    logger.error("API_KEY environment variable is not set.")
    raise RuntimeError("API_KEY environment variable is not set.")

# Database connection
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Initialize database
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS discourse_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER,
        topic_id INTEGER,
        topic_title TEXT,
        post_number INTEGER,
        author TEXT,
        created_at TEXT,
        likes INTEGER,
        chunk_index INTEGER,
        content TEXT,
        url TEXT,
        embedding BLOB
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS markdown_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_title TEXT,
        original_url TEXT,
        downloaded_at TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    conn.close()

# Cosine similarity
def cosine_similarity(vec1, vec2):
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0

# Get embedding with retry
async def get_embedding(text, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Getting embedding for text (length: {len(text)})")
            url = "https://aiproxy.org/openai/v1/embeddings"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received embedding")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error getting embedding (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)

# Find similar content
async def find_similar_content(query_embedding, conn):
    try:
        logger.info("Finding similar content in database")
        cursor = conn.cursor()
        results = []
        
        # Discourse chunks
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
               likes, chunk_index, content, url, embedding 
        FROM discourse_chunks 
        WHERE embedding IS NOT NULL
        """)
        discourse_chunks = cursor.fetchall()
        logger.info(f"Processing {len(discourse_chunks)} discourse chunks")
        processed_count = 0
        
        for chunk in discourse_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                if similarity >= SIMILARITY_THRESHOLD:
                    url = chunk["url"]
                    if not url.startswith("http"):
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                    results.append({
                        "source": "discourse",
                        "id": chunk["id"],
                        "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"],
                        "title": chunk["topic_title"],
                        "url": url,
                        "content": chunk["content"],
                        "author": chunk["author"],
                        "created_at": chunk["created_at"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(discourse_chunks)} discourse chunks")
            except Exception as e:
                logger.error(f"Error processing discourse chunk {chunk['id']}: {e}")
        
        # Markdown chunks
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
        FROM markdown_chunks 
        WHERE embedding IS NOT NULL
        """)
        markdown_chunks = cursor.fetchall()
        logger.info(f"Processing {len(markdown_chunks)} markdown chunks")
        processed_count = 0
        
        for chunk in markdown_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                if similarity >= SIMILARITY_THRESHOLD:
                    url = chunk["original_url"]
                    if not url or not url.startswith("http"):
                        url = f"https://docs.onlinedegree.iitm.ac.in/{chunk['doc_title']}"
                    results.append({
                        "source": "markdown",
                        "id": chunk["id"],
                        "title": chunk["doc_title"],
                        "url": url,
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(markdown_chunks)} markdown chunks")
            except Exception as e:
                logger.error(f"Error processing markdown chunk {chunk['id']}: {e}")
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} relevant results above threshold")
        
        grouped_results = {}
        for result in results:
            key = f"discourse_{result['post_id']}" if result["source"] == "discourse" else f"markdown_{result['title']}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        final_results = []
        for key, chunks in grouped_results.items():
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            final_results.extend(chunks[:MAX_CONTEXT_CHUNKS])
        
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Returning {len(final_results[:MAX_RESULTS])} final results after grouping")
        return final_results[:MAX_RESULTS]
    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Enrich with adjacent chunks
async def enrich_with_adjacent_chunks(conn, results):
    try:
        logger.info(f"Enriching {len(results)} results with adjacent chunks")
        cursor = conn.cursor()
        enriched_results = []
        
        for result in results:
            enriched_result = result.copy()
            additional_content = ""
            
            if result["source"] == "discourse":
                post_id = result["post_id"]
                current_chunk_index = result["chunk_index"]
                
                if current_chunk_index > 0:
                    cursor.execute("""
                    SELECT content FROM discourse_chunks 
                    WHERE post_id = ? AND chunk_index = ?
                    """, (post_id, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content = prev_chunk["content"] + " "
                
                cursor.execute("""
                SELECT content FROM discourse_chunks 
                WHERE post_id = ? AND chunk_index = ?
                """, (post_id, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content += " " + next_chunk["content"]
                
            elif result["source"] == "markdown":
                title = result["title"]
                current_chunk_index = result["chunk_index"]
                
                if current_chunk_index > 0:
                    cursor.execute("""
                    SELECT content FROM markdown_chunks 
                    WHERE doc_title = ? AND chunk_index = ?
                    """, (title, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content = prev_chunk["content"] + " "
                
                cursor.execute("""
                SELECT content FROM markdown_chunks 
                WHERE doc_title = ? AND chunk_index = ?
                """, (title, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content += " " + next_chunk["content"]
            
            if additional_content:
                enriched_result["content"] = f"{result['content']} {additional_content}"
            
            enriched_results.append(enriched_result)
        
        logger.info(f"Successfully enriched {len(enriched_results)} results")
        return enriched_results
    except Exception as e:
        error_msg = f"Error in enrich_with_adjacent_chunks: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Generate answer
async def generate_answer(question, relevant_results, max_retries=2):
    retries = 0
    while retries < max_retries:    
        try:
            logger.info(f"Generating answer for question: '{question[:50]}...'")
            context = ""
            for result in relevant_results:
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                context += f"\n\n{source_type} (URL: {result['url']}):\n{result['content'][:1500]}"
            
            prompt = f"""Answer the following question based ONLY on the provided context. 
            If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Return your response in this exact format:
            1. A comprehensive yet concise answer
            2. A "Sources:" section that lists the URLs and relevant text snippets you used
            
            Sources must be in this exact format:
            Sources:
            1. URL: [exact_url_1], Text: [brief quote or description]
            2. URL: [exact_url_2], Text: [brief quote or description]
            
            Make sure the URLs are copied exactly from the context without any changes.
            """
            
            logger.info("Sending request to LLM API")
            url = "https://aiproxy.org/openai/v1/chat/completions"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(3 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception generating answer: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)

# Process multimodal query
async def process_multimodal_query(question, image_base64):
    try:
        logger.info(f"Processing query: '{question[:50]}...', image provided: {image_base64 is not None}")
        if not image_base64:
            logger.info("No image provided, processing as text-only query")
            return await get_embedding(question)
        
        logger.info("Processing multimodal query with image")
        url = "https://aiproxy.org/openai/v1/chat/completions"
        headers = {
            "Authorization": API_KEY,
            "Content-Type": "application/json"
        }
        image_content = f"data:image/jpeg;base64,{image_base64}"
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Look at this image and tell me what you see related to this question: {question}"},
                        {"type": "image_url", "image_url": {"url": image_content}}
                    ]
                }
            ]
        }
        
        logger.info("Sending request to Vision API")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Received image description: '{image_description[:50]}...'")
                    combined_query = f"{question}\nImage context: {image_description}"
                    return await get_embedding(combined_query)
                else:
                    error_text = await response.text()
                    logger.error(f"Error processing image (status {response.status}): {error_text}")
                    logger.info("Falling back to text-only query")
                    return await get_embedding(question)
    except Exception as e:
        logger.error(f"Exception processing multimodal query: {e}")
        logger.error(traceback.format_exc())
        logger.info("Falling back to text-only query due to exception")
        return await get_embedding(question)

# Parse LLM response
def parse_llm_response(response):
    try:
        logger.info("Parsing LLM response")
        parts = response.split("Sources:", 1)
        if len(parts) == 1:
            for heading in ["Source:", "References:", "Reference:"]:
                if heading in response:
                    parts = response.split(heading, 1)
                    break
        
        answer = parts[0].strip()
        links = []
        
        if len(parts) > 1:
            sources_text = parts[1].strip()
            source_lines = sources_text.split("\n")
            
            for line in source_lines:
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)
                url_match = re.search(r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|\[(http[^\]]+)\]|URL:\s*(http\S+)|url:\s*(http\S+)|(http\S+)', line, re.IGNORECASE)
                text_match = re.search(r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|[""](.*?)[""]|Text:\s*"(.*?)"|text:\s*"(.*?)"', line, re.IGNORECASE)
                
                if url_match:
                    url = next((g for g in url_match.groups() if g), "").strip()
                    text = "Source reference"
                    if text_match:
                        text_value = next((g for g in text_match.groups() if g), "")
                        if text_value:
                            text = text_value.strip()
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})
        
        logger.info(f"Parsed answer (length: {len(answer)}) and {len(links)} sources")
        return {"answer": answer, "links": links}
    except Exception as e:
        error_msg = f"Error parsing LLM response: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "answer": "Error parsing the response from the language model.",
            "links": []
        }

# Query endpoint
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        logger.info(f"Received query request: question='{request.question[:50]}...', image_provided={request.image is not None}")
        conn = get_db_connection()
        
        try:
            logger.info("Processing query and generating embedding")
            query_embedding = await process_multimodal_query(
                request.question,
                request.image
            )
            
            logger.info("Finding similar content")
            relevant_results = await find_similar_content(query_embedding, conn)
            
            if not relevant_results:
                logger.info("No relevant results found")
                return {
                    "answer": "I couldn't find any relevant information in my knowledge base.",
                    "links": []
                }
            
            logger.info("Enriching results with adjacent chunks")
            enriched_results = await enrich_with_adjacent_chunks(conn, relevant_results)
            
            logger.info("Generating answer")
            llm_response = await generate_answer(request.question, enriched_results)
            
            logger.info("Parsing LLM response")
            result = parse_llm_response(llm_response)
            
            if not result["links"]:
                logger.info("No links extracted, creating from relevant results")
                links = []
                unique_urls = set()
                for res in relevant_results[:5]:
                    url = res["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                        links.append({"url": url, "text": snippet})
                result["links"] = links
            
            logger.info(f"Returning result: answer_length={len(result['answer'])}, num_links={len(result['links'])}")
            return result
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
        finally:
            conn.close()
    except Exception as e:
        error_msg = f"Unhandled exception in query_knowledge_base: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

# Root endpoint
@app.get("/")
async def root():
    return JSONResponse(content={"message": "TDS Virtual TA is live!"})

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        conn.close()
        return {
            "status": "healthy", 
            "database": "connected", 
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
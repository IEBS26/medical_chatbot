from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import asyncio
import json
import time
import logging
from datetime import datetime, date
from futurehouse_client.models import TaskRequest, RuntimeConfig
from futurehouse_client import FutureHouseClient, JobNames

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys
API_KEYS = [
    os.getenv("API_KEY"),
    os.getenv("API_KEY2"), 
    os.getenv("API_KEY3"),
    os.getenv("API_KEY4"),
    os.getenv("API_KEY5"),
    os.getenv("API_KEY6"),
    os.getenv("API_KEY7"),
    os.getenv("API_KEY8"),
    os.getenv("API_KEY9"),
    os.getenv("API_KEY10"),
    os.getenv("API_KEY11")
]

# Filter out None values
API_KEYS = [key for key in API_KEYS if key is not None]

if not API_KEYS:
    raise ValueError("❌Pinecone Down.")

logger.info(f"✅ Loaded {len(API_KEYS)} API keys")

# Initialize FastAPI
app = FastAPI(title="Medical Chatbot Backend", version="4.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PersistentAPIManager:
    def __init__(self, api_keys: List[str], max_requests_per_key: int = 10):
        self.api_keys = api_keys
        self.max_requests_per_key = max_requests_per_key
        self.clients = {}
        self.counter_file = "api_counter.json"
        self.current_api_index = 0  # Track current API index
        
        # Initialize all clients
        for i, key in enumerate(api_keys):
            try:
                self.clients[i] = FutureHouseClient(api_key=key)
                logger.info(f"✅ Initialized  client {i+1}/{len(api_keys)}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize client : {e}")
        
        # Load or create counter data
        self._load_counter_data()
        logger.info(f"📊 Counter initialized - Global count: {self.global_counter}")
    
    def _load_counter_data(self):
        """Load counter data from local file"""
        try:
            if os.path.exists(self.counter_file):
                with open(self.counter_file, 'r') as f:
                    data = json.load(f)
                
                # Check if it's a new day - if yes, reset
                saved_date = data.get('date', '')
                today = str(date.today())
                
                if saved_date == today:
                    self.global_counter = data.get('global_counter', 0)
                    self.api_usage = data.get('api_usage', {str(i): 0 for i in range(len(self.api_keys))})
                    self.current_api_index = data.get('current_api_index', 0)
                    logger.info(f"📅 Loaded existing counter data for {today}")
                else:
                    logger.info(f"🔄 New day detected ({saved_date} -> {today}). Resetting counters.")
                    self._reset_counters()
            else:
                logger.info("📝 No existing counter file found. Creating new one.")
                self._reset_counters()
        except Exception as e:
            logger.error(f"❌ Error loading counter data: {e}")
            self._reset_counters()
    
    def _reset_counters(self):
        """Reset all counters"""
        self.global_counter = 0
        self.api_usage = {str(i): 0 for i in range(len(self.api_keys))}
        self.current_api_index = 0
        self._save_counter_data()
        logger.info("🔄 All counters reset to 0")
    
    def _save_counter_data(self):
        """Save counter data to local file"""
        try:
            data = {
                'date': str(date.today()),
                'global_counter': self.global_counter,
                'api_usage': self.api_usage,
                'current_api_index': self.current_api_index,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.counter_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving counter data: {e}")
    
    def _find_next_available_api(self):
        """Find the next available API key that hasn't reached its limit"""
        # First, check from current index onwards
        for i in range(self.current_api_index, len(self.api_keys)):
            current_usage = self.api_usage.get(str(i), 0)
            if current_usage < self.max_requests_per_key and i in self.clients:
                return i
        
        # If no API found from current index onwards, check from beginning
        for i in range(0, self.current_api_index):
            current_usage = self.api_usage.get(str(i), 0)
            if current_usage < self.max_requests_per_key and i in self.clients:
                return i
        
        # No available API found
        return None
    
    def get_next_available_client(self):
        """Get the next available client, automatically finding available keys"""
        # Find next available API
        available_api_index = self._find_next_available_api()
        
        if available_api_index is None:
            raise Exception("❌Pinecone Down.")
        
        # Update current API index to the found available one
        self.current_api_index = available_api_index
        
        # Increment global counter
        self.global_counter += 1
        
        # Update individual API usage
        current_usage = self.api_usage.get(str(available_api_index), 0)
        self.api_usage[str(available_api_index)] = current_usage + 1
        
        # Save updated data
        self._save_counter_data()
        
        # Get client
        client = self.clients.get(available_api_index)
        
        if not client:
            logger.error(f"❌  client {available_api_index + 1} is not available")
            raise Exception(f"❌ client {available_api_index + 1} is not available")
        
        logger.info(f"🔑 Using API Key {available_api_index + 1} | Global Count: {self.global_counter} | Key Usage: {self.api_usage[str(available_api_index)]}/{self.max_requests_per_key}")
        
        return client, available_api_index
    
    def mark_api_key_as_failed(self, api_index: int, reason: str = "rate_limit"):
        """Mark an API key as failed (set its usage to max) and move to next available"""
        logger.warning(f"🚫 Marking API Key {api_index + 1} as failed due to {reason}")
        
        # Set usage to maximum (exhausted)
        self.api_usage[str(api_index)] = self.max_requests_per_key
        
        # Find and move to next available API
        next_available = self._find_next_available_api()
        if next_available is not None:
            self.current_api_index = next_available
            logger.info(f"🔄 Switched to API Key {next_available + 1}")
        else:
            logger.warning("⚠️ No more available API keys")
        
        # Save updated data
        self._save_counter_data()
        
        logger.info(f"🔄 API Key {api_index + 1} marked as exhausted.")
    
    def get_available_apis_count(self):
        """Get count of available (non-exhausted) APIs"""
        available_count = 0
        for i in range(len(self.api_keys)):
            current_usage = self.api_usage.get(str(i), 0)
            if current_usage < self.max_requests_per_key and i in self.clients:
                available_count += 1
        return available_count
    
    def get_status(self):
        """Get current status"""
        available_apis = self.get_available_apis_count()
        
        status = {
            'global_counter': self.global_counter,
            'current_api_key': self.current_api_index + 1,
            'max_requests_per_key': self.max_requests_per_key,
            'total_api_keys': len(self.api_keys),
            'available_api_keys': available_apis,
            'date': str(date.today()),
            'api_usage_details': []
        }
        
        for i in range(len(self.api_keys)):
            usage = self.api_usage.get(str(i), 0)
            is_current = (i == self.current_api_index)
            is_available = (usage < self.max_requests_per_key and i in self.clients)
            
            status['api_usage_details'].append({
                'api_key_number': i + 1,
                'requests_used': usage,
                'requests_remaining': max(0, self.max_requests_per_key - usage),
                'is_current': is_current,
                'is_available': is_available,
                'status': 'available' if is_available else 'exhausted'
            })
        
        return status
    
    def manual_reset(self):
        """Manually reset all counters"""
        logger.info("🔄 Manual reset requested")
        self._reset_counters()

# Initialize API manager
api_manager = PersistentAPIManager(API_KEYS, max_requests_per_key=10)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Medical Chatbot Backend started successfully!")
    status = api_manager.get_status()
    logger.info(f"📊 Current Status: Global Counter={status['global_counter']}, Current API Key={status['current_api_key']}, Available APIs={status['available_api_keys']}")

@app.on_event("shutdown") 
async def shutdown_event():
    # Close all clients
    for client in api_manager.clients.values():
        try:
            client.close()
        except:
            pass
    logger.info("🛑 Backend shutdown complete")

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    job_type: str
    continued_job_id: Optional[str] = None

class ChatResponse(BaseModel):
    formatted_answer: str
    task_id: str
    job_type: str
    processing_time: float

# Job name mapping
JOB_MAPPING = {
    "CROW": JobNames.CROW,
    "FALCON": JobNames.FALCON, 
    "PHOENIX": JobNames.PHOENIX,
    "OWL": JobNames.OWL
}

def smart_chunk_text(text: str, chunk_size: int = 50):
    """Intelligently chunk text to preserve formatting"""
    if not text:
        return []
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        chunk_end = min(current_pos + chunk_size, len(text))
        
        if chunk_end < len(text):
            # Find good breaking points
            breaking_points = ['. ', '! ', '? ', '; ', ': ', ', ', ') ', ' ']
            best_break = chunk_end
            
            for bp in breaking_points:
                bp_pos = text.rfind(bp, current_pos, chunk_end)
                if bp_pos != -1:
                    best_break = bp_pos + len(bp)
                    break
            
            # Don't break inside citations
            chunk_text = text[current_pos:best_break]
            open_parens = chunk_text.count('(')
            close_parens = chunk_text.count(')')
            
            if open_parens > close_parens:
                next_close = text.find(')', best_break)
                if next_close != -1 and next_close < len(text):
                    best_break = next_close + 1
            
            chunk_end = best_break
        
        chunk = text[current_pos:chunk_end]
        if chunk.strip():
            chunks.append(chunk)
        
        current_pos = chunk_end
    
    return chunks

@app.get("/")
async def root():
    status = api_manager.get_status()
    return {
        "message": "🚀 Medical Chatbot Backend is running!",
        "status": "healthy",
        "api_status": status
    }

@app.get("/health")
async def health_check():
    status = api_manager.get_status()
    logger.info(f"🏥 Health check - Global Counter: {status['global_counter']}, Current API: {status['current_api_key']}, Available: {status['available_api_keys']}")
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "api_manager_status": status
    }

@app.get("/api-status")
async def detailed_api_status():
    """Get detailed API status"""
    status = api_manager.get_status()
    logger.info(f"📊 Status requested - Available APIs: {status['available_api_keys']}/{status['total_api_keys']}")
    return status

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    async def generate_stream():
        start_time = time.time()

        try:
            # Validate input
            if request.job_type not in JOB_MAPPING:
                error_msg = f"Invalid job_type: {request.job_type}. Must be one of: {list(JOB_MAPPING.keys())}"
                logger.error(f"❌ {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Invalid request parameters'}})}\n\n"
                return

            logger.info(f"🔍 Processing query: {request.query[:50]}... | Job Type: {request.job_type}")

            # Check if any APIs are available
            available_apis = api_manager.get_available_apis_count()
            if available_apis == 0:
                logger.error("❌ No API keys available")
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Service temporarily unavailable.'}})}\n\n"
                return
            
            # Send progress updates
            yield f"data: {json.dumps({'type': 'progress', 'data': {'progress': 20, 'message': 'Initializing...'}})}\n\n"
            await asyncio.sleep(0.1)
   
            yield f"data: {json.dumps({'type': 'progress', 'data': {'progress': 70, 'message': 'Generating your answer...'}})}\n\n"
            await asyncio.sleep(0.1)

            # Retry logic with automatic failover
            max_retries = available_apis  # Try as many times as we have available APIs
            task_response = None

            for retry_attempt in range(max_retries):
                try:
                    # Get next available API client
                    client, api_index = api_manager.get_next_available_client()
                    
                    runtime_config = RuntimeConfig(
                        continued_job_id=request.continued_job_id
                    ) if request.continued_job_id else None

                    task_data = TaskRequest(
                        name=JOB_MAPPING[request.job_type],
                        query=request.query,
                        runtime_config=runtime_config
                    )

                    logger.info(f"🚀 Sending request to API Key {api_index + 1} (Attempt {retry_attempt + 1}/{max_retries})")
                    
                    # Make the API call
                    responses = client.run_tasks_until_done(task_data)
                    task_response = responses[0]
                    
                    logger.info(f"✅ Request successful with API Key {api_index + 1}")
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check for rate limiting (429 errors)
                    if any(keyword in error_msg for keyword in ["429", "rate limit", "quota", "too many requests"]):
                        logger.error(f"🚫 API Key {api_index + 1} hit rate limit: {str(e)}")
                        
                        # Mark this API key as failed
                        api_manager.mark_api_key_as_failed(api_index, "429_rate_limit")
                        
                        # Check if more APIs are available
                        remaining_apis = api_manager.get_available_apis_count()
                        if remaining_apis > 0 and retry_attempt < max_retries - 1:
                            logger.info(f"🔄 {remaining_apis} API keys still available. Trying next...")
                            continue
                        else:
                            logger.error("❌ All API keys have been exhausted")
                            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Service temporarily unavailable - all API keys exhausted'}})}\n\n"
                            return
                    
                    # For other errors, also try next key but don't mark as failed
                    else:
                        logger.error(f"❌ API Key {api_index + 1} failed with error: {str(e)}")
                        
                        # Check if more APIs are available
                        remaining_apis = api_manager.get_available_apis_count()
                        if remaining_apis > 0 and retry_attempt < max_retries - 1:
                            logger.info(f"🔄 Trying next API key due to error (Attempt {retry_attempt + 2})")
                            # Don't increment usage for this failed key - the get_next_available_client already did
                            continue
                        else:
                            logger.error("❌ All retry attempts exhausted")
                            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Service temporarily unavailable'}})}\n\n"
                            return

            # Check if we got a successful response
            if not task_response:
                logger.error("❌ No successful response received after all attempts")
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Service temporarily unavailable'}})}\n\n"
                return

            # Stream successful response
            yield f"data: {json.dumps({'type': 'answer_start', 'data': {'task_id': str(task_response.task_id)}})}\n\n"

            chunks = smart_chunk_text(task_response.formatted_answer, chunk_size=80)
            for chunk in chunks:
                yield f"data: {json.dumps({'type': 'answer_chunk', 'data': {'text': chunk}})}\n\n"
                await asyncio.sleep(0.03)

            processing_time = time.time() - start_time
            logger.info(f"⏱️ Request completed in {processing_time:.2f}s")
            
            yield f"data: {json.dumps({'type': 'complete', 'data': {'task_id': str(task_response.task_id), 'processing_time': processing_time, 'job_type': request.job_type}})}\n\n"

        except Exception as e:
            logger.error(f"❌ Unexpected error in stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'An unexpected error occurred'}})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.post("/reset-counters")
async def reset_counters():
    """Manually reset all counters"""
    logger.info("🔄 Manual counter reset requested")
    api_manager.manual_reset()
    return {
        "message": "All counters have been reset",
        "status": api_manager.get_status()
    }

@app.get("/counter-file-content")
async def get_counter_file_content():
    """Get the raw content of counter file (for debugging)"""
    try:
        if os.path.exists("api_counter.json"):
            with open("api_counter.json", 'r') as f:
                content = json.load(f)
            logger.info(f"📄 Counter file content requested")
            return content
        else:
            logger.warning("📄 Counter file does not exist")
            return {"message": "Counter file does not exist"}
    except Exception as e:
        logger.error(f"❌ Error reading counter file: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
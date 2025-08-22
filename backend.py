from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import asyncio
import json
import time
import logging
from datetime import datetime, date, timedelta
import httpx
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
    os.getenv("API_KEY11"),
    os.getenv("API_KEY12"),
    os.getenv("API_KEY13"),
    os.getenv("API_KEY14")
]

# Filter out None values
API_KEYS = [key for key in API_KEYS if key is not None]

if not API_KEYS:
    raise ValueError("❌Pinecone Down.")

logger.info(f"✅ Loaded {len(API_KEYS)} API keys")

# Initialize FastAPI
app = FastAPI(title="Medical Chatbot Backend", version="4.0.0")

# FIXED CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins for now
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://ieb-chatbot.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Cache-Control",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    expose_headers=["*"],
    max_age=3600,
)

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Keep-alive configuration
KEEP_ALIVE_INTERVAL = 45 * 60  # 45 minutes in seconds (5 minutes before Render shuts down)
RENDER_APP_URL = os.getenv("RENDER_APP_URL", "https://ieb-chatbot.onrender.com")  # Set your Render URL

class KeepAliveManager:
    def __init__(self):
        self.is_running = False
        self.last_ping_time = None
        self.ping_count = 0
        self.failed_pings = 0
        
    async def start_keep_alive(self):
        """Start the keep-alive background task"""
        if not self.is_running:
            self.is_running = True
            asyncio.create_task(self._keep_alive_loop())
            logger.info(f"🔄 Keep-alive started - will ping every {KEEP_ALIVE_INTERVAL/60:.1f} minutes")
    
    async def _keep_alive_loop(self):
        """Background loop to keep the server alive"""
        while self.is_running:
            try:
                await asyncio.sleep(KEEP_ALIVE_INTERVAL)
                await self._ping_self()
            except Exception as e:
                logger.error(f"❌ Keep-alive loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _ping_self(self):
        """Ping the server to keep it alive"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                start_time = time.time()
                response = await client.get(f"{RENDER_APP_URL}/health")
                ping_time = time.time() - start_time
                
                if response.status_code == 200:
                    self.ping_count += 1
                    self.last_ping_time = datetime.now()
                    self.failed_pings = 0
                    logger.info(f"✅ Keep-alive ping #{self.ping_count} successful ({ping_time:.2f}s)")
                else:
                    self.failed_pings += 1
                    logger.warning(f"⚠️ Keep-alive ping failed with status {response.status_code}")
                    
        except Exception as e:
            self.failed_pings += 1
            logger.error(f"❌ Keep-alive ping failed: {e}")
    
    def get_status(self):
        """Get keep-alive status"""
        return {
            "is_running": self.is_running,
            "ping_count": self.ping_count,
            "failed_pings": self.failed_pings,
            "last_ping_time": self.last_ping_time.isoformat() if self.last_ping_time else None,
            "next_ping_in_seconds": KEEP_ALIVE_INTERVAL if self.is_running else None,
            "ping_interval_minutes": KEEP_ALIVE_INTERVAL / 60
        }
    
    def stop(self):
        """Stop the keep-alive mechanism"""
        self.is_running = False
        logger.info("🛑 Keep-alive stopped")

# Initialize keep-alive manager
keep_alive_manager = KeepAliveManager()

# Add explicit preflight handler
@app.options("/{full_path:path}")
async def options_handler():
    """Handle all OPTIONS requests (preflight)"""
    return {"message": "OK"}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML file"""
    try:
        with open("frontend.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please upload frontend.html</p>",
            status_code=404
        )

class EnhancedAPIManager:
    def __init__(self, api_keys: List[str], max_requests_per_key: int = 10):
        self.api_keys = api_keys
        self.max_requests_per_key = max_requests_per_key
        self.clients = {}
        
        # Enhanced storage structure for individual API key tracking
        self.api_data = {}
        
        # Try to use environment variable for persistent storage (for external DB)
        self.use_external_storage = os.getenv("USE_EXTERNAL_STORAGE", "false").lower() == "true"
        self.storage_file = "api_counter.json"
        
        self.current_api_index = 0
        
        # Initialize all clients
        for i, key in enumerate(api_keys):
            try:
                self.clients[i] = FutureHouseClient(api_key=key)
                logger.info(f"✅ Initialized client {i+1}/{len(api_keys)}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize client {i}: {e}")
        
        # Initialize API data structure
        self._initialize_api_data()
        
        # Load existing data if available
        self._load_api_data()
        
        logger.info(f"📊 API Manager initialized")
    
    def _initialize_api_data(self):
        """Initialize the API data structure for all keys"""
        for i in range(len(self.api_keys)):
            if str(i) not in self.api_data:
                self.api_data[str(i)] = {
                    'usage_count': 0,
                    'exhausted_at': None,
                    'reset_at': None,
                    'is_available': True,
                    'last_used': None
                }
    
    def _load_api_data(self):
        """Load API data from storage"""
        try:
            if not self.use_external_storage and os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    stored_data = json.load(f)
                    
                for api_index, data in stored_data.get('api_data', {}).items():
                    if api_index in self.api_data:
                        self.api_data[api_index].update(data)
                        
                self.current_api_index = stored_data.get('current_api_index', 0)
                logger.info("📅 Loaded existing API data")
            else:
                logger.info("📝 No existing data found or using external storage")
                
            self._check_and_reset_expired_keys()
            
        except Exception as e:
            logger.error(f"❌ Error loading API data: {e}")
            self._initialize_api_data()
    
    def _save_api_data(self):
        """Save API data to storage"""
        if self.use_external_storage:
            logger.debug("Using in-memory storage for external deployment")
            return
            
        try:
            data = {
                'api_data': self.api_data,
                'current_api_index': self.current_api_index,
                'last_updated': datetime.now().isoformat(),
                'total_api_keys': len(self.api_keys)
            }
            
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error saving API data: {e}")
    
    def _check_and_reset_expired_keys(self):
        """Check and reset API keys that have passed their 24-hour reset time"""
        current_time = datetime.now()
        reset_count = 0
        
        for api_index, data in self.api_data.items():
            if data['reset_at']:
                try:
                    reset_time = datetime.fromisoformat(data['reset_at'])
                    if current_time >= reset_time:
                        self.api_data[api_index] = {
                            'usage_count': 0,
                            'exhausted_at': None,
                            'reset_at': None,
                            'is_available': True,
                            'last_used': data.get('last_used')
                        }
                        reset_count += 1
                        logger.info(f"🔄 API Key {int(api_index) + 1} has been reset after 24 hours")
                        
                except Exception as e:
                    logger.error(f"❌ Error parsing reset time for API {api_index}: {e}")
        
        if reset_count > 0:
            self._save_api_data()
            logger.info(f"✅ Reset {reset_count} API keys")
    
    def _find_next_available_api(self):
        """Find the next available API key"""
        self._check_and_reset_expired_keys()
        
        for offset in range(len(self.api_keys)):
            i = (self.current_api_index + offset) % len(self.api_keys)
            
            if (str(i) in self.api_data and 
                self.api_data[str(i)]['is_available'] and 
                self.api_data[str(i)]['usage_count'] < self.max_requests_per_key and 
                i in self.clients):
                return i
        
        return None
    
    def get_next_available_client(self):
        """Get the next available client"""
        available_api_index = self._find_next_available_api()
        
        if available_api_index is None:
            raise Exception("❌ All API keys are currently exhausted. Please try again later.")
        
        self.current_api_index = available_api_index
        api_key_str = str(available_api_index)
        
        self.api_data[api_key_str]['usage_count'] += 1
        self.api_data[api_key_str]['last_used'] = datetime.now().isoformat()
        
        if self.api_data[api_key_str]['usage_count'] >= self.max_requests_per_key:
            current_time = datetime.now()
            reset_time = current_time + timedelta(hours=24)
            
            self.api_data[api_key_str]['exhausted_at'] = current_time.isoformat()
            self.api_data[api_key_str]['reset_at'] = reset_time.isoformat()
            self.api_data[api_key_str]['is_available'] = False
            
            logger.warning(f"🚫 API Key {available_api_index + 1} has reached its limit. Will reset at {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self._save_api_data()
        
        client = self.clients.get(available_api_index)
        
        if not client:
            raise Exception(f"❌ Client {available_api_index + 1} is not available")
        
        logger.info(f"🔑 Using API Key {available_api_index + 1} | Usage: {self.api_data[api_key_str]['usage_count']}/{self.max_requests_per_key}")
        
        return client, available_api_index
    
    def mark_api_key_as_failed(self, api_index: int, reason: str = "rate_limit"):
        """Mark an API key as failed and set it to reset after 24 hours"""
        api_key_str = str(api_index)
        
        if api_key_str in self.api_data:
            current_time = datetime.now()
            reset_time = current_time + timedelta(hours=24)
            
            self.api_data[api_key_str]['usage_count'] = self.max_requests_per_key
            self.api_data[api_key_str]['exhausted_at'] = current_time.isoformat()
            self.api_data[api_key_str]['reset_at'] = reset_time.isoformat()
            self.api_data[api_key_str]['is_available'] = False
            
            logger.warning(f"🚫 API Key {api_index + 1} marked as failed due to {reason}. Will reset at {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            next_available = self._find_next_available_api()
            if next_available is not None:
                self.current_api_index = next_available
                logger.info(f"🔄 Switched to API Key {next_available + 1}")
            
            self._save_api_data()
    
    def get_available_apis_count(self):
        """Get count of currently available APIs"""
        self._check_and_reset_expired_keys()
        available_count = 0
        
        for i, data in self.api_data.items():
            if (data['is_available'] and 
                data['usage_count'] < self.max_requests_per_key and 
                int(i) in self.clients):
                available_count += 1
                
        return available_count
    
    def get_status(self):
        """Get comprehensive status"""
        self._check_and_reset_expired_keys()
        
        available_apis = self.get_available_apis_count()
        total_usage = sum(data['usage_count'] for data in self.api_data.values())
        
        status = {
            'total_requests_made': total_usage,
            'current_api_key': self.current_api_index + 1,
            'max_requests_per_key': self.max_requests_per_key,
            'total_api_keys': len(self.api_keys),
            'available_api_keys': available_apis,
            'exhausted_api_keys': len(self.api_keys) - available_apis,
            'storage_type': 'external' if self.use_external_storage else 'file',
            'api_usage_details': []
        }
        
        for i in range(len(self.api_keys)):
            data = self.api_data.get(str(i), {})
            usage_count = data.get('usage_count', 0)
            is_available = data.get('is_available', True)
            exhausted_at = data.get('exhausted_at')
            reset_at = data.get('reset_at')
            
            api_detail = {
                'api_key_number': i + 1,
                'requests_used': usage_count,
                'requests_remaining': max(0, self.max_requests_per_key - usage_count),
                'is_current': (i == self.current_api_index),
                'is_available': is_available and usage_count < self.max_requests_per_key,
                'status': 'available' if (is_available and usage_count < self.max_requests_per_key) else 'exhausted',
                'exhausted_at': exhausted_at,
                'reset_at': reset_at
            }
            
            if reset_at:
                try:
                    reset_time = datetime.fromisoformat(reset_at)
                    time_until_reset = reset_time - datetime.now()
                    if time_until_reset.total_seconds() > 0:
                        hours = int(time_until_reset.total_seconds() // 3600)
                        minutes = int((time_until_reset.total_seconds() % 3600) // 60)
                        api_detail['time_until_reset'] = f"{hours}h {minutes}m"
                    else:
                        api_detail['time_until_reset'] = "Ready for reset"
                except:
                    api_detail['time_until_reset'] = "Unknown"
            
            status['api_usage_details'].append(api_detail)
        
        return status
    
    def manual_reset(self):
        """Manually reset all counters"""
        logger.info("🔄 Manual reset requested")
        self._initialize_api_data()
        self.current_api_index = 0
        self._save_api_data()
    
    def reset_specific_key(self, api_index: int):
        """Reset a specific API key"""
        if 0 <= api_index < len(self.api_keys):
            api_key_str = str(api_index)
            self.api_data[api_key_str] = {
                'usage_count': 0,
                'exhausted_at': None,
                'reset_at': None,
                'is_available': True,
                'last_used': self.api_data[api_key_str].get('last_used')
            }
            self._save_api_data()
            logger.info(f"🔄 API Key {api_index + 1} has been manually reset")
            return True
        return False

# Initialize API manager
api_manager = EnhancedAPIManager(API_KEYS, max_requests_per_key=10)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Medical Chatbot Backend started successfully!")
    status = api_manager.get_status()
    logger.info(f"📊 Status: Available APIs={status['available_api_keys']}/{status['total_api_keys']}, Total Requests={status['total_requests_made']}")
    
    # Start keep-alive mechanism
    await keep_alive_manager.start_keep_alive()

@app.on_event("shutdown") 
async def shutdown_event():
    keep_alive_manager.stop()
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
            breaking_points = ['. ', '! ', '? ', '; ', ': ', ', ', ') ', ' ']
            best_break = chunk_end
            
            for bp in breaking_points:
                bp_pos = text.rfind(bp, current_pos, chunk_end)
                if bp_pos != -1:
                    best_break = bp_pos + len(bp)
                    break
            
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

@app.get("/health")
async def health_check():
    status = api_manager.get_status()
    keep_alive_status = keep_alive_manager.get_status()
    logger.info(f"🏥 Health check - Available: {status['available_api_keys']}/{status['total_api_keys']}")
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "api_manager_status": status,
        "keep_alive_status": keep_alive_status
    }

@app.get("/keep-alive-status")
async def get_keep_alive_status():
    """Get keep-alive status"""
    return keep_alive_manager.get_status()

@app.post("/start-keep-alive")
async def start_keep_alive():
    """Manually start keep-alive"""
    await keep_alive_manager.start_keep_alive()
    return {"message": "Keep-alive started", "status": keep_alive_manager.get_status()}

@app.post("/stop-keep-alive")
async def stop_keep_alive():
    """Manually stop keep-alive"""
    keep_alive_manager.stop()
    return {"message": "Keep-alive stopped", "status": keep_alive_manager.get_status()}

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
            if request.job_type not in JOB_MAPPING:
                error_msg = f"Invalid job_type: {request.job_type}. Must be one of: {list(JOB_MAPPING.keys())}"
                logger.error(f"❌ {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Invalid request parameters'}})}\n\n"
                return

            logger.info(f"🔍 Processing query: {request.query[:50]}... | Job Type: {request.job_type}")

            available_apis = api_manager.get_available_apis_count()
            if available_apis == 0:
                logger.error("❌ No API keys available")
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Service temporarily unavailable. All API keys are exhausted. Please try again later.'}})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'progress', 'data': {'progress': 20, 'message': 'Initializing...'}})}\n\n"
            await asyncio.sleep(0.1)
   
            yield f"data: {json.dumps({'type': 'progress', 'data': {'progress': 70, 'message': 'Generating your answer...'}})}\n\n"
            await asyncio.sleep(0.1)

            max_retries = available_apis
            task_response = None

            for retry_attempt in range(max_retries):
                try:
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
                    
                    responses = client.run_tasks_until_done(task_data)
                    task_response = responses[0]
                    
                    logger.info(f"✅ Request successful with API Key {api_index + 1}")
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    if any(keyword in error_msg for keyword in ["429", "rate limit", "quota", "too many requests"]):
                        logger.error(f"🚫 API Key {api_index + 1} hit rate limit: {str(e)}")
                        
                        api_manager.mark_api_key_as_failed(api_index, "429_rate_limit")
                        
                        remaining_apis = api_manager.get_available_apis_count()
                        if remaining_apis > 0 and retry_attempt < max_retries - 1:
                            logger.info(f"🔄 {remaining_apis} API keys still available. Trying next...")
                            continue
                        else:
                            logger.error("❌ All API keys have been exhausted")
                            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Service temporarily unavailable - all API keys exhausted'}})}\n\n"
                            return
                    
                    else:
                        logger.error(f"❌ API Key {api_index + 1} failed with error: {str(e)}")
                        
                        remaining_apis = api_manager.get_available_apis_count()
                        if remaining_apis > 0 and retry_attempt < max_retries - 1:
                            logger.info(f"🔄 Trying next API key due to error (Attempt {retry_attempt + 2})")
                            continue
                        else:
                            logger.error("❌ All retry attempts exhausted")
                            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Service temporarily unavailable'}})}\n\n"
                            return

            if not task_response:
                logger.error("❌ No successful response received after all attempts")
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Service temporarily unavailable'}})}\n\n"
                return

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
            "Connection": "keep-alive"
            # Removed manual CORS header - let middleware handle it
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

@app.post("/reset-api-key/{api_index}")
async def reset_specific_api_key(api_index: int):
    """Reset a specific API key"""
    if api_manager.reset_specific_key(api_index):
        return {
            "message": f"API Key {api_index + 1} has been reset",
            "status": api_manager.get_status()
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid API key index")

@app.get("/counter-file-content")
async def get_counter_file_content():
    """Get the current API data (for debugging)"""
    try:
        status = api_manager.get_status()
        logger.info(f"📄 API data requested")
        return {
            "api_data": api_manager.api_data,
            "current_status": status
        }
    except Exception as e:
        logger.error(f"❌ Error getting API data: {e}")
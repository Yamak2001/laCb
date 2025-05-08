# backend/middleware/monitoring.py

import time
import json
import uuid
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class MonitoringMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, store_limit: int = 100):
        super().__init__(app)
        self.requests = []
        self.store_limit = store_limit
        
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        
        # Capture request start time
        start_time = time.time()
        
        # Create request log entry
        request_log = {
            "id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Try to capture request body if it's a POST/PUT/PATCH request
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Clone the request body
                body = await request.body()
                request_log["body"] = body.decode() if body else None
                
                # We need to create a new request with the same body for the next middleware
                # to be able to read the body again
                async def receive():
                    return {"type": "http.request", "body": body}
                
                request._receive = receive
            except Exception as e:
                request_log["body_error"] = str(e)
        
        # Process the request and get response
        try:
            response = await call_next(request)
            
            # Capture response data
            response_log = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Try to capture response body (this is tricky and might not work for all responses)
            # For production, you'd need a more sophisticated approach
            request_log.update(response_log)
            
            # Store the log
            self._store_log(request_log)
            
            return response
            
        except Exception as e:
            # Log exception
            request_log["error"] = str(e)
            request_log["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            self._store_log(request_log)
            raise e
    
    def _store_log(self, log):
        """Store log and maintain size limit"""
        self.requests.append(log)
        if len(self.requests) > self.store_limit:
            self.requests.pop(0)
    
    def get_logs(self, limit=None, filter_path=None, filter_method=None):
        """Get stored logs with optional filtering"""
        filtered = self.requests
        
        if filter_path:
            filtered = [log for log in filtered if filter_path in log["path"]]
            
        if filter_method:
            filtered = [log for log in filtered if log["method"] == filter_method]
            
        # Return newest first
        filtered = sorted(filtered, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        if limit:
            return filtered[:limit]
        return filtered

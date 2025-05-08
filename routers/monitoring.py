# backend/routers/monitoring.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import APIKeyHeader
from starlette.requests import Request

router = APIRouter(
    prefix="/api/monitoring",
    tags=["monitoring"],
)

# Simple API key security
API_KEY = "your-secret-monitoring-key"  # In production, use environment variables
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@router.get("/requests")
async def get_request_logs(
    request: Request,
    limit: int = 50,
    path: str = None,
    method: str = None,
    _: str = Depends(verify_api_key)
):
    """Get request logs with optional filtering"""
    middleware = request.app.state.monitoring_middleware
    return {
        "requests": middleware.get_logs(limit=limit, filter_path=path, filter_method=method)
    }

@router.get("/stats")
async def get_stats(request: Request, _: str = Depends(verify_api_key)):
    """Get basic stats about captured requests"""
    middleware = request.app.state.monitoring_middleware
    logs = middleware.requests
    
    # Count requests by path
    paths = {}
    for log in logs:
        path = log["path"]
        paths[path] = paths.get(path, 0) + 1
    
    # Count requests by status code
    status_codes = {}
    for log in logs:
        if "status_code" in log:
            status = log["status_code"]
            status_codes[status] = status_codes.get(status, 0) + 1
    
    # Calculate average response time
    response_times = [log.get("response_time_ms", 0) for log in logs if "response_time_ms" in log]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Count errors
    error_count = sum(1 for log in logs if "error" in log)
    
    return {
        "total_requests": len(logs),
        "paths": paths,
        "status_codes": status_codes,
        "avg_response_time_ms": round(avg_response_time, 2),
        "error_count": error_count
    }

@router.delete("/clear")
async def clear_logs(request: Request, _: str = Depends(verify_api_key)):
    """Clear all stored request logs"""
    middleware = request.app.state.monitoring_middleware
    middleware.requests = []
    return {"message": "Request logs cleared"}

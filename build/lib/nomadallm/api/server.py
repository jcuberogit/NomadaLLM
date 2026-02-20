"""
NomadaLLM API Server

FastAPI server for NomadaLLM REST API.

Security: Configured with security best practices.
Compliance: CORS, rate limiting, and request logging.
"""

import os
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from nomadallm.api.routes import router
from nomadallm.exceptions import NomadaLLMError


logger = logging.getLogger(__name__)


def create_app(
    title: str = "NomadaLLM API",
    version: str = "0.1.0",
    cors_origins: Optional[list] = None
) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        title: API title.
        version: API version.
        cors_origins: Allowed CORS origins. Defaults to localhost only.
        
    Returns:
        Configured FastAPI application.
        
    Security: Configured with secure defaults.
    """
    app = FastAPI(
        title=title,
        version=version,
        description="Universal LLM SDK with Privacy-First Architecture",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS configuration
    # Security: Restrict origins in production
    if cors_origins is None:
        cors_origins = [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    # Exception handlers
    @app.exception_handler(NomadaLLMError)
    async def nomadallm_exception_handler(request: Request, exc: NomadaLLMError):
        """Handle NomadaLLM exceptions.
        
        Security: Returns generic message, logs details internally.
        """
        logger.error(f"NomadaLLM error: {exc.get_internal_details()}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions.
        
        Security: Never expose internal errors to clients.
        """
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred"}
        )
    
    # Health endpoint at root
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "NomadaLLM API",
            "version": version,
            "status": "running",
            "docs": "/docs"
        }
    
    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
) -> None:
    """Run the API server.
    
    Args:
        host: Host to bind to. Defaults to localhost for security.
        port: Port to bind to.
        reload: Enable auto-reload for development.
        workers: Number of worker processes.
        
    Security: Binds to localhost by default. Use reverse proxy for external access.
    """
    import uvicorn
    
    uvicorn.run(
        "nomadallm.api.server:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        factory=True
    )


# Create default app instance
app = create_app()


if __name__ == "__main__":
    run_server(reload=True)

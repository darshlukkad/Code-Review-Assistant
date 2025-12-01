"""
FastAPI backend for code review service.

Provides REST API endpoints for code review functionality.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Code Review Assistant API",
    description="REST API for automated code quality analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class CodeReviewRequest(BaseModel):
    """Request model for code review."""
    code: str
    language: str = "python"
    threshold: float = 0.5


class IssueDetail(BaseModel):
    """Model for a single detected issue."""
    type: str
    confidence: float
    description: str
    severity: str


class CodeReviewResponse(BaseModel):
    """Response model for code review."""
    code: str
    issues: List[IssueDetail]
    num_issues: int
    overall_quality_score: float


# Global model instance (loaded on startup)
reviewer = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global reviewer
    
    try:
        from src.deployment.inference import CodeReviewer
        
        # TODO: Update with actual model path
        model_path = "models/best_model.pt"
        
        logger.info(f"Loading model from {model_path}")
        reviewer = CodeReviewer(model_path=model_path)
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will run in demo mode with mock predictions")
        reviewer = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Code Review Assistant API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": reviewer is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": reviewer is not None
    }


@app.post("/review", response_model=CodeReviewResponse)
async def review_code(request: CodeReviewRequest):
    """
    Review code and detect quality issues.
    
    Args:
        request: Code review request
    
    Returns:
        Code review results with detected issues
    """
    try:
        if reviewer is None:
            # Demo mode - return mock results
            logger.warning("Model not loaded, returning demo results")
            return _get_demo_review(request.code)
        
        # Perform actual review
        result = reviewer.review(
            code=request.code,
            threshold=request.threshold
        )
        
        return CodeReviewResponse(**result)
    
    except Exception as e:
        logger.error(f"Error during review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": "CodeBERT",
                "description": "Fine-tuned CodeBERT for multi-label code quality classification",
                "version": "1.0.0",
                "loaded": reviewer is not None
            }
        ]
    }


@app.get("/labels")
async def get_labels():
    """Get available issue labels."""
    return {
        "labels": [
            {
                "name": "bug",
                "description": "Potential bug or error that could cause runtime failures"
            },
            {
                "name": "security",
                "description": "Security vulnerability or unsafe coding practice"
            },
            {
                "name": "code_smell",
                "description": "Code quality issue indicating poor design or complexity"
            },
            {
                "name": "style",
                "description": "Style or formatting issue violating best practices"
            },
            {
                "name": "performance",
                "description": "Performance issue that could be optimized"
            }
        ]
    }


def _get_demo_review(code: str) -> CodeReviewResponse:
    """
    Generate demo review results for testing.
    
    Args:
        code: Code to review
    
    Returns:
        Mock review results
    """
    # Simple heuristics for demo
    issues = []
    
    if '/ 0' in code or '/ len' in code:
        issues.append(IssueDetail(
            type="bug",
            confidence=0.92,
            description="Potential division by zero error",
            severity="high"
        ))
    
    if 'password' in code.lower() or 'secret' in code.lower():
        issues.append(IssueDetail(
            type="security",
            confidence=0.85,
            description="Potential security vulnerability with sensitive data",
            severity="critical"
        ))
    
    if code.count('for') >= 2:
        issues.append(IssueDetail(
            type="performance",
            confidence=0.75,
            description="Nested loops may impact performance",
            severity="medium"
        ))
    
    if '"""' not in code and "'''" not in code and 'def ' in code:
        issues.append(IssueDetail(
            type="style",
            confidence=0.68,
            description="Missing docstring for function",
            severity="low"
        ))
    
    quality_score = 100 - (len(issues) * 15)
    
    return CodeReviewResponse(
        code=code,
        issues=issues,
        num_issues=len(issues),
        overall_quality_score=max(0, quality_score)
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_current_user
from app.db import models, schemas
from app.services.inference_service import InferenceService

router = APIRouter()


@router.post("/predict", response_model=schemas.InferenceResponse)
async def predict(
    request: schemas.InferenceRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Generate text using a fine-tuned model.
    
    Loads base model and attaches workspace-specific adapter.
    """
    # Verify model ownership
    model = db.query(models.Model).join(models.Workspace).filter(
        models.Model.id == request.model_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    inference_service = InferenceService()
    
    try:
        result = await inference_service.generate(
            model=model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return schemas.InferenceResponse(
            model_id=model.id,
            prompt=request.prompt,
            generated_text=result["generated_text"],
            tokens_used=result["tokens_used"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@router.post("/batch", response_model=schemas.BatchInferenceResponse)
async def batch_predict(
    request: schemas.BatchInferenceRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Generate text for multiple prompts.
    """
    # Verify model ownership
    model = db.query(models.Model).join(models.Workspace).filter(
        models.Model.id == request.model_id,
        models.Workspace.owner_id == current_user.id
    ).first()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    inference_service = InferenceService()
    results = []
    
    for prompt in request.prompts:
        try:
            result = await inference_service.generate(
                model=model,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            results.append({
                "prompt": prompt,
                "generated_text": result["generated_text"],
                "tokens_used": result["tokens_used"]
            })
        except Exception as e:
            results.append({
                "prompt": prompt,
                "error": str(e)
            })
    
    return schemas.BatchInferenceResponse(
        model_id=model.id,
        results=results
    )

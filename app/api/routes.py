from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from app.application.services import ImageRestorationService
from app.infrastructure.image_processor import OpenCVImageProcessor

router = APIRouter()
processor = OpenCVImageProcessor()
service = ImageRestorationService(processor)

@router.post("/restore")
async def restore_image(
    file: UploadFile = File(...),
    method: str = Form("jacobi"),
    tolerance: float = Form(0.0001),
    max_iterations: int = Form(1000),
    omega: float = Form(1.5)
):
    image_data = await file.read()
    result = service.process(image_data, method, tolerance, max_iterations, omega)
    
    if "error" in result:
        return JSONResponse(status_code=400, content={"message": result["error"]})
        
    return JSONResponse(content={
        "message": f"Image {file.filename} processed.",
        "filename": file.filename,
        "status": "success",
        **result
    })

from fastapi import APIRouter
from app.models.schemas import HardwareInfoResponse
from app.services.hardware import detect_hardware_and_env

router = APIRouter(prefix="/api/system", tags=["System & Hardware"])


@router.get("/hardware", response_model=HardwareInfoResponse)
def get_hardware_info():
    return detect_hardware_and_env()

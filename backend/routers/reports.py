"""FastAPI router: report generation (PDF/Excel download)."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


@router.get("/reports/excel")
async def generate_excel():
    """Generate and return an Excel report of all segments."""
    try:
        from utils.report_generator import generate_excel_report
        from backend.main import app_state
        if not app_state.get("loaded"):
            raise HTTPException(503, "Models not loaded.")
        path = generate_excel_report(None, str(ARTIFACTS_DIR / "report.xlsx"))
        return FileResponse(path, media_type="application/vnd.ms-excel", filename="segmentation_report.xlsx")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/reports/pdf")
async def generate_pdf():
    """Generate and return a PDF executive summary."""
    try:
        from utils.report_generator import generate_pdf_report
        from backend.main import app_state
        if not app_state.get("loaded"):
            raise HTTPException(503, "Models not loaded.")
        path = generate_pdf_report(str(ARTIFACTS_DIR / "report.pdf"))
        return FileResponse(path, media_type="application/pdf", filename="segmentation_report.pdf")
    except Exception as e:
        raise HTTPException(500, str(e))

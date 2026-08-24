import uuid
from fastapi import APIRouter, HTTPException, Response
from typing import List, Optional

from app.db.engine import get_project_db
from app.db.models import LayoutProfileModel
from app.db.repository import StructureRepository
from app.models.schemas import LayoutProfileCreate, LayoutProfileResponse, PreviewRequest
from app.services.renderer.preview_service import PreviewService
from app.services.renderer.theme_engine import ThemeEngine

router = APIRouter(prefix="/api/projects/{project_id}/layout", tags=["Layout & Preview"])


@router.get("/profiles", response_model=List[LayoutProfileResponse])
def list_layout_profiles(project_id: str):
    db = get_project_db(project_id)
    try:
        profiles = db.query(LayoutProfileModel).filter(LayoutProfileModel.project_id == project_id).all()
        if not profiles:
            # Create default Classic Book profile
            p = LayoutProfileModel(
                id=str(uuid.uuid4()),
                project_id=project_id,
                name="Classic Book",
                is_default=True
            )
            db.add(p)
            db.commit()
            db.refresh(p)
            profiles = [p]

        return [
            LayoutProfileResponse(
                id=p.id,
                project_id=p.project_id,
                name=p.name,
                page_size=p.page_size,
                page_width_mm=p.page_width_mm,
                page_height_mm=p.page_height_mm,
                margin_top_mm=p.margin_top_mm,
                margin_bottom_mm=p.margin_bottom_mm,
                margin_left_mm=p.margin_left_mm,
                margin_right_mm=p.margin_right_mm,
                body_font=p.body_font,
                heading_font=p.heading_font,
                body_font_size_pt=p.body_font_size_pt,
                line_height=p.line_height,
                paragraph_spacing_pt=p.paragraph_spacing_pt,
                first_line_indent_mm=p.first_line_indent_mm,
                text_alignment=p.text_alignment,
                chapter_break_mode=p.chapter_break_mode,
                show_header=p.show_header,
                show_footer=p.show_footer,
                show_page_number=p.show_page_number,
                is_default=p.is_default,
                created_at=p.created_at
            ) for p in profiles
        ]
    finally:
        db.close()


@router.post("/profiles", response_model=LayoutProfileResponse)
def create_layout_profile(project_id: str, payload: LayoutProfileCreate):
    db = get_project_db(project_id)
    try:
        p = LayoutProfileModel(
            id=str(uuid.uuid4()),
            project_id=project_id,
            is_default=False,
            **payload.model_dump()
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        return LayoutProfileResponse(
            id=p.id,
            project_id=p.project_id,
            name=p.name,
            page_size=p.page_size,
            page_width_mm=p.page_width_mm,
            page_height_mm=p.page_height_mm,
            margin_top_mm=p.margin_top_mm,
            margin_bottom_mm=p.margin_bottom_mm,
            margin_left_mm=p.margin_left_mm,
            margin_right_mm=p.margin_right_mm,
            body_font=p.body_font,
            heading_font=p.heading_font,
            body_font_size_pt=p.body_font_size_pt,
            line_height=p.line_height,
            paragraph_spacing_pt=p.paragraph_spacing_pt,
            first_line_indent_mm=p.first_line_indent_mm,
            text_alignment=p.text_alignment,
            chapter_break_mode=p.chapter_break_mode,
            show_header=p.show_header,
            show_footer=p.show_footer,
            show_page_number=p.show_page_number,
            is_default=p.is_default,
            created_at=p.created_at
        )
    finally:
        db.close()


@router.patch("/profiles/{profile_id}", response_model=LayoutProfileResponse)
def update_layout_profile(project_id: str, profile_id: str, payload: LayoutProfileCreate):
    db = get_project_db(project_id)
    try:
        p = db.query(LayoutProfileModel).filter(LayoutProfileModel.id == profile_id).first()
        if not p:
            raise HTTPException(status_code=404, detail="Không tìm thấy mẫu định dạng layout.")

        for k, v in payload.model_dump(exclude_unset=True).items():
            setattr(p, k, v)

        db.commit()
        db.refresh(p)
        return LayoutProfileResponse(
            id=p.id,
            project_id=p.project_id,
            name=p.name,
            page_size=p.page_size,
            page_width_mm=p.page_width_mm,
            page_height_mm=p.page_height_mm,
            margin_top_mm=p.margin_top_mm,
            margin_bottom_mm=p.margin_bottom_mm,
            margin_left_mm=p.margin_left_mm,
            margin_right_mm=p.margin_right_mm,
            body_font=p.body_font,
            heading_font=p.heading_font,
            body_font_size_pt=p.body_font_size_pt,
            line_height=p.line_height,
            paragraph_spacing_pt=p.paragraph_spacing_pt,
            first_line_indent_mm=p.first_line_indent_mm,
            text_alignment=p.text_alignment,
            chapter_break_mode=p.chapter_break_mode,
            show_header=p.show_header,
            show_footer=p.show_footer,
            show_page_number=p.show_page_number,
            is_default=p.is_default,
            created_at=p.created_at
        )
    finally:
        db.close()


@router.post("/preview")
def generate_preview(project_id: str, payload: PreviewRequest):
    db = get_project_db(project_id)
    struct_repo = StructureRepository(db)
    try:
        doc = struct_repo.get_canonical_document(project_id)
        profile = None
        if payload.layout_profile_id:
            profile = db.query(LayoutProfileModel).filter(LayoutProfileModel.id == payload.layout_profile_id).first()
        if not profile:
            profile = db.query(LayoutProfileModel).filter(LayoutProfileModel.project_id == project_id).first()
        if not profile:
            profile = LayoutProfileModel(
                id=str(uuid.uuid4()),
                project_id=project_id,
                name="Classic Book"
            )

        target_ch = payload.chapter_ids[0] if payload.chapter_ids else None
        preview_html = PreviewService.render_smart_preview(
            doc=doc,
            profile=profile,
            sample_type=payload.sample_type,
            target_chapter_id=target_ch,
            node_limit=40
        )
        return Response(content=preview_html, media_type="text/html")
    finally:
        db.close()

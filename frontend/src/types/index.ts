export type NodeType =
  | 'heading'
  | 'paragraph'
  | 'quote'
  | 'list'
  | 'list_item'
  | 'image'
  | 'caption'
  | 'table'
  | 'footnote'
  | 'footnote_reference'
  | 'equation'
  | 'code_block'
  | 'horizontal_rule'
  | 'page_break_hint';

export type NodeStatus =
  | 'PENDING'
  | 'QUEUED'
  | 'TRANSLATING'
  | 'TRANSLATED'
  | 'QA_PENDING'
  | 'QA_PASSED'
  | 'NEEDS_REVIEW'
  | 'FAILED';

export type ApprovalStatus = 'UNREVIEWED' | 'APPROVED' | 'NEEDS_WORK';

export type DocumentType =
  | 'GENERAL'
  | 'ACADEMIC'
  | 'TECHNICAL'
  | 'BUSINESS'
  | 'FINANCE'
  | 'LEGAL'
  | 'LITERATURE'
  | 'SELF_HELP'
  | 'MANUAL'
  | 'BIOGRAPHY'
  | 'OTHER';

export type TranslationMode =
  | 'NATURAL'
  | 'BALANCED'
  | 'FAITHFUL'
  | 'ACADEMIC'
  | 'TECHNICAL'
  | 'CUSTOM';

export type QALevel = 'OFF' | 'DETERMINISTIC_ONLY' | 'BALANCED' | 'HIGH_QUALITY';

export interface SourceMapping {
  source_document: string;
  source_page_start: number;
  source_page_end: number;
  source_block_ids: number[];
  bounding_box?: number[];
}

export interface NodeMetadata {
  font_name?: string;
  font_size?: number;
  font_weight?: string;
  is_italic?: boolean;
  is_bold?: boolean;
  heading_level?: number;
  table_as_image?: boolean;
  table_rows?: string[][];
  image_asset_id?: string;
  image_caption?: string;
  footnote_number?: string;
  confidence: number;
  structure_issues?: string[];
}

export interface DocumentNode {
  id: string;
  type: NodeType;
  content: string;
  translated_content?: string;
  source_mapping: SourceMapping;
  metadata: NodeMetadata;
  status: NodeStatus;
  approval_status: ApprovalStatus;
  version: number;
  order_index: number;
}

export interface Chapter {
  id: string;
  number?: string;
  title: string;
  translated_title?: string;
  level: number;
  source_pages: number[];
  summary?: string;
  order_index: number;
  nodes: DocumentNode[];
}

export interface CanonicalDocument {
  id: string;
  metadata: {
    title: string;
    author?: string;
    total_pages: number;
    total_words: number;
    total_chapters: number;
    total_nodes: number;
  };
  chapters: Chapter[];
  assets: Array<{
    id: string;
    original_path: string;
    source_page: number;
  }>;
}

export interface Project {
  id: string;
  title: string;
  description?: string;
  source_language: string;
  target_language: string;
  document_type: DocumentType;
  translation_mode: TranslationMode;
  custom_instructions?: string;
  current_stage: string;
  structure_version: number;
  structure_confirmed: boolean;
  selected_model: string;
  qa_level: QALevel;
  style_guide?: Record<string, any>;
  total_pages: number;
  total_words: number;
  total_nodes: number;
  translatable_nodes: number;
  skipped_nodes: number;
  translated_nodes: number;
  progress_percent: number;
  created_at: string;
  updated_at: string;
}

export interface GlossaryItem {
  id: string;
  source_term: string;
  target_term: string;
  category: string;
  notes?: string;
  locked: boolean;
  preferred_target?: string;
  allowed_variants?: string[];
  sense_hint?: string;
  domain?: string;
  part_of_speech?: string;
  preserve_original?: boolean;
  lock_level?: 'HARD' | 'SOFT';
  created_at: string;
}

export interface TranslationPreviewSample {
  node_id: string;
  source: string;
  translation: string;
  quality: {
    passed: boolean;
    issues: Array<{ code: string; severity: string; message: string }>;
    naturalness?: {
      status: string;
      score?: number;
      issues: Array<{ type: string; severity: string; message: string }>;
    };
  };
}

export interface TranslationPreviewResponse {
  samples: TranslationPreviewSample[];
  profile: Record<string, any>;
  prompt_version: string;
}

export interface QAIssue {
  id: string;
  node_id?: string;
  issue_type: string;
  severity: 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  source_snippet?: string;
  translation_snippet?: string;
  suggested_fix?: string;
  status: 'OPEN' | 'RESOLVED' | 'IGNORED';
  created_at: string;
}

export interface SemanticSummary {
  nodes_total: number;
  risk_low: number;
  risk_medium: number;
  risk_high: number;
  critic_reviewed: number;
  semantic_pass: number;
  semantic_failed: number;
  semantic_error: number;
  needs_review: number;
}

export interface SemanticReview {
  id: string;
  node_id: string;
  risk_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  critic_status: 'NOT_REQUIRED' | 'PASS' | 'FAIL' | 'ERROR';
  critic_score?: number;
  node_status: NodeStatus;
  issues: Array<{ type: string; severity: string; message: string }>;
  source_excerpt: string;
  translation_excerpt: string;
}

export interface EntityDecision {
  id: string;
  source_key: string;
  preferred_translation: string;
  entity_type: string;
  aliases?: string[];
  locked: boolean;
  occurrences: number;
  conflicts: number;
}

export interface LayoutProfile {
  id: string;
  project_id: string;
  name: string;
  page_size: string;
  page_width_mm: number;
  page_height_mm: number;
  margin_top_mm: number;
  margin_bottom_mm: number;
  margin_left_mm: number;
  margin_right_mm: number;
  body_font: string;
  heading_font: string;
  body_font_size_pt: number;
  line_height: number;
  paragraph_spacing_pt: number;
  first_line_indent_mm: number;
  text_alignment: string;
  chapter_break_mode: string;
  show_header: boolean;
  show_footer: boolean;
  show_page_number: boolean;
  is_default: boolean;
}

export interface HardwareInfo {
  cpu_name: string;
  cpu_cores: number;
  ram_total_gb: number;
  ram_available_gb: number;
  gpu_name?: string;
  vram_total_gb?: number;
  vram_free_gb?: number;
  cuda_available: boolean;
  disk_free_gb: number;
  ollama_running: boolean;
  installed_models: string[];
  tesseract_available: boolean;
  calibre_available: boolean;
  recommended_preset: string;
}

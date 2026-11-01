// Logic thuần của Bước 4/6, tách khỏi component để test bằng `node --test` không cần thư viện thêm.
// Chỉ dùng cú pháp TypeScript xóa được (không enum/namespace) để Node chạy trực tiếp.
import type { AddressTerms } from '../types';

// Khớp PRESETS trong backend/app/services/translation/glossary_presets.py.
export const GLOSSARY_PRESET_VALUES = ['LEGAL', 'TECHNICAL', 'FINANCE', 'BUSINESS', 'ACADEMIC'];

export type CriticMode = 'AUTO' | 'LINT_ONLY' | 'ALWAYS';

export interface StyleGuideInput {
  register: string;
  sentenceStyle: string;
  addressTerms: AddressTerms;
  criticMode: CriticMode;
  glossaryPresets: string[];
  criticModel: string;
  semanticCriticModel: string;
}

/** Chưa cấu hình thì bật preset của chính lĩnh vực tài liệu, giống mặc định ở backend. */
export function defaultGlossaryPresets(saved: string[] | null | undefined, documentType: string): string[] {
  if (Array.isArray(saved)) {
    return saved;
  }
  return GLOSSARY_PRESET_VALUES.includes(documentType) ? [documentType] : [];
}

/** Gộp cấu hình Bước 4 vào style_guide hiện có của project. */
export function buildStyleGuide(base: Record<string, any> | null | undefined, input: StyleGuideInput): Record<string, any> {
  // Bỏ các dòng xưng hô nhập dở để backend không nhận mục rỗng.
  const pairs = (input.addressTerms.pairs || []).filter((p) => p.speaker && p.listener && p.self && p.other);
  const characters = Object.fromEntries(
    Object.entries(input.addressTerms.characters || {}).filter(([name, term]) => name.trim() && term.trim()),
  );
  const guide: Record<string, any> = {
    ...(base || {}),
    register: input.register,
    sentence_style: input.sentenceStyle,
    address_terms: { ...input.addressTerms, pairs, characters },
    naturalness_critic_mode: input.criticMode,
    glossary_presets: input.glossaryPresets,
  };
  // Model trống nghĩa là "giống model dịch": xóa khóa để backend dùng mặc định.
  for (const [key, value] of [
    ['semantic_critic_model', input.semanticCriticModel],
    ['naturalness_critic_model', input.criticModel],
  ] as const) {
    if (value) {
      guide[key] = value;
    } else {
      delete guide[key];
    }
  }
  return guide;
}

const DRAFT_STAGE_LABELS: Record<string, string> = {
  QUALITY_GATE: 'kiểm tra tự động',
  SEMANTIC: 'kiểm tra ngữ nghĩa',
  NATURALNESS: 'kiểm tra văn phong',
};

export function draftStageLabel(stage: string): string {
  return DRAFT_STAGE_LABELS[stage] || stage;
}

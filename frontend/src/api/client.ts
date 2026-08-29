import axios from 'axios';
import {
  Project,
  CanonicalDocument,
  GlossaryItem,
  QAIssue,
  LayoutProfile,
  HardwareInfo,
  TranslationPreviewResponse,
  SemanticSummary,
  SemanticReview,
  EntityDecision,
} from '../types';

const api = axios.create({
  baseURL: '/api',
});

export const apiClient = {
  // System & Hardware
  getHardwareInfo: () => api.get<HardwareInfo>('/system/hardware').then((r) => r.data),

  // Projects
  getProjects: () => api.get<Project[]>('/projects').then((r) => r.data),
  getProject: (id: string) => api.get<Project>(`/projects/${id}`).then((r) => r.data),
  createProject: (data: Partial<Project>) => api.post<Project>('/projects', data).then((r) => r.data),
  updateProject: (id: string, data: Partial<Project>) => api.patch<Project>(`/projects/${id}`, data).then((r) => r.data),
  deleteProject: (id: string) => api.delete(`/projects/${id}`).then((r) => r.data),
  backupProjectUrl: (id: string) => `/api/projects/${id}/backup`,
  restoreProject: (formData: FormData) => api.post('/projects/restore', formData).then((r) => r.data),

  // Documents
  uploadDocument: (projectId: string, formData: FormData) =>
    api.post(`/projects/${projectId}/documents`, formData).then((r) => r.data),

  // Analysis & OCR
  startAnalysis: (projectId: string) =>
    api.post(`/projects/${projectId}/analyze`).then((r) => r.data),

  // Structure
  getStructure: (projectId: string) =>
    api.get<CanonicalDocument>(`/projects/${projectId}/structure`).then((r) => r.data),
  updateNode: (projectId: string, nodeId: string, data: any) =>
    api.patch(`/projects/${projectId}/structure/nodes/${nodeId}`, data).then((r) => r.data),
  mergeNextNode: (projectId: string, nodeId: string) =>
    api.post(`/projects/${projectId}/structure/nodes/${nodeId}/merge_next`).then((r) => r.data),
  confirmStructure: (projectId: string) =>
    api.post(`/projects/${projectId}/structure/confirm`, { lock_version: true }).then((r) => r.data),

  // Translation
  startTranslation: (projectId: string, data: any) =>
    api.post(`/projects/${projectId}/translation/start`, data).then((r) => r.data),
  pauseTranslation: (projectId: string) =>
    api.post(`/projects/${projectId}/translation/pause`).then((r) => r.data),
  resumeTranslation: (projectId: string) =>
    api.post(`/projects/${projectId}/translation/resume`).then((r) => r.data),
  stopTranslation: (projectId: string) =>
    api.post(`/projects/${projectId}/translation/stop`).then((r) => r.data),
  retryFailedTranslation: (projectId: string) =>
    api.post(`/projects/${projectId}/translation/retry_failed`).then((r) => r.data),
  getTranslationStatus: (projectId: string) =>
    api.get(`/projects/${projectId}/translation/status`).then((r) => r.data),
  retranslateNode: (projectId: string, nodeId: string, instruction: string, customModel?: string) =>
    api.post(`/projects/${projectId}/translation/nodes/${nodeId}/retranslate`, { instruction, custom_model: customModel }).then((r) => r.data),
  previewTranslation: (projectId: string, data: Record<string, any>) =>
    api.post<TranslationPreviewResponse>(`/projects/${projectId}/translation/preview`, data).then((r) => r.data),

  // Glossary
  getGlossary: (projectId: string) =>
    api.get<GlossaryItem[]>(`/projects/${projectId}/glossary`).then((r) => r.data),
  addGlossaryTerm: (projectId: string, data: Partial<GlossaryItem>) =>
    api.post<GlossaryItem>(`/projects/${projectId}/glossary`, data).then((r) => r.data),
  updateGlossaryTerm: (projectId: string, termId: string, data: Partial<GlossaryItem>) =>
    api.patch<GlossaryItem>(`/projects/${projectId}/glossary/${termId}`, data).then((r) => r.data),
  deleteGlossaryTerm: (projectId: string, termId: string) =>
    api.delete(`/projects/${projectId}/glossary/${termId}`).then((r) => r.data),
  autoExtractGlossary: (projectId: string) =>
    api.post(`/projects/${projectId}/glossary/extract_auto`).then((r) => r.data),
  exportGlossaryUrl: (projectId: string) => `/api/projects/${projectId}/glossary/export_csv`,
  importGlossaryCsv: (projectId: string, formData: FormData) =>
    api.post(`/projects/${projectId}/glossary/import_csv`, formData).then((r) => r.data),

  // QA
  runQA: (projectId: string, enableAiQa: boolean = false) =>
    api.post(`/projects/${projectId}/qa/run?enable_ai_qa=${enableAiQa}`).then((r) => r.data),
  getQAIssues: (projectId: string, status?: string) =>
    api.get<QAIssue[]>(`/projects/${projectId}/qa/issues`, { params: { status } }).then((r) => r.data),
  updateQAIssue: (projectId: string, issueId: string, status: string) =>
    api.patch<QAIssue>(`/projects/${projectId}/qa/issues/${issueId}`, { status }).then((r) => r.data),
  getConsistencyScan: (projectId: string) =>
    api.get(`/projects/${projectId}/qa/consistency`).then((r) => r.data),
  findAndReplace: (projectId: string, find_text: string, replace_text: string, apply_changes: boolean = false) =>
    api.post(`/projects/${projectId}/qa/find_replace`, null, { params: { find_text, replace_text, apply_changes } }).then((r) => r.data),
  retranslateAllQAIssues: (projectId: string, instruction?: string, customModel?: string) =>
    api.post(`/projects/${projectId}/qa/retranslate_all_issues`, { instruction, custom_model: customModel }).then((r) => r.data),
  getSemanticSummary: (projectId: string) =>
    api.get<SemanticSummary>(`/projects/${projectId}/qa/semantic-summary`).then((r) => r.data),
  getSemanticReviews: (projectId: string) =>
    api.get<SemanticReview[]>(`/projects/${projectId}/qa/semantic-reviews`).then((r) => r.data),
  runSemanticReview: (projectId: string) =>
    api.post(`/projects/${projectId}/qa/run-semantic-review`).then((r) => r.data),
  repairSemanticNode: (projectId: string, nodeId: string) =>
    api.post(`/projects/${projectId}/qa/repair-semantic/${nodeId}`).then((r) => r.data),
  runGlobalConsistency: (projectId: string) =>
    api.post(`/projects/${projectId}/qa/global-consistency`).then((r) => r.data),
  getEntities: (projectId: string) =>
    api.get<EntityDecision[]>(`/projects/${projectId}/entities`).then((r) => r.data),
  createEntity: (projectId: string, data: {
    source_key: string;
    preferred_translation: string;
    entity_type: string;
    aliases: string[];
    locked: boolean;
  }) => api.post<EntityDecision>(`/projects/${projectId}/entities`, data).then((r) => r.data),
  updateEntity: (projectId: string, entityId: string, data: Partial<EntityDecision>) =>
    api.patch(`/projects/${projectId}/entities/${entityId}`, data).then((r) => r.data),

  // Layout & Preview
  getLayoutProfiles: (projectId: string) =>
    api.get<LayoutProfile[]>(`/projects/${projectId}/layout/profiles`).then((r) => r.data),
  createLayoutProfile: (projectId: string, data: any) =>
    api.post<LayoutProfile>(`/projects/${projectId}/layout/profiles`, data).then((r) => r.data),
  updateLayoutProfile: (projectId: string, profileId: string, data: any) =>
    api.patch<LayoutProfile>(`/projects/${projectId}/layout/profiles/${profileId}`, data).then((r) => r.data),
  getPreviewHtml: (projectId: string, data: any) =>
    api.post(`/projects/${projectId}/layout/preview`, data, { responseType: 'text' }).then((r) => r.data),

  // Export
  exportDocument: (projectId: string, data: any) =>
    api.post(`/projects/${projectId}/export`, data).then((r) => r.data),
};

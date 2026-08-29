import React, { useState, useEffect, useCallback } from 'react';
import {
  CheckSquare,
  AlertTriangle,
  Sparkles,
  RefreshCw,
  Search,
  Replace,
  Save,
  CheckCircle2,
  ChevronRight,
  ChevronDown,
  ArrowRight,
  Eye,
  FileText,
  HelpCircle,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { CanonicalDocument, DocumentNode, Project, QAIssue, Chapter, SemanticSummary, SemanticReview, EntityDecision } from '../types';

interface Step6QAEditorProps {
  project: Project;
  onNext: () => void;
  onRefreshProject: () => void;
}

export const Step6QAEditor: React.FC<Step6QAEditorProps> = ({ project, onNext, onRefreshProject }) => {
  const [doc, setDoc] = useState<CanonicalDocument | null>(null);
  const [selectedNode, setSelectedNode] = useState<DocumentNode | null>(null);
  const [qaIssues, setQaIssues] = useState<QAIssue[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<string | null>(null);
  const [editedText, setEditedText] = useState('');
  
  // Re-translate modal
  const [showRetranslate, setShowRetranslate] = useState(false);
  const [retranslateInstruction, setRetranslateInstruction] = useState('');
  const [retranslating, setRetranslating] = useState(false);

  // Find & Replace
  const [showFindReplace, setShowFindReplace] = useState(false);
  const [findText, setFindText] = useState('');
  const [replaceText, setReplaceText] = useState('');
  const [findResult, setFindResult] = useState<any>(null);

  // Bulk Re-translate state
  const [showBulkConfirmModal, setShowBulkConfirmModal] = useState(false);
  const [bulkInstruction, setBulkInstruction] = useState('');
  const [bulkRetranslating, setBulkRetranslating] = useState(false);
  const [bulkProgress, setBulkProgress] = useState({ current: 0, total: 0, percent: 0, currentTitle: '' });

  const [expandedChapters, setExpandedChapters] = useState<Record<string, boolean>>({});
  const [semanticSummary, setSemanticSummary] = useState<SemanticSummary | null>(null);
  const [semanticReviews, setSemanticReviews] = useState<SemanticReview[]>([]);
  const [entities, setEntities] = useState<EntityDecision[]>([]);
  const [consistencyIssues, setConsistencyIssues] = useState<any[]>([]);
  const [semanticBusy, setSemanticBusy] = useState(false);
  const [semanticFilter, setSemanticFilter] = useState<'ALL' | 'HIGH' | 'FAIL' | 'ERROR' | 'NEEDS_REVIEW'>('ALL');
  const [showEntityForm, setShowEntityForm] = useState(false);
  const [entityForm, setEntityForm] = useState({
    source_key: '', preferred_translation: '', entity_type: 'OTHER', aliases: '', locked: false,
  });
  const [entityError, setEntityError] = useState('');
  const [entitySaving, setEntitySaving] = useState(false);

  const fetchData = async () => {
    try {
      const [structData, issuesData, summaryData, reviewData, entityData] = await Promise.all([
        apiClient.getStructure(project.id),
        apiClient.getQAIssues(project.id),
        apiClient.getSemanticSummary(project.id),
        apiClient.getSemanticReviews(project.id),
        apiClient.getEntities(project.id),
      ]);
      setDoc(structData);
      setQaIssues(issuesData);
      setSemanticSummary(summaryData);
      setSemanticReviews(reviewData);
      setEntities(entityData);
      if (structData.chapters.length > 0) {
        setExpandedChapters({ [structData.chapters[0].id]: true });
        if (structData.chapters[0].nodes.length > 0) {
          const firstNode = structData.chapters[0].nodes[0];
          setSelectedNode(firstNode);
          setEditedText(firstNode.translated_content || '');
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [project.id]);

  // Debounced autosave (PRD Section 90)
  useEffect(() => {
    if (!selectedNode || editedText === (selectedNode.translated_content || '')) return;

    const timer = setTimeout(async () => {
      setSaving(true);
      try {
        await apiClient.updateNode(project.id, selectedNode.id, {
          translated_content: editedText,
        });
        setLastSaved(new Date().toLocaleTimeString('vi-VN'));
        // Update local doc
        if (doc) {
          for (const ch of doc.chapters) {
            const n = ch.nodes.find((item) => item.id === selectedNode.id);
            if (n) {
              n.translated_content = editedText;
              break;
            }
          }
        }
      } catch (e) {
        console.error('Autosave error:', e);
      } finally {
        setSaving(false);
      }
    }, 1000);

    return () => clearTimeout(timer);
  }, [editedText, selectedNode, project.id]);

  const handleSelectNode = (node: DocumentNode) => {
    setSelectedNode(node);
    setEditedText(node.translated_content || '');
  };

  const handleRunQA = async () => {
    try {
      const res = await apiClient.runQA(project.id, false);
      const issues = await apiClient.getQAIssues(project.id);
      setQaIssues(issues);
      alert(`Đã kiểm tra xong: ${res.total_issues} cảnh báo.`);
    } catch (e) {
      alert('Lỗi kiểm tra QA: ' + e);
    }
  };

  const handleRetranslate = async () => {
    if (!selectedNode) return;
    setRetranslating(true);
    try {
      const res = await apiClient.retranslateNode(
        project.id,
        selectedNode.id,
        retranslateInstruction.trim() || 'Viết lại tự nhiên hơn và đúng ngữ cảnh.'
      );
      setEditedText(res.translated_content);
      setSelectedNode({ ...selectedNode, translated_content: res.translated_content });
      setShowRetranslate(false);
      setRetranslateInstruction('');
    } catch (e) {
      alert('Lỗi dịch lại: ' + e);
    } finally {
      setRetranslating(false);
    }
  };

  const handleStartBulkRetranslate = async () => {
    const issueNodeIds = Array.from(new Set(qaIssues.map((i) => i.node_id).filter(Boolean))) as string[];
    if (issueNodeIds.length === 0) {
      alert('Không có đoạn văn nào có cảnh báo QA.');
      return;
    }

    setBulkRetranslating(true);
    setBulkProgress({ current: 0, total: issueNodeIds.length, percent: 0, currentTitle: '' });

    try {
      for (let i = 0; i < issueNodeIds.length; i++) {
        const nid = issueNodeIds[i];
        setBulkProgress({
          current: i + 1,
          total: issueNodeIds.length,
          percent: Math.round(((i + 1) / issueNodeIds.length) * 100),
          currentTitle: `Đoạn ${nid}`,
        });

        try {
          const res = await apiClient.retranslateNode(
            project.id,
            nid,
            bulkInstruction.trim() || 'Dịch lại tự nhiên hơn, chuẩn xác và đúng ngữ cảnh xuất bản.'
          );

          // Update local document
          if (doc) {
            for (const ch of doc.chapters) {
              const n = ch.nodes.find((item) => item.id === nid);
              if (n) {
                n.translated_content = res.translated_content;
                if (selectedNode && selectedNode.id === nid) {
                  setSelectedNode({ ...selectedNode, translated_content: res.translated_content });
                  setEditedText(res.translated_content);
                }
                break;
              }
            }
          }
        } catch (errNode) {
          console.error(`Failed to retranslate node ${nid}:`, errNode);
        }
      }

      // Re-run QA scan to update warnings
      const qaRes = await apiClient.runQA(project.id, false);
      const updatedIssues = await apiClient.getQAIssues(project.id);
      setQaIssues(updatedIssues);

      setShowBulkConfirmModal(false);
      setBulkInstruction('');
      alert(`Đã hoàn tất dịch lại toàn bộ ${issueNodeIds.length} đoạn cảnh báo! Còn lại ${qaRes.total_issues} cảnh báo.`);
    } catch (e) {
      alert('Lỗi trong tiến trình dịch lại tất cả: ' + e);
    } finally {
      setBulkRetranslating(false);
    }
  };

  const handleFindPreview = async () => {
    if (!findText) return;
    try {
      const res = await apiClient.findAndReplace(project.id, findText, replaceText, false);
      setFindResult(res);
    } catch (e) {
      alert('Lỗi: ' + e);
    }
  };

  const handleApplyReplace = async () => {
    if (!findText) return;
    try {
      const res = await apiClient.findAndReplace(project.id, findText, replaceText, true);
      alert(`Đã thay thế ${res.total_matches} vị trí.`);
      setFindResult(null);
      setShowFindReplace(false);
      fetchData();
    } catch (e) {
      alert('Lỗi: ' + e);
    }
  };

  const uniqueIssuesCount = Array.from(new Set(qaIssues.map((i) => i.node_id).filter(Boolean))).length;

  const handleSemanticScan = async () => {
    setSemanticBusy(true);
    try {
      await apiClient.runSemanticReview(project.id);
      const [summary, reviews, issues] = await Promise.all([
        apiClient.getSemanticSummary(project.id),
        apiClient.getSemanticReviews(project.id),
        apiClient.getQAIssues(project.id),
      ]);
      setSemanticSummary(summary);
      setSemanticReviews(reviews);
      setQaIssues(issues);
    } catch (e) {
      alert('Lỗi Semantic Assurance: ' + e);
    } finally {
      setSemanticBusy(false);
    }
  };

  const handleConsistencyScan = async () => {
    try {
      const result = await apiClient.runGlobalConsistency(project.id);
      setConsistencyIssues(result.issues || []);
      setQaIssues(await apiClient.getQAIssues(project.id));
    } catch (e) {
      alert('Lỗi quét nhất quán: ' + e);
    }
  };

  const handleSemanticRepair = async (nodeId: string) => {
    setSemanticBusy(true);
    try {
      await apiClient.repairSemanticNode(project.id, nodeId);
      await fetchData();
    } catch (e) {
      alert('Semantic repair chưa đạt: ' + e);
    } finally {
      setSemanticBusy(false);
    }
  };

  const handleToggleEntity = async (entity: EntityDecision) => {
    await apiClient.updateEntity(project.id, entity.id, { locked: !entity.locked });
    setEntities(await apiClient.getEntities(project.id));
  };

  const handleEntityTranslation = async (entity: EntityDecision, preferredTranslation: string) => {
    const value = preferredTranslation.trim();
    if (!value || value === entity.preferred_translation) return;
    await apiClient.updateEntity(project.id, entity.id, { preferred_translation: value });
    setEntities(await apiClient.getEntities(project.id));
  };

  const entityErrorMessage = (error: any) => {
    const detail = error?.response?.data?.detail;
    const code = typeof detail === 'object' ? detail?.code : '';
    const messages: Record<string, string> = {
      ENTITY_ALREADY_EXISTS: 'Entity này đã tồn tại; hãy chỉnh sửa dòng hiện có.',
      ENTITY_GLOSSARY_CONFLICT: 'Entity xung đột với thuật ngữ khóa trong Glossary.',
      INVALID_ENTITY_TYPE: 'Loại entity không hợp lệ.',
      INVALID_ENTITY: 'Source và bản dịch entity không được rỗng.',
    };
    if (messages[code]) return messages[code];
    if (error?.response?.status === 422) return 'Dữ liệu entity chưa hợp lệ.';
    return typeof detail === 'string' ? detail : 'Không thể tạo entity.';
  };

  const handleCreateEntity = async (event: React.FormEvent) => {
    event.preventDefault();
    setEntitySaving(true);
    setEntityError('');
    try {
      await apiClient.createEntity(project.id, {
        source_key: entityForm.source_key.trim(),
        preferred_translation: entityForm.preferred_translation.trim(),
        entity_type: entityForm.entity_type,
        aliases: entityForm.aliases.split(',').map((item) => item.trim()).filter(Boolean),
        locked: entityForm.locked,
      });
      setEntities(await apiClient.getEntities(project.id));
      setEntityForm({ source_key: '', preferred_translation: '', entity_type: 'OTHER', aliases: '', locked: false });
      setShowEntityForm(false);
    } catch (error) {
      setEntityError(entityErrorMessage(error));
    } finally {
      setEntitySaving(false);
    }
  };

  return (
    <div className="h-[calc(100vh-7.5rem)] flex flex-col">
      {/* Top action toolbar */}
      <div className="bg-slate-950 border-b border-slate-800 px-6 py-2.5 flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-4">
          <h2 className="text-sm font-bold text-white flex items-center space-x-2">
            <CheckSquare className="w-4 h-4 text-sky-400" />
            <span>Bước 6: Trình biên tập song song & Kiểm định QA</span>
          </h2>

          <div className="text-xs text-slate-400 flex items-center space-x-2">
            <span>{saving ? 'Đang tự động lưu...' : lastSaved ? `Đã lưu lúc ${lastSaved}` : 'Đã đồng bộ'}</span>
          </div>
        </div>

        <div className="flex items-center space-x-2.5">
          <button
            onClick={() => setShowFindReplace(true)}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 border border-slate-700 transition-colors"
          >
            <Search className="w-3.5 h-3.5 text-slate-400" />
            <span>Tìm & Thay thế</span>
          </button>

          <button
            onClick={handleRunQA}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 text-sky-400 text-xs font-semibold border border-sky-500/30 transition-all"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Quét QA ({qaIssues.length} cảnh báo)</span>
          </button>

          {uniqueIssuesCount > 0 && (
            <button
              onClick={() => setShowBulkConfirmModal(true)}
              className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 text-xs font-semibold border border-amber-500/40 shadow-sm transition-all"
              title={`${uniqueIssuesCount} đoạn văn bản chứa tổng cộng ${qaIssues.length} cảnh báo QA`}
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Dịch lại {uniqueIssuesCount} đoạn ({qaIssues.length} cảnh báo)</span>
            </button>
          )}

          <button
            onClick={onNext}
            className="flex items-center space-x-2 px-5 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-white text-xs font-semibold shadow-md shadow-emerald-500/20 transition-all"
          >
            <span>Sang Định dạng Layout (Bước 7)</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="shrink-0 grid grid-cols-3 gap-3 px-5 py-3 bg-slate-950/70 border-b border-slate-800 max-h-52 overflow-y-auto">
        <section className="rounded-xl border border-sky-900/60 bg-sky-950/20 p-3 text-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-sky-300">Semantic Assurance</span>
            <button onClick={handleSemanticScan} disabled={semanticBusy} className="px-2 py-1 rounded bg-sky-500/20 text-sky-300">
              {semanticBusy ? 'Đang quét...' : 'Quét có chọn lọc'}
            </button>
          </div>
          <div className="grid grid-cols-4 gap-1 text-center text-slate-300">
            <span>LOW {semanticSummary?.risk_low || 0}</span>
            <span>MED {semanticSummary?.risk_medium || 0}</span>
            <span>HIGH {semanticSummary?.risk_high || 0}</span>
            <span className="text-amber-300">Review {semanticSummary?.needs_review || 0}</span>
          </div>
          <div className="flex gap-1 mt-2">
            {(['ALL', 'HIGH', 'FAIL', 'ERROR', 'NEEDS_REVIEW'] as const).map((value) => (
              <button key={value} onClick={() => setSemanticFilter(value)} className={`px-1 rounded ${semanticFilter === value ? 'bg-sky-500/30 text-sky-200' : 'text-slate-500'}`}>{value}</button>
            ))}
          </div>
          {semanticReviews.filter((item) => semanticFilter === 'ALL' || item.risk_level === semanticFilter || item.critic_status === semanticFilter || item.node_status === semanticFilter).slice(0, 2).map((item) => (
            <div key={item.id} className="mt-2 flex items-center gap-2 text-rose-300">
              <span className="truncate flex-1">{item.risk_level} · {item.critic_status} · {item.issues[0]?.type || 'Không cần critic'}</span>
              {['FAIL', 'ERROR'].includes(item.critic_status) && <button onClick={() => handleSemanticRepair(item.node_id)} className="text-sky-300 shrink-0">Sửa</button>}
            </div>
          ))}
        </section>

        <section className="rounded-xl border border-amber-900/60 bg-amber-950/20 p-3 text-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-amber-300">Global Consistency</span>
            <button onClick={handleConsistencyScan} className="px-2 py-1 rounded bg-amber-500/20 text-amber-300">Quét toàn cục</button>
          </div>
          <div className="text-slate-400">{consistencyIssues.length} nhóm không nhất quán</div>
          {consistencyIssues.slice(0, 2).map((item, index) => (
            <div key={`${item.source_key}-${index}`} className="mt-2 truncate text-amber-200">{item.issue_type}: {item.source_key}</div>
          ))}
        </section>

        <section className="rounded-xl border border-emerald-900/60 bg-emerald-950/20 p-3 text-xs overflow-x-auto">
          <div className="flex items-center justify-between font-semibold text-emerald-300 mb-2">
            <span>Entity Decisions ({entities.length})</span>
            <button onClick={() => { setShowEntityForm((value) => !value); setEntityError(''); }} className="px-2 py-1 rounded bg-emerald-500/20 text-emerald-200">
              {showEntityForm ? 'Đóng' : 'Tạo entity'}
            </button>
          </div>
          {showEntityForm && (
            <form onSubmit={handleCreateEntity} className="mb-3 space-y-1.5 border-b border-emerald-950 pb-3">
              <input required value={entityForm.source_key} onChange={(event) => setEntityForm({ ...entityForm, source_key: event.target.value })} placeholder="Source Entity" className="w-full bg-slate-950 rounded px-2 py-1" />
              <input required value={entityForm.preferred_translation} onChange={(event) => setEntityForm({ ...entityForm, preferred_translation: event.target.value })} placeholder="Bản dịch ưu tiên" className="w-full bg-slate-950 rounded px-2 py-1" />
              <select value={entityForm.entity_type} onChange={(event) => setEntityForm({ ...entityForm, entity_type: event.target.value })} className="w-full bg-slate-950 rounded px-2 py-1">
                {['PERSON', 'ORGANIZATION', 'PRODUCT', 'LOCATION', 'ACRONYM', 'TECHNICAL_IDENTIFIER', 'OTHER'].map((type) => <option key={type} value={type}>{type}</option>)}
              </select>
              <input value={entityForm.aliases} onChange={(event) => setEntityForm({ ...entityForm, aliases: event.target.value })} placeholder="Aliases, phân cách bằng dấu phẩy" className="w-full bg-slate-950 rounded px-2 py-1" />
              <label className="flex items-center gap-1 text-emerald-200"><input type="checkbox" checked={entityForm.locked} onChange={(event) => setEntityForm({ ...entityForm, locked: event.target.checked })} /> Khóa quyết định</label>
              {entityError && <div className="text-rose-300">{entityError}</div>}
              <button type="submit" disabled={entitySaving} className="px-2 py-1 rounded bg-emerald-500/30 text-emerald-100">{entitySaving ? 'Đang lưu...' : 'Lưu entity'}</button>
            </form>
          )}
          {entities.length === 0 ? <div className="text-slate-500">Chưa có quyết định entity.</div> : (
            <table className="w-full text-left text-[10px] text-slate-300">
              <thead><tr className="text-slate-500"><th>Nguồn</th><th>Bản dịch ưu tiên</th><th>Loại</th><th>Lần/Xung đột</th><th>Khóa</th></tr></thead>
              <tbody>{entities.slice(0, 4).map((entity) => (
                <tr key={entity.id} className="border-t border-emerald-950">
                  <td className="pr-1 max-w-24 truncate">{entity.source_key}</td>
                  <td className="pr-1"><input defaultValue={entity.preferred_translation} onBlur={(event) => handleEntityTranslation(entity, event.target.value)} className="w-28 bg-slate-950 rounded px-1 py-0.5" /></td>
                  <td>{entity.entity_type}</td>
                  <td>{entity.occurrences}/{entity.conflicts}</td>
                  <td><button onClick={() => handleToggleEntity(entity)} className="text-emerald-300">{entity.locked ? 'Mở' : 'Khóa'}</button></td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </section>
      </div>

      {/* 3-Pane Editor Layout */}
      {loading ? (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">Đang tải tài liệu...</div>
      ) : !doc ? (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">Không có dữ liệu.</div>
      ) : (
        <div className="flex-1 flex overflow-hidden">
          {/* Pane 1: Chapter Outline */}
          <div className="w-1/5 border-r border-slate-800 overflow-y-auto p-3 space-y-2 bg-slate-950/40">
            <div className="text-[11px] font-semibold text-slate-400 px-2 uppercase tracking-wider mb-2">
              Mục lục chương
            </div>
            {doc.chapters.map((ch) => {
              const isExpanded = !!expandedChapters[ch.id];
              return (
                <div key={ch.id} className="space-y-1">
                  <div
                    onClick={() => setExpandedChapters((p) => ({ ...p, [ch.id]: !p[ch.id] }))}
                    className="flex items-center justify-between p-2 rounded-lg cursor-pointer hover:bg-slate-900 text-xs font-medium text-slate-200 transition-colors"
                  >
                    <div className="flex items-center space-x-1.5 truncate">
                      {isExpanded ? <ChevronDown className="w-3 h-3 text-slate-400 shrink-0" /> : <ChevronRight className="w-3 h-3 text-slate-400 shrink-0" />}
                      <span className="truncate">{ch.translated_title || ch.title}</span>
                    </div>
                    <span className="text-[10px] text-slate-500 shrink-0 ml-1">{ch.nodes.length}</span>
                  </div>

                  {isExpanded && (
                    <div className="pl-3 space-y-1">
                      {ch.nodes.map((n) => {
                        const isSelected = selectedNode?.id === n.id;
                        const hasIssue = qaIssues.some((i) => i.node_id === n.id);
                        return (
                          <div
                            key={n.id}
                            onClick={() => handleSelectNode(n)}
                            className={`p-2 rounded-lg text-[11px] cursor-pointer truncate transition-colors flex items-center justify-between ${
                              isSelected
                                ? 'bg-sky-500/20 text-sky-400 font-semibold'
                                : 'hover:bg-slate-900 text-slate-400'
                            }`}
                          >
                            <span className="truncate">{n.translated_content || n.content}</span>
                            {hasIssue && <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0 ml-1" />}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Pane 2: Source English Viewer */}
          <div className="w-2/5 border-r border-slate-800 overflow-y-auto p-6 space-y-4 bg-slate-900/10">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Văn bản gốc tiếng Anh
              </span>
              {selectedNode && (
                <span className="text-[11px] text-slate-500 font-mono">
                  Trang nguồn: {selectedNode.source_mapping.source_page_start}
                </span>
              )}
            </div>

            {selectedNode ? (
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800/80 text-sm font-serif text-slate-200 leading-relaxed min-h-[300px] whitespace-pre-wrap selection:bg-sky-500/30">
                {selectedNode.content}
              </div>
            ) : (
              <div className="text-center py-20 text-slate-500 text-xs">Chọn một đoạn văn để xem bản gốc.</div>
            )}
          </div>

          {/* Pane 3: Vietnamese Editable Area */}
          <div className="w-2/5 overflow-y-auto p-6 space-y-4 bg-slate-900/30">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-sky-400 uppercase tracking-wider flex items-center space-x-1.5">
                <span>Bản dịch tiếng Việt (Chỉnh sửa trực tiếp)</span>
              </span>

              {selectedNode && (
                <button
                  onClick={() => setShowRetranslate(true)}
                  className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 text-sky-400 text-xs font-medium border border-sky-500/30 transition-all"
                >
                  <Sparkles className="w-3 h-3" />
                  <span>Dịch lại đoạn này với AI</span>
                </button>
              )}
            </div>

            {selectedNode ? (
              <div className="space-y-4">
                <textarea
                  value={editedText}
                  onChange={(e) => setEditedText(e.target.value)}
                  rows={14}
                  className="w-full bg-slate-950 border border-slate-800 rounded-2xl p-5 text-sm font-serif text-white leading-relaxed focus:outline-none focus:border-sky-500 resize-none selection:bg-sky-500"
                />

                {/* QA Issues for this node */}
                {qaIssues.filter((i) => i.node_id === selectedNode.id).length > 0 && (
                  <div className="space-y-2">
                    <span className="text-xs font-semibold text-amber-400 flex items-center space-x-1">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      <span>Cảnh báo kiểm định QA cho đoạn này:</span>
                    </span>
                    {qaIssues.filter((i) => i.node_id === selectedNode.id).map((issue) => (
                      <div key={issue.id} className="bg-amber-950/30 border border-amber-800/60 rounded-xl p-3 text-xs text-amber-200 space-y-1">
                        <span className="font-semibold">{issue.issue_type}: {issue.message}</span>
                        {issue.suggested_fix && (
                          <div className="text-[11px] text-amber-300">
                            Gợi ý sửa: <em>{issue.suggested_fix}</em>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-20 text-slate-500 text-xs">Chọn một đoạn văn để chỉnh sửa.</div>
            )}
          </div>
        </div>
      )}

      {/* Retranslate Modal */}
      {showRetranslate && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md p-6 shadow-2xl space-y-4">
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-sky-400" />
              <span>Dịch lại đoạn văn với AI</span>
            </h3>

            <p className="text-xs text-slate-400">
              Nhập chỉ dẫn cụ thể để mô hình AI điều chỉnh phong cách, từ ngữ hoặc mức độ tự nhiên cho riêng đoạn này.
            </p>

            <textarea
              placeholder="Ví dụ: Diễn đạt tự nhiên hơn, bám sát nguyên văn hơn, hoặc giữ nguyên cụm từ tiếng Anh..."
              value={retranslateInstruction}
              onChange={(e) => setRetranslateInstruction(e.target.value)}
              rows={4}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-white focus:outline-none focus:border-sky-500 resize-none"
            />

            <div className="flex items-center justify-end space-x-3 pt-2">
              <button
                type="button"
                onClick={() => setShowRetranslate(false)}
                className="px-4 py-2 rounded-xl text-xs text-slate-400 hover:text-white bg-slate-800"
              >
                Hủy
              </button>
              <button
                type="button"
                onClick={handleRetranslate}
                disabled={retranslating}
                className="px-5 py-2 rounded-xl text-xs text-white bg-sky-500 hover:bg-sky-400 font-semibold transition-all"
              >
                {retranslating ? 'Đang dịch lại...' : 'Tiến hành dịch'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Find & Replace Modal */}
      {showFindReplace && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md p-6 shadow-2xl space-y-4">
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <Replace className="w-4 h-4 text-sky-400" />
              <span>Tìm & Thay thế toàn tài liệu</span>
            </h3>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Cụm từ tìm kiếm</label>
                <input
                  type="text"
                  placeholder="Ví dụ: luồng tiền"
                  value={findText}
                  onChange={(e) => setFindText(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Thay thế bằng</label>
                <input
                  type="text"
                  placeholder="Ví dụ: dòng tiền"
                  value={replaceText}
                  onChange={(e) => setReplaceText(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                />
              </div>

              {findResult && (
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-[11px] text-sky-400">
                  Tìm thấy {findResult.total_matches} vị trí phù hợp trong {findResult.affected_nodes_count} đoạn văn bản.
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-slate-800">
              <button
                type="button"
                onClick={handleFindPreview}
                className="px-3.5 py-1.5 rounded-xl text-xs text-sky-400 bg-sky-500/10 hover:bg-sky-500/20 border border-sky-500/30"
              >
                Kiểm tra trước
              </button>

              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  onClick={() => setShowFindReplace(false)}
                  className="px-3.5 py-1.5 rounded-xl text-xs text-slate-400 hover:text-white bg-slate-800"
                >
                  Đóng
                </button>
                <button
                  type="button"
                  onClick={handleApplyReplace}
                  className="px-4 py-1.5 rounded-xl text-xs text-white bg-sky-500 hover:bg-sky-400 font-semibold"
                >
                  Thay thế tất cả
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Bulk Retranslate All Issues Modal */}
      {showBulkConfirmModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg p-6 shadow-2xl space-y-5">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <RefreshCw className={`w-4 h-4 text-amber-400 ${bulkRetranslating ? 'animate-spin' : ''}`} />
                <span>Dịch lại tất cả đoạn có cảnh báo QA</span>
              </h3>
              {!bulkRetranslating && (
                <button
                  onClick={() => setShowBulkConfirmModal(false)}
                  className="text-slate-500 hover:text-slate-300 text-xs px-2 py-1"
                >
                  ✕
                </button>
              )}
            </div>

            {!bulkRetranslating ? (
              <>
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-xl bg-amber-950/30 border border-amber-800/40 text-center">
                      <div className="text-xl font-bold text-amber-400 font-mono">{qaIssues.length}</div>
                      <div className="text-[11px] font-medium text-slate-300 mt-0.5">Tổng số cảnh báo QA</div>
                    </div>
                    <div className="p-3 rounded-xl bg-sky-950/30 border border-sky-800/40 text-center">
                      <div className="text-xl font-bold text-sky-400 font-mono">{uniqueIssuesCount}</div>
                      <div className="text-[11px] font-medium text-slate-300 mt-0.5">Đoạn văn bản cần dịch lại</div>
                    </div>
                  </div>

                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    💡 <strong>Giải thích:</strong> Tổng cộng có <strong>{qaIssues.length} cảnh báo</strong> nằm trong <strong>{uniqueIssuesCount} đoạn văn bản</strong> (một số đoạn dính đồng thời 2 lỗi như lệch số liệu và dính chữ Hán). Dịch lại {uniqueIssuesCount} đoạn này sẽ giải quyết toàn bộ {qaIssues.length} cảnh báo.
                  </p>
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-medium text-slate-300">
                    Chỉ dẫn bổ sung cho AI (Tùy chọn):
                  </label>
                  <textarea
                    placeholder="Ví dụ: Dịch tự nhiên, đúng văn phong sách in, kiểm tra kỹ ngữ nghĩa..."
                    value={bulkInstruction}
                    onChange={(e) => setBulkInstruction(e.target.value)}
                    rows={3}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-white focus:outline-none focus:border-amber-500 resize-none"
                  />
                </div>

                <div className="flex items-center justify-end space-x-3 pt-2">
                  <button
                    type="button"
                    onClick={() => setShowBulkConfirmModal(false)}
                    className="px-4 py-2 rounded-xl text-xs text-slate-400 hover:text-white bg-slate-800"
                  >
                    Hủy
                  </button>
                  <button
                    type="button"
                    onClick={handleStartBulkRetranslate}
                    className="px-5 py-2 rounded-xl text-xs text-slate-950 bg-amber-400 hover:bg-amber-300 font-bold transition-all shadow-lg shadow-amber-500/20"
                  >
                    Bắt đầu dịch lại ({uniqueIssuesCount} đoạn)
                  </button>
                </div>
              </>
            ) : (
              <div className="space-y-4 py-2">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-slate-300 font-medium flex items-center space-x-2">
                    <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                    <span>Đang dịch lại: <span className="text-white font-mono">{bulkProgress.currentTitle || `Đoạn ${bulkProgress.current}`}</span></span>
                  </div>
                  <div className="text-sm font-bold text-amber-400 font-mono">
                    {bulkProgress.percent}%
                  </div>
                </div>

                {/* Animated Glowing Progress Bar */}
                <div className="w-full bg-slate-950 rounded-full h-3.5 p-0.5 border border-slate-800 overflow-hidden shadow-inner">
                  <div
                    className="bg-gradient-to-r from-amber-500 via-sky-400 to-emerald-400 h-full rounded-full transition-all duration-300 shadow-md shadow-amber-500/30"
                    style={{ width: `${bulkProgress.percent}%` }}
                  />
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-400">
                  <span>Tiến độ: <strong className="text-white">{bulkProgress.current}</strong> / {bulkProgress.total} đoạn</span>
                  <span>Vui lòng giữ nguyên màn hình...</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

import React, { useState, useEffect } from 'react';
import {
  Sliders,
  Sparkles,
  BookA,
  Lock,
  Unlock,
  Plus,
  Trash2,
  Download,
  Upload,
  ArrowRight,
  HelpCircle,
  FileCode,
  RefreshCw,
  CheckCircle2,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { Project, GlossaryItem, TranslationMode, DocumentType, HardwareInfo, TranslationPreviewResponse } from '../types';

interface Step4SetupProps {
  project: Project;
  hardware: HardwareInfo | null;
  onNext: () => void;
  onRefreshProject: () => void;
}

export const Step4Setup: React.FC<Step4SetupProps> = ({ project, hardware, onNext, onRefreshProject }) => {
  const [translationMode, setTranslationMode] = useState<TranslationMode>(project.translation_mode || 'NATURAL');
  const [documentType, setDocumentType] = useState<DocumentType>(project.document_type || 'GENERAL');
  const [selectedModel, setSelectedModel] = useState<string>(project.selected_model || 'qwen2.5:7b');
  const [customInstructions, setCustomInstructions] = useState<string>(project.custom_instructions || '');
  const [glossary, setGlossary] = useState<GlossaryItem[]>([]);
  const [loadingGlossary, setLoadingGlossary] = useState(true);
  const [extracting, setExtracting] = useState(false);
  const [register, setRegister] = useState<string>((project.style_guide?.register as string) || 'ACCESSIBLE');
  const [sentenceStyle, setSentenceStyle] = useState<string>((project.style_guide?.sentence_style as string) || 'MODERATE');
  const [previewing, setPreviewing] = useState(false);
  const [preview, setPreview] = useState<TranslationPreviewResponse | null>(null);

  // New term modal
  const [showAddTerm, setShowAddTerm] = useState(false);
  const [newSrc, setNewSrc] = useState('');
  const [newTgt, setNewTgt] = useState('');
  const [newCat, setNewCat] = useState('FINANCE');

  const fetchGlossary = async () => {
    try {
      const items = await apiClient.getGlossary(project.id);
      setGlossary(items);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingGlossary(false);
    }
  };

  useEffect(() => {
    fetchGlossary();
  }, [project.id]);

  const handleSaveSetup = async () => {
    try {
      await apiClient.updateProject(project.id, {
        translation_mode: translationMode,
        document_type: documentType,
        selected_model: selectedModel,
        custom_instructions: customInstructions,
        style_guide: { ...(project.style_guide || {}), register, sentence_style: sentenceStyle },
      });
      onRefreshProject();
      onNext();
    } catch (e) {
      alert('Lỗi lưu cấu hình: ' + e);
    }
  };

  const handlePreview = async () => {
    setPreviewing(true);
    try {
      const result = await apiClient.previewTranslation(project.id, {
        model_name: selectedModel,
        translation_mode: translationMode,
        document_type: documentType,
        custom_instructions: customInstructions,
        style_register: register,
        sentence_style: sentenceStyle,
      });
      setPreview(result);
    } catch (e: any) {
      alert('Không thể dịch thử: ' + (e?.response?.data?.detail || e));
    } finally {
      setPreviewing(false);
    }
  };

  const handleAcceptStyle = async () => {
    try {
      await apiClient.updateProject(project.id, {
        translation_mode: translationMode,
        document_type: documentType,
        selected_model: selectedModel,
        custom_instructions: customInstructions,
        style_guide: { ...(project.style_guide || {}), register, sentence_style: sentenceStyle },
      });
      onRefreshProject();
      alert('Đã chấp nhận và lưu phong cách dịch thử.');
    } catch (e) {
      alert('Không thể lưu phong cách: ' + e);
    }
  };

  const handleToggleLock = async (term: GlossaryItem) => {
    try {
      await apiClient.updateGlossaryTerm(project.id, term.id, { locked: !term.locked });
      fetchGlossary();
    } catch (e) {
      alert('Lỗi: ' + e);
    }
  };

  const handleDeleteTerm = async (termId: string) => {
    try {
      await apiClient.deleteGlossaryTerm(project.id, termId);
      fetchGlossary();
    } catch (e) {
      alert('Lỗi: ' + e);
    }
  };

  const handleAddTerm = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSrc.trim() || !newTgt.trim()) return;
    try {
      await apiClient.addGlossaryTerm(project.id, {
        source_term: newSrc.trim(),
        target_term: newTgt.trim(),
        category: newCat,
        locked: true,
      });
      setShowAddTerm(false);
      setNewSrc('');
      setNewTgt('');
      fetchGlossary();
    } catch (e) {
      alert('Lỗi thêm thuật ngữ: ' + e);
    }
  };

  const handleAutoExtract = async () => {
    setExtracting(true);
    try {
      await apiClient.autoExtractGlossary(project.id);
      fetchGlossary();
      alert('Đã tự động trích xuất và thêm thuật ngữ thành công!');
    } catch (e) {
      alert('Lỗi trích xuất thuật ngữ: ' + e);
    } finally {
      setExtracting(false);
    }
  };

  const handleImportCsv = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const formData = new FormData();
      formData.append('file', file);
      await apiClient.importGlossaryCsv(project.id, formData);
      fetchGlossary();
      alert('Nhập thuật ngữ từ CSV thành công!');
    } catch (e) {
      alert('Lỗi nhập CSV: ' + e);
    }
  };

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div className="space-y-1">
        <h2 className="text-xl font-bold text-white tracking-tight">Bước 4: Thiết lập chế độ dịch & Bảng thuật ngữ</h2>
        <p className="text-slate-400 text-xs leading-relaxed">
          Tùy chỉnh phong cách dịch tiếng Việt, chọn mô hình AI cục bộ và quản lý bảng thuật ngữ bắt buộc (Locked Glossary) để đảm bảo tính nhất quán xuyên suốt hàng trăm nghìn từ.
        </p>
      </div>

      {/* Grid Settings */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: Modes & Models */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-sky-400" />
            <span>Phong cách & Mô hình dịch</span>
          </h3>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Phong cách dịch (Translation Mode)</label>
            <select
              value={translationMode}
              onChange={(e) => setTranslationMode(e.target.value as TranslationMode)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-sky-500"
            >
              <option value="NATURAL">Natural Vietnamese (Ưu tiên tiếng Việt tự nhiên, chuẩn mực)</option>
              <option value="BALANCED">Balanced (Cân bằng giữa tự nhiên và bám sát nguồn)</option>
              <option value="FAITHFUL">Faithful (Bám sát cấu trúc nguyên tác)</option>
              <option value="ACADEMIC">Academic (Văn phong học thuật, chuyên khảo)</option>
              <option value="TECHNICAL">Technical (Độ chính xác kỹ thuật cao)</option>
              <option value="CUSTOM">Custom (Nhập chỉ dẫn riêng bên dưới)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Lĩnh vực tài liệu (Document Domain)</label>
            <select
              value={documentType}
              onChange={(e) => setDocumentType(e.target.value as DocumentType)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-sky-500"
            >
              <option value="GENERAL">General (Tổng quan / Đa mục đích)</option>
              <option value="FINANCE">Finance (Tài chính, Chứng khoán, Ngân hàng)</option>
              <option value="BUSINESS">Business (Kinh doanh, Quản trị, Marketing)</option>
              <option value="TECHNICAL">Technical (Kỹ thuật, Lập trình, Công nghệ)</option>
              <option value="ACADEMIC">Academic (Khoa học, Luận văn, Nghiên cứu)</option>
              <option value="LEGAL">Legal (Pháp lý, Hợp đồng, Luật)</option>
              <option value="LITERATURE">Literature (Văn học, Tiểu thuyết)</option>
              <option value="SELF_HELP">Self Help (Kỹ năng, Phát triển bản thân)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Mô hình AI Local (Ollama / Local LLM)</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-sky-500 font-mono"
            >
              {hardware?.installed_models && hardware.installed_models.length > 0 ? (
                hardware.installed_models.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))
              ) : (
                <>
                  <option value="qwen2.5:7b">qwen2.5:7b (Khuyên dùng - Cân bằng tốc độ & chất lượng)</option>
                  <option value="qwen2.5:14b">qwen2.5:14b (Chất lượng rất cao)</option>
                  <option value="llama3.1:8b">llama3.1:8b</option>
                  <option value="mock-qwen2.5:7b">mock-qwen2.5:7b (Mô phỏng thử nghiệm nhanh / Test)</option>
                </>
              )}
            </select>
            {hardware && (
              <span className="text-[11px] text-sky-400/80 mt-1 block">
                Đề xuất cho cấu hình máy của bạn: {hardware.recommended_preset}
              </span>
            )}
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Chỉ dẫn dịch bổ sung (Custom Instruction)</label>
            <textarea
              placeholder="Ví dụ: Dịch cho người mới học tài chính, diễn đạt đơn giản nhưng giữ nguyên thuật ngữ..."
              value={customInstructions}
              onChange={(e) => setCustomInstructions(e.target.value)}
              rows={3}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-sky-500 resize-none"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Sắc thái</label>
              <select value={register} onChange={(e) => setRegister(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white">
                <option value="FORMAL">Trang trọng</option>
                <option value="NEUTRAL">Trung tính</option>
                <option value="ACCESSIBLE">Dễ tiếp cận</option>
                <option value="CONVERSATIONAL">Gần gũi</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Tái cấu trúc câu</label>
              <select value={sentenceStyle} onChange={(e) => setSentenceStyle(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white">
                <option value="PRESERVE">Giữ cấu trúc</option>
                <option value="MODERATE">Điều chỉnh vừa phải</option>
                <option value="FREE">Tự nhiên linh hoạt</option>
              </select>
            </div>
          </div>

          {/* Engine Optimizations Info */}
          <div className="bg-sky-500/10 border border-sky-500/20 rounded-xl p-3.5 space-y-1.5">
            <div className="flex items-center space-x-2 text-sky-400 font-semibold text-xs">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Đã kích hoạt Bộ tối ưu Tốc độ & Độ chính xác cao</span>
            </div>
            <ul className="text-[11px] text-slate-400 space-y-1 pl-4 list-disc">
              <li><strong className="text-slate-300">Ngữ cảnh thích ứng:</strong> Ngân sách chunk tính theo model, memory và phần output cần dự trữ.</li>
              <li><strong className="text-slate-300">Độ chính xác xuất bản:</strong> Document profile, chapter memory, ngữ cảnh song ngữ và Quality Gate Phase 1.</li>
            </ul>
          </div>
        </div>


        {/* Right: Glossary Management */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <BookA className="w-4 h-4 text-emerald-400" />
                <span>Bảng thuật ngữ cố định ({glossary.length})</span>
              </h3>

              <div className="flex items-center space-x-2">
                <button
                  onClick={handleAutoExtract}
                  disabled={extracting}
                  className="px-2.5 py-1 rounded-lg bg-sky-500/20 text-sky-400 hover:bg-sky-500/30 border border-sky-500/30 text-[11px] font-medium transition-all"
                >
                  {extracting ? 'Đang trích xuất...' : 'Tự động quét'}
                </button>
                <button
                  onClick={() => setShowAddTerm(true)}
                  className="px-2.5 py-1 rounded-lg bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/30 text-[11px] font-medium transition-all flex items-center space-x-1"
                >
                  <Plus className="w-3 h-3" />
                  <span>Thêm từ</span>
                </button>
              </div>
            </div>

            {/* Glossary Table */}
            <div className="border border-slate-800 rounded-xl overflow-hidden max-h-[300px] overflow-y-auto bg-slate-950/40">
              {loadingGlossary ? (
                <div className="p-8 text-center text-slate-500 text-xs">Đang tải bảng thuật ngữ...</div>
              ) : glossary.length === 0 ? (
                <div className="p-8 text-center text-slate-500 text-xs">
                  Chưa có thuật ngữ nào. Bấm "Tự động quét" hoặc "Thêm từ" để thiết lập.
                </div>
              ) : (
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="py-2 px-3 font-semibold">Thuật ngữ gốc (EN)</th>
                      <th className="py-2 px-3 font-semibold">Bản dịch (VI)</th>
                      <th className="py-2 px-3 font-semibold w-16 text-center">Khóa</th>
                      <th className="py-2 px-3 font-semibold w-12 text-center">Xóa</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {glossary.map((g) => (
                      <tr key={g.id} className="hover:bg-slate-800/30">
                        <td className="py-2 px-3 font-medium text-white">{g.source_term}</td>
                        <td className="py-2 px-3 text-sky-300">{g.target_term}</td>
                        <td className="py-2 px-3 text-center">
                          <button
                            onClick={() => handleToggleLock(g)}
                            title={g.locked ? 'Đang khóa (Bắt buộc dùng)' : 'Chưa khóa'}
                            className="p-1 hover:text-white transition-colors"
                          >
                            {g.locked ? <Lock className="w-3.5 h-3.5 text-emerald-400 mx-auto" /> : <Unlock className="w-3.5 h-3.5 text-slate-500 mx-auto" />}
                          </button>
                        </td>
                        <td className="py-2 px-3 text-center">
                          <button
                            onClick={() => handleDeleteTerm(g.id)}
                            className="p-1 text-slate-500 hover:text-red-400 transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5 mx-auto" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* Import / Export CSV */}
          <div className="flex items-center justify-between pt-2 text-xs border-t border-slate-800/80">
            <label className="cursor-pointer text-slate-400 hover:text-slate-200 flex items-center space-x-1">
              <Upload className="w-3.5 h-3.5 text-slate-500" />
              <span>Nhập từ file CSV</span>
              <input type="file" accept=".csv" onChange={handleImportCsv} className="hidden" />
            </label>

            <a
              href={apiClient.exportGlossaryUrl(project.id)}
              className="text-slate-400 hover:text-slate-200 flex items-center space-x-1"
            >
              <Download className="w-3.5 h-3.5 text-slate-500" />
              <span>Xuất CSV</span>
            </a>
          </div>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2"><BookA className="w-4 h-4 text-sky-400" />Dịch thử theo ngữ cảnh</h3>
            <p className="text-[11px] text-slate-400 mt-1">Xem 5–7 đoạn đại diện trước khi dịch toàn bộ. Kết quả này không ghi vào bản dịch chính.</p>
          </div>
          <button onClick={handlePreview} disabled={previewing} className="shrink-0 flex items-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-500 hover:bg-indigo-400 disabled:opacity-50 text-white text-xs font-semibold">
            {previewing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            <span>{previewing ? 'Đang dịch thử...' : preview ? 'Tạo lại bản dịch thử' : 'Dịch thử'}</span>
          </button>
        </div>
        {preview && (
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="text-[11px] text-sky-300">Profile: {preview.profile.document_type} · {preview.profile.tone} · {preview.prompt_version}</div>
              <button onClick={handleAcceptStyle} className="px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-[11px] font-semibold">Chấp nhận phong cách</button>
            </div>
            {preview.samples.map((sample) => (
              <div key={sample.node_id} className="grid md:grid-cols-2 gap-3 bg-slate-950 border border-slate-800 rounded-xl p-4">
                <div><span className="text-[10px] uppercase text-slate-500">Nguồn</span><p className="text-xs text-slate-300 mt-1 leading-relaxed">{sample.source}</p></div>
                <div><span className="text-[10px] uppercase text-slate-500 flex items-center gap-1">Tiếng Việt {sample.quality.passed && <CheckCircle2 className="w-3 h-3 text-emerald-400" />}</span><p className="text-xs text-white mt-1 leading-relaxed">{sample.translation || 'Mô hình chưa trả về bản dịch hợp lệ.'}</p></div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Save & Proceed button */}
      <div className="flex justify-end pt-4">
        <button
          onClick={handleSaveSetup}
          className="flex items-center space-x-2 px-7 py-3 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold shadow-lg shadow-sky-500/20 transition-all"
        >
          <span>Lưu cấu hình & Bắt đầu Dịch (Bước 5)</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Add Term Modal */}
      {showAddTerm && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md p-6 shadow-2xl space-y-4">
            <h3 className="text-base font-bold text-white">Thêm thuật ngữ cố định</h3>

            <form onSubmit={handleAddTerm} className="space-y-3.5 text-xs">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Thuật ngữ tiếng Anh gốc *</label>
                <input
                  type="text"
                  required
                  placeholder="Ví dụ: cash flow"
                  value={newSrc}
                  onChange={(e) => setNewSrc(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Bản dịch tiếng Việt bắt buộc *</label>
                <input
                  type="text"
                  required
                  placeholder="Ví dụ: dòng tiền"
                  value={newTgt}
                  onChange={(e) => setNewTgt(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Phân loại</label>
                <select
                  value={newCat}
                  onChange={(e) => setNewCat(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                >
                  <option value="FINANCE">Finance / Tài chính</option>
                  <option value="TECHNICAL">Technical / Kỹ thuật</option>
                  <option value="BUSINESS">Business / Kinh doanh</option>
                  <option value="ACADEMIC">Academic / Học thuật</option>
                  <option value="GENERAL">General / Chung</option>
                </select>
              </div>

              <div className="flex items-center justify-end space-x-3 pt-3">
                <button
                  type="button"
                  onClick={() => setShowAddTerm(false)}
                  className="px-4 py-2 rounded-xl text-slate-400 hover:text-white bg-slate-800"
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-xl text-white bg-sky-500 hover:bg-sky-400 font-semibold"
                >
                  Thêm & Khóa
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

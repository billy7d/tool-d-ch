import React, { useState, useEffect } from 'react';
import {
  FolderTree,
  FileText,
  Heading,
  Layers,
  Lock,
  ArrowRight,
  Split,
  Combine,
  CheckCircle2,
  AlertTriangle,
  ChevronRight,
  ChevronDown,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { CanonicalDocument, DocumentNode, Project, NodeType } from '../types';

interface Step3StructureProps {
  project: Project;
  onNext: () => void;
  onRefreshProject: () => void;
}

export const Step3Structure: React.FC<Step3StructureProps> = ({ project, onNext, onRefreshProject }) => {
  const [doc, setDoc] = useState<CanonicalDocument | null>(null);
  const [selectedNode, setSelectedNode] = useState<DocumentNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [locking, setLocking] = useState(false);
  const [expandedChapters, setExpandedChapters] = useState<Record<string, boolean>>({});

  const fetchStructure = async () => {
    try {
      const data = await apiClient.getStructure(project.id);
      setDoc(data);
      if (data.chapters.length > 0) {
        // Expand first chapter by default
        setExpandedChapters({ [data.chapters[0].id]: true });
        if (data.chapters[0].nodes.length > 0) {
          setSelectedNode(data.chapters[0].nodes[0]);
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStructure();
  }, [project.id]);

  const toggleChapter = (chId: string) => {
    setExpandedChapters((prev) => ({ ...prev, [chId]: !prev[chId] }));
  };

  const handleUpdateNodeType = async (newType: NodeType, headingLevel?: number) => {
    if (!selectedNode) return;
    try {
      await apiClient.updateNode(project.id, selectedNode.id, {
        node_type: newType,
        heading_level: headingLevel,
      });
      // Update local state
      fetchStructure();
    } catch (e) {
      alert('Lỗi cập nhật node: ' + e);
    }
  };

  const handleMergeNext = async () => {
    if (!selectedNode) return;
    try {
      await apiClient.mergeNextNode(project.id, selectedNode.id);
      fetchStructure();
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Không thể gộp đoạn văn.');
    }
  };

  const handleConfirmStructure = async () => {
    setLocking(true);
    try {
      await apiClient.confirmStructure(project.id);
      onRefreshProject();
      onNext();
    } catch (e) {
      alert('Lỗi khóa cấu trúc: ' + e);
    } finally {
      setLocking(false);
    }
  };

  return (
    <div className="h-[calc(100vh-7.5rem)] flex flex-col">
      {/* Top action toolbar */}
      <div className="bg-slate-950 border-b border-slate-800 px-6 py-3 flex items-center justify-between shrink-0">
        <div>
          <h2 className="text-sm font-bold text-white flex items-center space-x-2">
            <FolderTree className="w-4 h-4 text-sky-400" />
            <span>Bước 3: Kiểm tra & Khóa cấu trúc tài liệu</span>
          </h2>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Xem lại cấu trúc cây chương hồi và đoạn văn. Xác nhận cấu trúc trước khi dịch.
          </p>
        </div>

        <button
          onClick={handleConfirmStructure}
          disabled={locking}
          className="flex items-center space-x-2 px-5 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white text-xs font-semibold shadow-md shadow-emerald-500/20 transition-all disabled:opacity-50"
        >
          <Lock className="w-3.5 h-3.5" />
          <span>{locking ? 'Đang khóa...' : 'Xác nhận & Khóa cấu trúc (Sang Bước 4)'}</span>
        </button>
      </div>

      {/* Main 2-pane Layout */}
      {loading ? (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">Đang tải cây cấu trúc...</div>
      ) : !doc || doc.chapters.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
          Chưa có cấu trúc. Hãy thực hiện Bước 2 (Phân tích) trước.
        </div>
      ) : (
        <div className="flex-1 flex overflow-hidden">
          {/* Left Pane: Document Tree */}
          <div className="w-1/2 border-r border-slate-800 overflow-y-auto p-4 space-y-3 bg-slate-950/30">
            <div className="text-xs font-semibold text-slate-400 px-2 uppercase tracking-wider">
              Cấu trúc cây tài liệu ({doc.chapters.length} chương - {doc.metadata.total_nodes} phần tử)
            </div>

            {doc.chapters.map((ch) => {
              const isExpanded = !!expandedChapters[ch.id];
              return (
                <div key={ch.id} className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                  <div
                    onClick={() => toggleChapter(ch.id)}
                    className="flex items-center justify-between p-3 cursor-pointer hover:bg-slate-850 transition-colors"
                  >
                    <div className="flex items-center space-x-2 font-semibold text-xs text-white">
                      {isExpanded ? <ChevronDown className="w-3.5 h-3.5 text-slate-400" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-400" />}
                      <span className="text-sky-400">{ch.number ? `Chương ${ch.number}:` : ''}</span>
                      <span className="truncate max-w-sm">{ch.title}</span>
                    </div>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-medium">
                      {ch.nodes.length} đoạn
                    </span>
                  </div>

                  {/* Chapter nodes */}
                  {isExpanded && (
                    <div className="p-2 space-y-1.5 border-t border-slate-800/60 bg-slate-950/40">
                      {ch.nodes.map((node) => {
                        const isSelected = selectedNode?.id === node.id;
                        return (
                          <div
                            key={node.id}
                            onClick={() => setSelectedNode(node)}
                            className={`p-2.5 rounded-lg text-xs cursor-pointer transition-all border ${
                              isSelected
                                ? 'bg-sky-500/10 border-sky-500/50 text-white'
                                : 'bg-slate-900/60 border-slate-800/60 hover:border-slate-700 text-slate-300'
                            }`}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-[10px] uppercase font-bold text-sky-400">
                                {node.type} {node.metadata.heading_level ? `(H${node.metadata.heading_level})` : ''}
                              </span>
                              <span className="text-[10px] text-slate-500">
                                Trang nguồn: {node.source_mapping.source_page_start}
                              </span>
                            </div>
                            <p className="line-clamp-2 text-slate-300 font-serif leading-relaxed">
                              {node.content}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Right Pane: Node Inspector & Source Preview */}
          <div className="w-1/2 overflow-y-auto p-6 space-y-6 bg-slate-900/20">
            {selectedNode ? (
              <div className="space-y-6">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-white">Chỉnh sửa & Kiểm tra phần tử</h3>
                    <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono">
                      ID: {selectedNode.id}
                    </span>
                  </div>

                  {/* Actions Toolbar */}
                  <div className="flex flex-wrap items-center gap-2 p-3 bg-slate-900 border border-slate-800 rounded-xl">
                    <span className="text-xs text-slate-400 font-medium mr-1">Đổi loại:</span>
                    <button
                      onClick={() => handleUpdateNodeType('heading', 1)}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-white border border-slate-700 transition-colors"
                    >
                      Tiêu đề H1
                    </button>
                    <button
                      onClick={() => handleUpdateNodeType('heading', 2)}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-white border border-slate-700 transition-colors"
                    >
                      Tiêu đề H2
                    </button>
                    <button
                      onClick={() => handleUpdateNodeType('paragraph')}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-white border border-slate-700 transition-colors"
                    >
                      Đoạn văn (Paragraph)
                    </button>
                    <button
                      onClick={handleMergeNext}
                      className="flex items-center space-x-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-sky-400 border border-slate-700 transition-colors"
                    >
                      <Combine className="w-3 h-3" />
                      <span>Gộp với đoạn sau</span>
                    </button>
                  </div>
                </div>

                {/* Content Box */}
                <div className="space-y-2">
                  <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Nội dung tiếng Anh trích xuất
                  </label>
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-sm text-slate-200 font-serif leading-relaxed min-h-[160px] whitespace-pre-wrap">
                    {selectedNode.content}
                  </div>
                </div>

                {/* Source Mapping Info */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2 text-xs">
                  <span className="font-semibold text-slate-300 block mb-1">Thông tin tọa độ nguồn (Source Mapping)</span>
                  <div className="grid grid-cols-2 gap-2 text-slate-400">
                    <div>Tài liệu: <strong className="text-white">{selectedNode.source_mapping.source_document || 'source.pdf'}</strong></div>
                    <div>Trang nguồn: <strong className="text-white">{selectedNode.source_mapping.source_page_start}</strong></div>
                    <div>Độ tin cậy: <strong className="text-emerald-400">{(selectedNode.metadata.confidence * 100).toFixed(0)}%</strong></div>
                    <div>Font gốc: <strong className="text-white">{selectedNode.metadata.font_name || 'Standard'} ({selectedNode.metadata.font_size || 11}pt)</strong></div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-20 text-slate-500 text-xs">
                Chọn một đoạn văn hoặc tiêu đề ở cột bên trái để kiểm tra và chỉnh sửa.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

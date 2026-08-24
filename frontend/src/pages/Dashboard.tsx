import React, { useState, useEffect } from 'react';
import {
  Plus,
  BookOpen,
  Download,
  Trash2,
  Upload,
  Calendar,
  FileText,
  Languages,
  CheckCircle2,
  Clock,
  ArrowRight,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { Project, HardwareInfo } from '../types';

interface DashboardProps {
  onSelectProject: (project: Project) => void;
  hardware: HardwareInfo | null;
}

export const Dashboard: React.FC<DashboardProps> = ({ onSelectProject, hardware }) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [restoreLoading, setRestoreLoading] = useState(false);

  const fetchProjects = async () => {
    try {
      const list = await apiClient.getProjects();
      setProjects(list);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) return;
    try {
      const proj = await apiClient.createProject({
        title: newTitle.trim(),
        description: newDesc.trim() || undefined,
      });
      setShowCreateModal(false);
      setNewTitle('');
      setNewDesc('');
      onSelectProject(proj);
    } catch (e) {
      alert('Không thể tạo dự án: ' + e);
    }
  };

  const handleDeleteProject = async (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Bạn có chắc chắn muốn xóa dự án này? Toàn bộ file và bản dịch sẽ bị xóa.')) {
      await apiClient.deleteProject(projectId);
      fetchProjects();
    }
  };

  const handleRestoreFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setRestoreLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await apiClient.restoreProject(formData);
      alert('Phục hồi thành công!');
      fetchProjects();
    } catch (err) {
      alert('Lỗi phục hồi dự án: ' + err);
    } finally {
      setRestoreLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-8 space-y-8">
      {/* Welcome Banner */}
      <div className="bg-gradient-to-r from-sky-900/40 via-slate-900 to-slate-900 p-8 rounded-2xl border border-sky-800/30 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Hệ thống Dịch thuật & Xuất bản Ebook AI Local
          </h1>
          <p className="text-slate-400 text-sm mt-1.5 max-w-2xl leading-relaxed">
            Dịch tài liệu tiếng Anh lên đến 1.000+ trang theo kiến trúc Reflow-first. Bảo toàn cấu trúc, thuật ngữ nhất quán, hỗ trợ chỉnh sửa và xuất bản PDF / EPUB / MOBI offline.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <label className="cursor-pointer flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium border border-slate-700 transition-all">
            <Upload className="w-4 h-4 text-sky-400" />
            <span>{restoreLoading ? 'Đang phục hồi...' : 'Phục hồi (.project.zip)'}</span>
            <input type="file" accept=".zip" onChange={handleRestoreFile} className="hidden" />
          </label>

          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-sm font-semibold shadow-lg shadow-sky-500/20 transition-all"
          >
            <Plus className="w-4 h-4" />
            <span>Tạo dự án mới</span>
          </button>
        </div>
      </div>

      {/* Projects Grid */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <BookOpen className="w-5 h-5 text-sky-400" />
            <span>Danh sách dự án ({projects.length})</span>
          </h2>
        </div>

        {loading ? (
          <div className="text-center py-20 text-slate-500 text-sm">Đang tải danh sách dự án...</div>
        ) : projects.length === 0 ? (
          <div className="text-center py-20 bg-slate-900/40 rounded-2xl border border-dashed border-slate-800">
            <BookOpen className="w-12 h-12 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400 font-medium">Chưa có dự án nào</p>
            <p className="text-slate-500 text-xs mt-1">Bấm nút "Tạo dự án mới" để bắt đầu dịch sách</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((p) => (
              <div
                key={p.id}
                onClick={() => onSelectProject(p)}
                className="group bg-slate-900/70 hover:bg-slate-900 border border-slate-800 hover:border-sky-500/40 rounded-2xl p-6 transition-all duration-200 cursor-pointer flex flex-col justify-between hover:shadow-xl hover:shadow-sky-500/5"
              >
                <div>
                  <div className="flex items-start justify-between gap-3">
                    <h3 className="font-semibold text-white text-base group-hover:text-sky-400 transition-colors line-clamp-1">
                      {p.title}
                    </h3>
                    <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-slate-800 text-sky-400 border border-slate-700/60 font-medium shrink-0">
                      {p.current_stage}
                    </span>
                  </div>

                  {p.description && (
                    <p className="text-slate-400 text-xs mt-2 line-clamp-2 leading-relaxed">
                      {p.description}
                    </p>
                  )}

                  {/* Metadata tags */}
                  <div className="grid grid-cols-2 gap-2.5 mt-5 text-xs text-slate-400 bg-slate-950/40 p-3 rounded-xl border border-slate-800/60">
                    <div className="flex items-center space-x-1.5">
                      <FileText className="w-3.5 h-3.5 text-slate-500" />
                      <span>{p.total_pages || 0} trang ({p.total_nodes || 0} đoạn)</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <Languages className="w-3.5 h-3.5 text-slate-500" />
                      <span>{p.source_language.toUpperCase()} ➔ {p.target_language.toUpperCase()}</span>
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div className="mt-5">
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span className="text-slate-400 font-medium">Tiến độ dịch</span>
                      <span className="text-sky-400 font-bold">{p.progress_percent || 0}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-sky-500 to-emerald-400 transition-all duration-300"
                        style={{ width: `${Math.min(100, Math.max(0, p.progress_percent || 0))}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Card footer */}
                <div className="flex items-center justify-between pt-5 mt-5 border-t border-slate-800/80 text-xs text-slate-500">
                  <div className="flex items-center space-x-1">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{new Date(p.updated_at).toLocaleDateString('vi-VN')}</span>
                  </div>

                  <div className="flex items-center space-x-1">
                    <a
                      href={apiClient.backupProjectUrl(p.id)}
                      onClick={(e) => e.stopPropagation()}
                      title="Sao lưu dự án (.project.zip)"
                      className="p-1.5 hover:text-sky-400 hover:bg-slate-800 rounded-lg transition-colors"
                    >
                      <Download className="w-4 h-4" />
                    </a>
                    <button
                      onClick={(e) => handleDeleteProject(p.id, e)}
                      title="Xóa dự án"
                      className="p-1.5 hover:text-red-400 hover:bg-slate-800 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create Project Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md p-6 shadow-2xl space-y-5">
            <h3 className="text-lg font-bold text-white">Tạo dự án dịch mới</h3>

            <form onSubmit={handleCreateProject} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">Tên tài liệu / Tên sách *</label>
                <input
                  type="text"
                  required
                  placeholder="Ví dụ: The Intelligent Investor"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-sky-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">Mô tả ghi chú (Tùy chọn)</label>
                <textarea
                  placeholder="Ghi chú về bản dịch..."
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  rows={3}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-sky-500 resize-none"
                />
              </div>

              <div className="flex items-center justify-end space-x-3 pt-3">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 transition-colors"
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-xl text-xs font-semibold text-white bg-sky-500 hover:bg-sky-400 transition-all shadow-md shadow-sky-500/20"
                >
                  Tạo dự án
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

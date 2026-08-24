import React, { useState } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, ArrowRight } from 'lucide-react';
import { apiClient } from '../api/client';
import { Project } from '../types';

interface Step1ImportProps {
  project: Project;
  onNext: () => void;
  onRefreshProject: () => void;
}

export const Step1Import: React.FC<Step1ImportProps> = ({ project, onNext, onRefreshProject }) => {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadedDoc, setUploadedDoc] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setError(null);
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await apiClient.uploadDocument(project.id, formData);
      setUploadedDoc(res);
      onRefreshProject();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Không thể tải file lên.');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-8 px-6 space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-bold text-white tracking-tight">Bước 1: Tải tài liệu nguồn</h2>
        <p className="text-slate-400 text-xs leading-relaxed">
          Chọn file tiếng Anh bạn muốn dịch. Hỗ trợ <strong>PDF (Text & Scan), EPUB, DOCX, TXT và Markdown</strong>. Hệ thống tự động phân tích và khôi phục cấu trúc văn bản.
        </p>
      </div>

      {/* Upload Box */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all ${
          dragOver ? 'border-sky-500 bg-sky-500/10' : 'border-slate-800 bg-slate-900/50 hover:border-slate-700'
        }`}
      >
        <UploadCloud className="w-14 h-14 text-sky-400 mx-auto mb-4" />
        <h3 className="text-base font-semibold text-white">Kéo thả file tài liệu vào đây</h3>
        <p className="text-slate-400 text-xs mt-1 mb-5">hoặc duyệt tìm file từ máy tính của bạn</p>

        <label className="cursor-pointer inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold shadow-lg shadow-sky-500/20 transition-all">
          <span>{uploading ? 'Đang tải lên...' : 'Chọn file từ máy tính'}</span>
          <input
            type="file"
            accept=".pdf,.epub,.docx,.txt,.md"
            disabled={uploading}
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
            className="hidden"
          />
        </label>
      </div>

      {error && (
        <div className="flex items-center space-x-2.5 p-4 rounded-xl bg-red-950/40 border border-red-800 text-red-300 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Uploaded File Summary */}
      {uploadedDoc && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center space-x-3 text-emerald-400 text-sm font-semibold">
            <CheckCircle2 className="w-5 h-5" />
            <span>Tải lên và xác thực file thành công</span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80">
              <span className="text-slate-500 block">Tên file</span>
              <span className="text-white font-medium truncate block mt-0.5">{uploadedDoc.filename}</span>
            </div>
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80">
              <span className="text-slate-500 block">Định dạng</span>
              <span className="text-sky-400 font-semibold uppercase block mt-0.5">{uploadedDoc.file_format}</span>
            </div>
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80">
              <span className="text-slate-500 block">Dung lượng</span>
              <span className="text-white font-medium block mt-0.5">{(uploadedDoc.file_size_bytes / (1024 * 1024)).toFixed(2)} MB</span>
            </div>
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80">
              <span className="text-slate-500 block">Ước tính số trang</span>
              <span className="text-white font-medium block mt-0.5">{uploadedDoc.page_count} trang</span>
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <button
              onClick={onNext}
              className="flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold shadow-lg shadow-sky-500/20 transition-all"
            >
              <span>Tiếp tục sang Phân tích & OCR</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

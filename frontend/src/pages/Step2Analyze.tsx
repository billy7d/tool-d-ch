import React, { useState } from 'react';
import { FileSearch, Sparkles, CheckCircle2, ArrowRight, Loader2, Image as ImageIcon } from 'lucide-react';
import { apiClient } from '../api/client';
import { Project } from '../types';

interface Step2AnalyzeProps {
  project: Project;
  onNext: () => void;
  onRefreshProject: () => void;
}

export const Step2Analyze: React.FC<Step2AnalyzeProps> = ({ project, onNext, onRefreshProject }) => {
  const [analyzing, setAnalyzing] = useState(false);
  const [completed, setCompleted] = useState(project.current_stage !== 'IMPORTED');
  const [message, setMessage] = useState<string | null>(null);

  const handleStartAnalysis = async () => {
    setAnalyzing(true);
    setMessage('Đang tiến hành trích xuất văn bản, nhận diện trang scan và tái tạo cấu trúc tài liệu...');
    try {
      await apiClient.startAnalysis(project.id);
      // Wait a moment for background task or listen via SSE
      setTimeout(async () => {
        setAnalyzing(false);
        setCompleted(true);
        onRefreshProject();
        setMessage('Phân tích và tái tạo cấu trúc thành công!');
      }, 3500);
    } catch (e: any) {
      setAnalyzing(false);
      alert('Lỗi phân tích tài liệu: ' + e);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-8 px-6 space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-bold text-white tracking-tight">Bước 2: Phân tích tài liệu & OCR</h2>
        <p className="text-slate-400 text-xs leading-relaxed">
          Hệ thống sẽ bóc tách các khối văn bản (text blocks), phát hiện tiêu đề, chương hồi, sửa lỗi ngắt dòng gạch nối (hyphen repair), lọc bỏ header/footer lặp lại và thực hiện <strong>Selective OCR</strong> cho các trang scan.
        </p>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center space-y-6">
        <div className="w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center mx-auto text-sky-400">
          {analyzing ? <Loader2 className="w-8 h-8 animate-spin" /> : <FileSearch className="w-8 h-8" />}
        </div>

        <div>
          <h3 className="text-lg font-bold text-white">
            {analyzing ? 'Đang phân tích cấu trúc...' : completed ? 'Tài liệu đã sẵn sàng để kiểm tra cấu trúc' : 'Sẵn sàng phân tích'}
          </h3>
          <p className="text-slate-400 text-xs mt-1 max-w-lg mx-auto leading-relaxed">
            {message || 'Bấm nút bên dưới để bắt đầu quá trình trích xuất văn bản và phân tích cấu trúc ngữ nghĩa.'}
          </p>
        </div>

        {/* Feature badges */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 max-w-xl mx-auto text-left text-xs">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
            <span className="font-semibold text-sky-400 block mb-0.5">Selective OCR</span>
            <span className="text-slate-400 text-[11px]">Chỉ OCR những trang scan bị thiếu selectable text</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
            <span className="font-semibold text-emerald-400 block mb-0.5">Paragraph Repair</span>
            <span className="text-slate-400 text-[11px]">Gộp các dòng bị ngắt và sửa từ có dấu gạch nối</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
            <span className="font-semibold text-purple-400 block mb-0.5">Hierarchy Detector</span>
            <span className="text-slate-400 text-[11px]">Nhận diện các cấp tiêu đề H1-H4 và danh sách</span>
          </div>
        </div>

        <div className="flex items-center justify-center space-x-4 pt-4">
          {!completed && (
            <button
              onClick={handleStartAnalysis}
              disabled={analyzing}
              className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold shadow-lg shadow-sky-500/20 transition-all disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>{analyzing ? 'Đang xử lý...' : 'Bắt đầu Phân tích & OCR'}</span>
            </button>
          )}

          {completed && (
            <button
              onClick={onNext}
              className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white text-xs font-semibold shadow-lg shadow-emerald-500/20 transition-all"
            >
              <span>Kiểm tra cấu trúc (Bước 3)</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

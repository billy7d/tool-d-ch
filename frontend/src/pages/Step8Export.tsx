import React, { useState } from 'react';
import {
  Download,
  FileText,
  Book,
  Smartphone,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Sparkles,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { Project, HardwareInfo } from '../types';

interface Step8ExportProps {
  project: Project;
  hardware: HardwareInfo | null;
  onRefreshProject: () => void;
}

export const Step8Export: React.FC<Step8ExportProps> = ({ project, hardware, onRefreshProject }) => {
  const [exportingFormat, setExportingFormat] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [exportedFilename, setExportedFilename] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async (format: 'pdf' | 'epub' | 'mobi') => {
    setError(null);
    setDownloadUrl(null);
    setExportingFormat(format);

    try {
      const res = await apiClient.exportDocument(project.id, {
        format,
        title: project.title,
      });
      setDownloadUrl(res.download_url);
      setExportedFilename(`${project.title}.${format}`);
      onRefreshProject();
    } catch (err: any) {
      setError(err?.response?.data?.detail || `Lỗi xuất file ${format.toUpperCase()}: ${err}`);
    } finally {
      setExportingFormat(null);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-8 px-6 space-y-8">
      <div className="space-y-1">
        <h2 className="text-xl font-bold text-white tracking-tight">Bước 8: Xuất bản tài liệu (Export)</h2>
        <p className="text-slate-400 text-xs leading-relaxed">
          Tất cả định dạng xuất bản đều được tạo từ <strong>Một Canonical Document Model duy nhất</strong>. Đảm bảo văn bản Unicode chuẩn tiếng Việt, hình ảnh chất lượng cao và mục lục tự động điều chỉnh theo số trang mới.
        </p>
      </div>

      {/* Export Format Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* PDF Card */}
        <div className="bg-slate-900 border border-slate-800 hover:border-sky-500/40 rounded-2xl p-6 flex flex-col justify-between space-y-6 transition-all">
          <div className="space-y-3">
            <div className="w-12 h-12 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400">
              <FileText className="w-6 h-6" />
            </div>
            <h3 className="font-bold text-white text-base">Tài liệu PDF Reflow</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Văn bản rõ nét, tìm kiếm được (searchable), nhúng font tiếng Việt Noto Serif/Sans, hỗ trợ bookmarks và dàn trang chuyên nghiệp.
            </p>
          </div>

          <button
            onClick={() => handleExport('pdf')}
            disabled={!!exportingFormat}
            className="w-full flex items-center justify-center space-x-2 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold shadow-lg shadow-sky-500/20 transition-all disabled:opacity-50"
          >
            {exportingFormat === 'pdf' ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Đang render PDF...</span>
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                <span>Xuất file PDF</span>
              </>
            )}
          </button>
        </div>

        {/* EPUB Card */}
        <div className="bg-slate-900 border border-slate-800 hover:border-emerald-500/40 rounded-2xl p-6 flex flex-col justify-between space-y-6 transition-all">
          <div className="space-y-3">
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <Book className="w-6 h-6" />
            </div>
            <h3 className="font-bold text-white text-base">Ebook EPUB (EPUB3)</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Định dạng sách điện tử chuẩn quốc tế, tự co giãn chữ (reflowable) tương thích Apple Books, Google Play Books, Kobo, Boox.
            </p>
          </div>

          <button
            onClick={() => handleExport('epub')}
            disabled={!!exportingFormat}
            className="w-full flex items-center justify-center space-x-2 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white text-xs font-semibold shadow-lg shadow-emerald-500/20 transition-all disabled:opacity-50"
          >
            {exportingFormat === 'epub' ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Đang đóng gói EPUB...</span>
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                <span>Xuất file EPUB</span>
              </>
            )}
          </button>
        </div>

        {/* MOBI Card */}
        <div className="bg-slate-900 border border-slate-800 hover:border-amber-500/40 rounded-2xl p-6 flex flex-col justify-between space-y-6 transition-all">
          <div className="space-y-3">
            <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
              <Smartphone className="w-6 h-6" />
            </div>
            <h3 className="font-bold text-white text-base">MOBI (Kindle Legacy)</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Định dạng dành cho các dòng máy đọc sách Kindle đời cũ (yêu cầu máy có cài đặt Calibre).
            </p>
            {hardware && !hardware.calibre_available && (
              <span className="text-[11px] text-amber-400/90 block">
                Chưa phát hiện Calibre trên máy. Bạn nên ưu tiên xuất định dạng EPUB.
              </span>
            )}
          </div>

          <button
            onClick={() => handleExport('mobi')}
            disabled={!!exportingFormat}
            className="w-full flex items-center justify-center space-x-2 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-semibold shadow-lg shadow-amber-500/20 transition-all disabled:opacity-50"
          >
            {exportingFormat === 'mobi' ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Đang chuyển đổi MOBI...</span>
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                <span>Xuất file MOBI</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error alert */}
      {error && (
        <div className="flex items-start space-x-3 p-4 rounded-2xl bg-red-950/40 border border-red-800 text-red-200 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="leading-relaxed">{error}</div>
        </div>
      )}

      {/* Download Success Box */}
      {downloadUrl && (
        <div className="bg-slate-900 border border-emerald-500/40 rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-4 shadow-xl shadow-emerald-500/5">
          <div className="flex items-center space-x-3 text-emerald-400 text-sm font-semibold">
            <CheckCircle2 className="w-6 h-6 shrink-0" />
            <div>
              <span className="block text-white">Xuất bản tài liệu thành công!</span>
              <span className="text-xs text-slate-400 font-normal">File đã được kiểm định tính toàn vẹn và sẵn sàng để tải về máy.</span>
            </div>
          </div>

          <a
            href={downloadUrl}
            download={exportedFilename || 'document'}
            className="flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white text-xs font-semibold shadow-lg shadow-emerald-500/20 transition-all shrink-0"
          >
            <Download className="w-4 h-4" />
            <span>Tải xuống {exportedFilename}</span>
          </a>
        </div>
      )}
    </div>
  );
};

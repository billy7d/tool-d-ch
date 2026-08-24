import React from 'react';
import {
  Sliders,
  Cpu,
  HardDrive,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Folder,
  Shield,
  Sparkles,
} from 'lucide-react';
import { HardwareInfo } from '../types';

interface SettingsPageProps {
  hardware: HardwareInfo | null;
  onRefreshHardware: () => void;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({ hardware, onRefreshHardware }) => {
  return (
    <div className="max-w-4xl mx-auto py-8 px-6 space-y-8">
      <div className="space-y-1">
        <h2 className="text-xl font-bold text-white tracking-tight">Cài đặt hệ thống & Trạng thái phần cứng</h2>
        <p className="text-slate-400 text-xs leading-relaxed">
          Kiểm tra tài nguyên máy tính, kết nối Ollama local và các công cụ bổ trợ xuất bản (Tesseract OCR, Calibre).
        </p>
      </div>

      {hardware ? (
        <div className="space-y-6">
          {/* Hardware Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* CPU & RAM */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-sky-400" />
                <span>Vi xử lý & Bộ nhớ RAM</span>
              </h3>

              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Tên CPU:</span>
                  <span className="text-white font-medium">{hardware.cpu_name}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Số luồng CPU:</span>
                  <span className="text-white font-medium">{hardware.cpu_cores} Cores / Threads</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Tổng dung lượng RAM:</span>
                  <span className="text-white font-medium">{hardware.ram_total_gb} GB</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-slate-400">RAM khả dụng:</span>
                  <span className="text-emerald-400 font-medium">{hardware.ram_available_gb} GB</span>
                </div>
              </div>
            </div>

            {/* GPU & VRAM */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <Sparkles className="w-4 h-4 text-emerald-400" />
                <span>Card đồ họa & VRAM</span>
              </h3>

              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">GPU:</span>
                  <span className="text-white font-medium">{hardware.gpu_name || 'Không phát hiện card rời'}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Bộ nhớ VRAM:</span>
                  <span className="text-white font-medium">{hardware.vram_total_gb ? `${hardware.vram_total_gb} GB` : 'N/A'}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Hỗ trợ NVIDIA CUDA:</span>
                  <span className={hardware.cuda_available ? 'text-emerald-400 font-medium' : 'text-slate-500'}>
                    {hardware.cuda_available ? 'Khả dụng' : 'Không'}
                  </span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-slate-400">Cấu hình đề xuất:</span>
                  <span className="text-sky-400 font-semibold">{hardware.recommended_preset}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Dependencies Status */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <Shield className="w-4 h-4 text-purple-400" />
              <span>Trạng thái dịch vụ & Công cụ cục bộ</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white">Dịch vụ Ollama</span>
                  {hardware.ollama_running ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-400" />
                  )}
                </div>
                <p className="text-slate-400 text-[11px]">
                  {hardware.ollama_running ? `Online (${hardware.installed_models.length} model đã cài)` : 'Chưa chạy. Hãy mở Ollama trên Windows.'}
                </p>
              </div>

              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white">Tesseract OCR</span>
                  {hardware.tesseract_available ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <span className="text-[10px] text-slate-500 font-medium">Tùy chọn</span>
                  )}
                </div>
                <p className="text-slate-400 text-[11px]">
                  {hardware.tesseract_available ? 'Đã cài đặt' : 'Chưa cài đặt (Chỉ cần khi dịch sách scan)'}
                </p>
              </div>

              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white">Calibre (ebook-convert)</span>
                  {hardware.calibre_available ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <span className="text-[10px] text-slate-500 font-medium">Tùy chọn</span>
                  )}
                </div>
                <p className="text-slate-400 text-[11px]">
                  {hardware.calibre_available ? 'Đã cài đặt' : 'Chưa cài đặt (Chỉ cần khi xuất file MOBI)'}
                </p>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={onRefreshHardware}
                className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Quét lại phần cứng & Công cụ</span>
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-20 text-slate-500 text-sm">Đang tải thông tin phần cứng...</div>
      )}
    </div>
  );
};

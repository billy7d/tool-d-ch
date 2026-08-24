import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  Square,
  RefreshCw,
  ArrowRight,
  AlertCircle,
  Sparkles,
  Terminal,
  ExternalLink,
  CheckCircle2,
  Cpu,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { Project } from '../types';

interface Step5TranslateProps {
  project: Project;
  onNext: () => void;
  onRefreshProject: () => void;
}

export const Step5Translate: React.FC<Step5TranslateProps> = ({ project, onNext, onRefreshProject }) => {
  const [status, setStatus] = useState<string>(project.current_stage);
  const [stats, setStats] = useState({
    total_nodes: project.total_nodes || 0,
    translated_nodes: project.translated_nodes || 0,
    failed_nodes: 0,
    needs_review_nodes: 0,
    progress_percent: project.progress_percent || 0,
    current_chapter_title: '',
    current_chunk_id: '',
    context_mode: 'CONTEXTUAL_BALANCED',
    retry_count: 0,
    quality_state: 'READY',
  });
  const [ollamaOnline, setOllamaOnline] = useState<boolean>(true);
  const prevOllamaRef = useRef<boolean>(true);

  const fetchStatus = async () => {
    try {
      const res = await apiClient.getTranslationStatus(project.id);
      setStatus(res.status);
      setStats((prev) => ({
        ...prev,
        total_nodes: res.total_nodes,
        translated_nodes: res.translated_nodes,
        failed_nodes: res.failed_nodes,
        needs_review_nodes: res.needs_review_nodes,
        progress_percent: res.progress_percent,
        current_chapter_title: res.current_chapter_title || prev.current_chapter_title,
        current_chunk_id: res.current_chunk_id || prev.current_chunk_id,
        context_mode: res.context_mode || prev.context_mode,
        retry_count: res.retry_count ?? prev.retry_count,
        quality_state: res.quality_state || prev.quality_state,
      }));

      const hw = await apiClient.getHardwareInfo();
      setOllamaOnline(hw.ollama_running);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 2000);
    return () => clearInterval(timer);
  }, [project.id]);

  const handleStart = async () => {
    try {
      await apiClient.startTranslation(project.id, {});
      fetchStatus();
      onRefreshProject();
    } catch (e) {
      alert('Lỗi bắt đầu dịch: ' + e);
    }
  };

  const handlePause = async () => {
    try {
      await apiClient.pauseTranslation(project.id);
      fetchStatus();
    } catch (e) {
      alert('Lỗi: ' + e);
    }
  };

  const handleResume = async () => {
    try {
      await apiClient.resumeTranslation(project.id);
      fetchStatus();
    } catch (e) {
      alert('Lỗi: ' + e);
    }
  };

  const handleStop = async () => {
    try {
      await apiClient.stopTranslation(project.id);
      fetchStatus();
    } catch (e) {
      alert('Lỗi: ' + e);
    }
  };

  const handleRetryFailed = async () => {
    try {
      await apiClient.retryFailedTranslation(project.id);
      fetchStatus();
    } catch (e) {
      alert('Lỗi: ' + e);
    }
  };

  const isRunning = status === 'RUNNING' || status === 'TRANSLATING';
  const isPaused = status === 'PAUSED';
  const isCompleted = stats.translated_nodes >= stats.total_nodes && stats.total_nodes > 0;
  const isAutoHealing = isRunning && stats.failed_nodes > 0 && (stats.translated_nodes + stats.failed_nodes >= stats.total_nodes);

  // Auto-retry when Ollama transitions from Offline to Online without user click
  useEffect(() => {
    if (!prevOllamaRef.current && ollamaOnline && stats.failed_nodes > 0 && !isRunning) {
      console.log('Ollama is back online! Automatically auto-healing failed nodes...');
      handleRetryFailed();
    }
    prevOllamaRef.current = ollamaOnline;
  }, [ollamaOnline, stats.failed_nodes, isRunning]);

  return (
    <div className="max-w-4xl mx-auto py-8 px-6 space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-bold text-white tracking-tight">Bước 5: Giám sát tiến trình dịch</h2>
        <p className="text-slate-400 text-xs leading-relaxed">
          Hệ thống dịch tự động với cơ chế <strong>Auto-Self-Healing</strong> & <strong>Single-Node Fallback</strong>. Mỗi đoạn văn bản được lưu tức thì (Immediate SQLite Commit) và tự động sửa chữa để đảm bảo 100% tài liệu không bị gián đoạn.
        </p>
      </div>

      {/* Main Status Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 space-y-6">
        {/* Progress header */}
        <div className="flex items-center justify-between">
          <div>
            <span className="text-xs text-slate-400 font-medium">Trạng thái hiện tại:</span>
            <div className="flex items-center space-x-2 mt-0.5">
              <span className={`w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-sky-400 animate-ping' : isPaused ? 'bg-amber-400' : isCompleted ? 'bg-emerald-400' : 'bg-slate-500'}`} />
              <h3 className="text-lg font-bold text-white uppercase tracking-wide">
                {isAutoHealing
                  ? 'Đang tự động vá đoạn lỗi (Auto-Healing)...'
                  : isRunning
                  ? 'Đang dịch...'
                  : isPaused
                  ? 'Đã tạm dừng'
                  : isCompleted
                  ? 'Hoàn thành bản dịch (100%)'
                  : 'Chưa bắt đầu'}
              </h3>
            </div>
          </div>

          <div className="text-right">
            <span className="text-2xl font-black text-sky-400">{stats.progress_percent}%</span>
            <span className="text-xs text-slate-500 block">Đã hoàn thành</span>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[11px]">
          <div className="bg-sky-500/10 border border-sky-500/20 rounded-xl p-3"><span className="text-slate-500 block">Chương hiện tại</span><span className="text-sky-300 block mt-1 truncate">{stats.current_chapter_title || 'Đang chuẩn bị'}</span></div>
          <div className="bg-sky-500/10 border border-sky-500/20 rounded-xl p-3"><span className="text-slate-500 block">Chunk</span><span className="text-sky-300 block mt-1">{stats.current_chunk_id || '—'}</span></div>
          <div className="bg-sky-500/10 border border-sky-500/20 rounded-xl p-3"><span className="text-slate-500 block">Context engine</span><span className="text-sky-300 block mt-1">{stats.context_mode}</span></div>
          <div className="bg-sky-500/10 border border-sky-500/20 rounded-xl p-3"><span className="text-slate-500 block">Quality / Retry</span><span className="text-sky-300 block mt-1">{stats.quality_state} · {stats.retry_count}</span></div>
        </div>

        {/* Large Progress bar */}
        <div className="w-full h-4 rounded-full bg-slate-950 border border-slate-800 overflow-hidden p-0.5">
          <div
            className="h-full rounded-full bg-gradient-to-r from-sky-500 via-indigo-500 to-emerald-400 transition-all duration-300 shadow-sm"
            style={{ width: `${Math.min(100, Math.max(0, stats.progress_percent))}%` }}
          />
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80">
            <span className="text-slate-500 block">Tổng số đoạn văn</span>
            <span className="text-white text-base font-bold mt-1 block">{stats.total_nodes}</span>
          </div>
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80">
            <span className="text-slate-500 block">Đã dịch thành công</span>
            <span className="text-emerald-400 text-base font-bold mt-1 block">{stats.translated_nodes}</span>
          </div>
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80">
            <span className="text-slate-500 block">Cần kiểm tra lại</span>
            <span className="text-amber-400 text-base font-bold mt-1 block">{stats.needs_review_nodes}</span>
          </div>
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80">
            <span className="text-slate-500 block">Đoạn bị lỗi (Failed)</span>
            <span className={`text-base font-bold mt-1 block ${stats.failed_nodes > 0 ? 'text-amber-400' : 'text-slate-400'}`}>
              {stats.failed_nodes}
            </span>
          </div>
        </div>

        {/* Real Ollama Offline Guide Box (Only shown if Ollama is actually offline) */}
        {!ollamaOnline && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-5 space-y-3">
            <div className="flex items-center space-x-2.5 text-amber-400">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <h4 className="text-xs font-bold uppercase tracking-wide">
                Dịch vụ Ollama chưa chạy trên Windows
              </h4>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Hệ thống sử dụng mô hình AI cục bộ qua dịch vụ <strong>Ollama</strong> (cổng mặc định <code>http://localhost:11434</code>). Khi bạn khởi động Ollama, hệ thống sẽ <strong>tự động nhận diện và tiếp tục dịch ngay</strong> mà không cần bấm thêm thao tác nào.
            </p>

            <div className="bg-slate-950/80 rounded-lg p-3.5 border border-slate-800/80 space-y-2 text-xs">
              <span className="text-slate-400 font-semibold flex items-center space-x-1.5">
                <Terminal className="w-3.5 h-3.5 text-sky-400" />
                <span>Cách khởi động Ollama trên Windows:</span>
              </span>
              <ol className="list-decimal list-inside space-y-1.5 text-slate-300 text-[11px] pl-1">
                <li>
                  Tải và cài đặt Ollama từ trang chủ:{' '}
                  <a
                    href="https://ollama.com/download/windows"
                    target="_blank"
                    rel="noreferrer"
                    className="text-sky-400 underline inline-flex items-center space-x-0.5"
                  >
                    <span>ollama.com/download/windows</span>
                    <ExternalLink className="w-3 h-3 ml-0.5" />
                  </a>
                </li>
                <li>Mở ứng dụng Ollama trên máy tính hoặc mở Command Prompt (cmd) và gõ lệnh tải mô hình:</li>
                <pre className="bg-slate-900 px-2.5 py-1.5 rounded border border-slate-800 text-emerald-400 font-mono text-[11px]">
                  ollama run qwen2.5:7b
                </pre>
              </ol>
            </div>
          </div>
        )}

        {/* Auto-Healing Active Banner (When Ollama is online and auto-repairing failed nodes) */}
        {ollamaOnline && isRunning && stats.failed_nodes > 0 && (
          <div className="bg-sky-500/10 border border-sky-500/30 rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3 text-sky-400">
              <RefreshCw className="w-5 h-5 animate-spin shrink-0" />
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wide">
                  Đang tự động vá lỗi ({stats.failed_nodes} đoạn văn bản)
                </h4>
                <p className="text-[11px] text-slate-300 mt-0.5">
                  Cơ chế Single-Node Fallback và Auto-Healing đang tự động dịch độc lập từng đoạn mà bạn không cần thao tác click.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Ollama Online with Completed State */}
        {isCompleted && (
          <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 flex items-center space-x-3 text-emerald-400">
            <CheckCircle2 className="w-5 h-5 shrink-0" />
            <div className="text-xs">
              <span className="font-bold uppercase tracking-wide block">Dịch hoàn tất 100%</span>
              <span className="text-slate-300 text-[11px]">Toàn bộ các đoạn văn bản đã được dịch và lưu trữ an toàn trong cơ sở dữ liệu.</span>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-slate-800">
          <div className="flex items-center space-x-3">
            {!isRunning && !isPaused && (
              <button
                onClick={handleStart}
                className="flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold shadow-lg shadow-sky-500/20 transition-all"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>{stats.translated_nodes > 0 ? 'Tiếp tục dịch' : 'Bắt đầu dịch'}</span>
              </button>
            )}

            {isRunning && (
              <button
                onClick={handlePause}
                className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-semibold transition-all"
              >
                <Pause className="w-4 h-4 fill-slate-950" />
                <span>Tạm dừng (Pause)</span>
              </button>
            )}

            {isPaused && (
              <button
                onClick={handleResume}
                className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold transition-all"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>Tiếp tục (Resume)</span>
              </button>
            )}

            {(isRunning || isPaused) && (
              <button
                onClick={handleStop}
                className="flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition-all"
              >
                <Square className="w-3.5 h-3.5" />
                <span>Dừng (Stop)</span>
              </button>
            )}

            {stats.failed_nodes > 0 && !isRunning && (
              <button
                onClick={handleRetryFailed}
                className="flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 text-xs font-medium border border-sky-500/40 transition-all"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Dịch lại đoạn lỗi ({stats.failed_nodes})</span>
              </button>
            )}
          </div>

          <button
            onClick={onNext}
            className="flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold border border-slate-700 transition-all"
          >
            <span>Sang QA & Chỉnh sửa (Bước 6)</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

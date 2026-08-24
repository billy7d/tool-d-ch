import React from 'react';
import { Cpu, HardDrive, RefreshCw, Layers, Sparkles } from 'lucide-react';
import { Project, HardwareInfo } from '../types';

interface HeaderProps {
  currentProject: Project | null;
  hardware: HardwareInfo | null;
  onRefreshHardware: () => void;
  onBackToDashboard: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentProject,
  hardware,
  onRefreshHardware,
  onBackToDashboard,
}) => {
  return (
    <header className="h-14 bg-slate-950 border-b border-slate-800 px-6 flex items-center justify-between z-30 sticky top-0">
      <div className="flex items-center space-x-4">
        <button
          onClick={onBackToDashboard}
          className="flex items-center space-x-2.5 font-bold text-sky-400 hover:text-sky-300 transition-colors"
        >
          <Layers className="w-5 h-5 text-sky-500" />
          <span className="text-base tracking-tight text-white">Local AI Publisher</span>
        </button>

        {currentProject && (
          <div className="flex items-center space-x-2 pl-4 border-l border-slate-800 text-sm">
            <span className="text-slate-400 font-medium truncate max-w-xs">{currentProject.title}</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              {currentProject.current_stage}
            </span>
          </div>
        )}
      </div>

      <div className="flex items-center space-x-3 text-xs">
        {/* Hardware Status */}
        {hardware && (
          <div className="flex items-center space-x-3 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-300">
            <div className="flex items-center space-x-1.5" title={`CPU: ${hardware.cpu_name}`}>
              <Cpu className="w-3.5 h-3.5 text-sky-400" />
              <span>{hardware.cpu_cores} Cores</span>
            </div>

            {hardware.gpu_name && (
              <div className="flex items-center space-x-1.5" title={`GPU: ${hardware.gpu_name} (${hardware.vram_total_gb}GB VRAM)`}>
                <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                <span className="truncate max-w-[120px]">{hardware.gpu_name}</span>
              </div>
            )}

            <div className="flex items-center space-x-1.5">
              <span className={`w-2 h-2 rounded-full ${hardware.ollama_running ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
              <span>Ollama: {hardware.ollama_running ? 'Online' : 'Offline'}</span>
            </div>

            <button
              onClick={onRefreshHardware}
              className="text-slate-400 hover:text-white transition-colors ml-1"
              title="Làm mới trạng thái phần cứng"
            >
              <RefreshCw className="w-3 h-3" />
            </button>
          </div>
        )}
      </div>
    </header>
  );
};

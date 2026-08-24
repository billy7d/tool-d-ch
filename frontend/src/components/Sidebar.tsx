import React from 'react';
import {
  LayoutDashboard,
  UploadCloud,
  FileSearch,
  FolderTree,
  Sliders,
  Languages,
  CheckSquare,
  Palette,
  Download,
  Settings as SettingsIcon,
} from 'lucide-react';
import { Project } from '../types';
import { WORKFLOW_STEPS } from './WorkflowStepper';

interface SidebarProps {
  currentProject: Project | null;
  currentStep: number;
  onSelectStep: (step: number) => void;
  activeView: 'dashboard' | 'workflow' | 'settings';
  onSelectView: (view: 'dashboard' | 'workflow' | 'settings') => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentProject,
  currentStep,
  onSelectStep,
  activeView,
  onSelectView,
}) => {
  return (
    <aside className="w-56 bg-slate-950 border-r border-slate-800 flex flex-col justify-between py-4 shrink-0">
      <div className="space-y-6">
        {/* Navigation Items */}
        <div className="px-3 space-y-1">
          <button
            onClick={() => onSelectView('dashboard')}
            className={`w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs font-medium transition-all ${
              activeView === 'dashboard'
                ? 'bg-sky-500/20 text-sky-400 font-semibold'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <LayoutDashboard className="w-4 h-4" />
            <span>Danh sách dự án</span>
          </button>
        </div>

        {/* Workflow Steps (when a project is open) */}
        {currentProject && (
          <div className="px-3 space-y-1">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-3 mb-2 block">
              Các bước quy trình
            </span>

            {WORKFLOW_STEPS.map((step) => {
              const Icon = step.icon;
              const isActive = activeView === 'workflow' && currentStep === step.id;

              return (
                <button
                  key={step.id}
                  onClick={() => {
                    onSelectView('workflow');
                    onSelectStep(step.id);
                  }}
                  className={`w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-sky-500/20 text-sky-400 font-semibold shadow-sm'
                      : 'text-slate-400 hover:text-white hover:bg-slate-900'
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span className="truncate">{step.title}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer Settings Button */}
      <div className="px-3">
        <button
          onClick={() => onSelectView('settings')}
          className={`w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs font-medium transition-all ${
            activeView === 'settings'
              ? 'bg-sky-500/20 text-sky-400 font-semibold'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <SettingsIcon className="w-4 h-4" />
          <span>Cài đặt & Phần cứng</span>
        </button>
      </div>
    </aside>
  );
};

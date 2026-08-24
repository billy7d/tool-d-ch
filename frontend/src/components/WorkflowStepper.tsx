import React from 'react';
import {
  UploadCloud,
  FileSearch,
  FolderTree,
  Sliders,
  Languages,
  CheckSquare,
  Palette,
  Download,
  CheckCircle2,
} from 'lucide-react';

export const WORKFLOW_STEPS = [
  { id: 1, title: 'Import', icon: UploadCloud },
  { id: 2, title: 'Phân tích / OCR', icon: FileSearch },
  { id: 3, title: 'Kiểm tra cấu trúc', icon: FolderTree },
  { id: 4, title: 'Thiết lập dịch', icon: Sliders },
  { id: 5, title: 'Dịch', icon: Languages },
  { id: 6, title: 'QA & Chỉnh sửa', icon: CheckSquare },
  { id: 7, title: 'Layout & Preview', icon: Palette },
  { id: 8, title: 'Export', icon: Download },
];

interface WorkflowStepperProps {
  currentStep: number;
  onSelectStep: (stepId: number) => void;
  maxAccessibleStep?: number;
}

export const WorkflowStepper: React.FC<WorkflowStepperProps> = ({
  currentStep,
  onSelectStep,
  maxAccessibleStep = 8,
}) => {
  return (
    <div className="w-full bg-slate-950/60 border-b border-slate-800/80 px-6 py-2.5 flex items-center justify-between overflow-x-auto">
      <div className="flex items-center space-x-1 min-w-max mx-auto">
        {WORKFLOW_STEPS.map((step, idx) => {
          const Icon = step.icon;
          const isActive = currentStep === step.id;
          const isCompleted = currentStep > step.id;
          const isAccessible = step.id <= maxAccessibleStep;

          return (
            <React.Fragment key={step.id}>
              <button
                onClick={() => isAccessible && onSelectStep(step.id)}
                disabled={!isAccessible}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-sky-500/20 text-sky-400 border border-sky-500/40 shadow-sm shadow-sky-500/10'
                    : isCompleted
                    ? 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                    : 'text-slate-500 cursor-not-allowed'
                }`}
              >
                <div
                  className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                    isActive
                      ? 'bg-sky-500 text-white'
                      : isCompleted
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {isCompleted ? <CheckCircle2 className="w-3.5 h-3.5" /> : step.id}
                </div>
                <span>{step.title}</span>
              </button>

              {idx < WORKFLOW_STEPS.length - 1 && (
                <div className={`w-4 h-0.5 ${isCompleted ? 'bg-emerald-500/30' : 'bg-slate-800'}`} />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};

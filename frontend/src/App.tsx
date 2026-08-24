import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { WorkflowStepper } from './components/WorkflowStepper';
import { Dashboard } from './pages/Dashboard';
import { Step1Import } from './pages/Step1Import';
import { Step2Analyze } from './pages/Step2Analyze';
import { Step3Structure } from './pages/Step3Structure';
import { Step4Setup } from './pages/Step4Setup';
import { Step5Translate } from './pages/Step5Translate';
import { Step6QAEditor } from './pages/Step6QAEditor';
import { Step7Layout } from './pages/Step7Layout';
import { Step8Export } from './pages/Step8Export';
import { SettingsPage } from './pages/SettingsPage';
import { apiClient } from './api/client';
import { Project, HardwareInfo } from './types';

export const App: React.FC = () => {
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [activeView, setActiveView] = useState<'dashboard' | 'workflow' | 'settings'>('dashboard');
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);

  const fetchHardware = async () => {
    try {
      const data = await apiClient.getHardwareInfo();
      setHardware(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchHardware();
  }, []);

  // Listen to SSE live events from backend
  useEffect(() => {
    const eventSource = new EventSource('/api/events');
    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'TRANSLATION_PROGRESS' || payload.type === 'TRANSLATION_COMPLETED') {
          if (currentProject && payload.data?.project_id === currentProject.id) {
            refreshCurrentProject();
          }
        }
      } catch (e) {
        // Heartbeat or ping
      }
    };

    return () => {
      eventSource.close();
    };
  }, [currentProject?.id]);

  const refreshCurrentProject = async () => {
    if (!currentProject) return;
    try {
      const updated = await apiClient.getProject(currentProject.id);
      setCurrentProject(updated);
    } catch (e) {
      console.error(e);
    }
  };

  const handleSelectProject = (project: Project) => {
    setCurrentProject(project);
    setActiveView('workflow');
    // Map stage to step
    const stageMap: Record<string, number> = {
      IMPORTED: 1,
      ANALYZED: 2,
      STRUCTURE_READY: 3,
      STRUCTURE_CONFIRMED: 4,
      TRANSLATION_CONFIGURED: 5,
      TRANSLATING: 5,
      TRANSLATED: 6,
      QA: 6,
      LAYOUT: 7,
      EXPORTED: 8,
    };
    setCurrentStep(stageMap[project.current_stage] || 1);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      {/* Top Header */}
      <Header
        currentProject={currentProject}
        hardware={hardware}
        onRefreshHardware={fetchHardware}
        onBackToDashboard={() => setActiveView('dashboard')}
      />

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          currentProject={currentProject}
          currentStep={currentStep}
          onSelectStep={(step) => setCurrentStep(step)}
          activeView={activeView}
          onSelectView={(view) => setActiveView(view)}
        />

        {/* Content Area */}
        <main className="flex-1 flex flex-col overflow-y-auto bg-slate-950/20">
          {activeView === 'workflow' && currentProject && (
            <WorkflowStepper
              currentStep={currentStep}
              onSelectStep={(step) => setCurrentStep(step)}
            />
          )}

          <div className="flex-1">
            {activeView === 'dashboard' && (
              <Dashboard
                onSelectProject={handleSelectProject}
                hardware={hardware}
              />
            )}

            {activeView === 'settings' && (
              <SettingsPage
                hardware={hardware}
                onRefreshHardware={fetchHardware}
              />
            )}

            {activeView === 'workflow' && currentProject && (
              <>
                {currentStep === 1 && (
                  <Step1Import
                    project={currentProject}
                    onNext={() => setCurrentStep(2)}
                    onRefreshProject={refreshCurrentProject}
                  />
                )}
                {currentStep === 2 && (
                  <Step2Analyze
                    project={currentProject}
                    onNext={() => setCurrentStep(3)}
                    onRefreshProject={refreshCurrentProject}
                  />
                )}
                {currentStep === 3 && (
                  <Step3Structure
                    project={currentProject}
                    onNext={() => setCurrentStep(4)}
                    onRefreshProject={refreshCurrentProject}
                  />
                )}
                {currentStep === 4 && (
                  <Step4Setup
                    project={currentProject}
                    hardware={hardware}
                    onNext={() => setCurrentStep(5)}
                    onRefreshProject={refreshCurrentProject}
                  />
                )}
                {currentStep === 5 && (
                  <Step5Translate
                    project={currentProject}
                    onNext={() => setCurrentStep(6)}
                    onRefreshProject={refreshCurrentProject}
                  />
                )}
                {currentStep === 6 && (
                  <Step6QAEditor
                    project={currentProject}
                    onNext={() => setCurrentStep(7)}
                    onRefreshProject={refreshCurrentProject}
                  />
                )}
                {currentStep === 7 && (
                  <Step7Layout
                    project={currentProject}
                    onNext={() => setCurrentStep(8)}
                    onRefreshProject={refreshCurrentProject}
                  />
                )}
                {currentStep === 8 && (
                  <Step8Export
                    project={currentProject}
                    hardware={hardware}
                    onRefreshProject={refreshCurrentProject}
                  />
                )}
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

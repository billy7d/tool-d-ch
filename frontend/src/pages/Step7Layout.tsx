import React, { useState, useEffect } from 'react';
import {
  Palette,
  Eye,
  Type,
  BookOpen,
  Layout,
  ArrowRight,
  RefreshCw,
  Sliders,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { Project, LayoutProfile } from '../types';

interface Step7LayoutProps {
  project: Project;
  onNext: () => void;
  onRefreshProject: () => void;
}

export const Step7Layout: React.FC<Step7LayoutProps> = ({ project, onNext, onRefreshProject }) => {
  const [profiles, setProfiles] = useState<LayoutProfile[]>([]);
  const [currentProfile, setCurrentProfile] = useState<LayoutProfile | null>(null);
  const [previewHtml, setPreviewHtml] = useState<string>('');
  const [loadingPreview, setLoadingPreview] = useState(true);
  const [sampleType, setSampleType] = useState('representative');

  const fetchProfilesAndPreview = async () => {
    try {
      const list = await apiClient.getLayoutProfiles(project.id);
      setProfiles(list);
      const active = list.find((p) => p.is_default) || list[0];
      setCurrentProfile(active);
      if (active) {
        loadPreview(active.id, sampleType);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchProfilesAndPreview();
  }, [project.id]);

  const loadPreview = async (profileId?: string, sample: string = 'representative') => {
    setLoadingPreview(true);
    try {
      const html = await apiClient.getPreviewHtml(project.id, {
        layout_profile_id: profileId || currentProfile?.id,
        sample_type: sample,
      });
      setPreviewHtml(html);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingPreview(false);
    }
  };

  const handleApplyPreset = async (presetName: string) => {
    if (!currentProfile) return;
    const presets: Record<string, Partial<LayoutProfile>> = {
      'Classic Book': {
        page_size: 'A5',
        page_width_mm: 148,
        page_height_mm: 210,
        body_font: 'Noto Serif',
        heading_font: 'Noto Serif',
        body_font_size_pt: 11,
        line_height: 1.5,
        first_line_indent_mm: 5,
        text_alignment: 'justify',
      },
      'Modern Book': {
        page_size: 'A5',
        page_width_mm: 148,
        page_height_mm: 210,
        body_font: 'Noto Sans',
        heading_font: 'Noto Sans',
        body_font_size_pt: 10.5,
        line_height: 1.55,
        first_line_indent_mm: 0,
        text_alignment: 'left',
      },
      'Academic': {
        page_size: 'A4',
        page_width_mm: 210,
        page_height_mm: 297,
        body_font: 'Noto Serif',
        heading_font: 'Noto Serif',
        body_font_size_pt: 10,
        line_height: 1.4,
        first_line_indent_mm: 6,
        text_alignment: 'justify',
      },
      'Technical Manual': {
        page_size: 'A4',
        page_width_mm: 210,
        page_height_mm: 297,
        body_font: 'Noto Sans',
        heading_font: 'Noto Sans',
        body_font_size_pt: 10.5,
        line_height: 1.5,
        first_line_indent_mm: 0,
        text_alignment: 'left',
      },
    };

    const targetPreset = presets[presetName];
    if (!targetPreset) return;

    try {
      const updated = await apiClient.updateLayoutProfile(project.id, currentProfile.id, targetPreset);
      setCurrentProfile(updated);
      loadPreview(updated.id, sampleType);
    } catch (e) {
      alert('Lỗi áp dụng mẫu: ' + e);
    }
  };

  const handleUpdateField = async (field: keyof LayoutProfile, value: any) => {
    if (!currentProfile) return;
    const updated = { ...currentProfile, [field]: value };
    setCurrentProfile(updated);
    try {
      await apiClient.updateLayoutProfile(project.id, currentProfile.id, { [field]: value });
      loadPreview(currentProfile.id, sampleType);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="h-[calc(100vh-7.5rem)] flex flex-col">
      {/* Top bar */}
      <div className="bg-slate-950 border-b border-slate-800 px-6 py-2.5 flex items-center justify-between shrink-0">
        <div>
          <h2 className="text-sm font-bold text-white flex items-center space-x-2">
            <Palette className="w-4 h-4 text-sky-400" />
            <span>Bước 7: Định dạng Typography & Smart Preview Reflow</span>
          </h2>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Tách biệt hoàn toàn nội dung và cách trình bày. Thay đổi font chữ và căn lề cập nhật tức thì mà không cần dịch lại.
          </p>
        </div>

        <button
          onClick={onNext}
          className="flex items-center space-x-2 px-6 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white text-xs font-semibold shadow-md shadow-emerald-500/20 transition-all"
        >
          <span>Sang Xuất bản (Bước 8)</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Main 2-column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Layout Controls */}
        <div className="w-96 border-r border-slate-800 overflow-y-auto p-5 space-y-6 bg-slate-950/40 shrink-0 text-xs">
          {/* Preset Buttons */}
          <div className="space-y-2">
            <span className="font-semibold text-slate-300 block uppercase tracking-wider text-[11px]">
              Mẫu phong cách (Presets)
            </span>
            <div className="grid grid-cols-2 gap-2">
              {['Classic Book', 'Modern Book', 'Academic', 'Technical Manual'].map((pname) => (
                <button
                  key={pname}
                  onClick={() => handleApplyPreset(pname)}
                  className="p-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-sky-500/50 text-slate-300 font-medium text-left transition-all"
                >
                  {pname}
                </button>
              ))}
            </div>
          </div>

          {currentProfile && (
            <div className="space-y-4 pt-2 border-t border-slate-800">
              <span className="font-semibold text-slate-300 block uppercase tracking-wider text-[11px]">
                Tùy chỉnh chi tiết
              </span>

              {/* Page Size */}
              <div>
                <label className="block text-slate-400 mb-1">Khổ trang (Page Size)</label>
                <select
                  value={currentProfile.page_size}
                  onChange={(e) => {
                    const val = e.target.value;
                    let w = 148, h = 210;
                    if (val === 'A4') { w = 210; h = 297; }
                    else if (val === 'LETTER') { w = 215.9; h = 279.4; }
                    else if (val === '6X9') { w = 152.4; h = 228.6; }
                    handleUpdateField('page_size', val);
                    handleUpdateField('page_width_mm', w);
                    handleUpdateField('page_height_mm', h);
                  }}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                >
                  <option value="A5">A5 (148 × 210 mm) - Chuẩn Ebook / Sách</option>
                  <option value="A4">A4 (210 × 297 mm) - Chuẩn Tài liệu</option>
                  <option value="6X9">6 × 9 inch (152 × 229 mm) - Sách Quốc tế</option>
                  <option value="LETTER">US Letter</option>
                </select>
              </div>

              {/* Font selection */}
              <div>
                <label className="block text-slate-400 mb-1">Font chữ nội dung (Body Font)</label>
                <select
                  value={currentProfile.body_font}
                  onChange={(e) => handleUpdateField('body_font', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                >
                  <option value="Noto Serif">Noto Serif (Có chân - Dễ đọc khi in & PDF)</option>
                  <option value="Noto Sans">Noto Sans (Không chân - Hiện đại, rõ nét)</option>
                  <option value="Inter">Inter</option>
                  <option value="Times New Roman">Times New Roman</option>
                </select>
              </div>

              {/* Font size & Line height */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-400 mb-1">Cỡ chữ (pt)</label>
                  <input
                    type="number"
                    step="0.5"
                    min="8"
                    max="24"
                    value={currentProfile.body_font_size_pt}
                    onChange={(e) => handleUpdateField('body_font_size_pt', parseFloat(e.target.value))}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Giãn dòng (Line height)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="1.0"
                    max="2.5"
                    value={currentProfile.line_height}
                    onChange={(e) => handleUpdateField('line_height', parseFloat(e.target.value))}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                  />
                </div>
              </div>

              {/* Text Alignment & First line indent */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-400 mb-1">Căn lề (Alignment)</label>
                  <select
                    value={currentProfile.text_alignment}
                    onChange={(e) => handleUpdateField('text_alignment', e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                  >
                    <option value="justify">Căn đều 2 bên (Justified)</option>
                    <option value="left">Căn trái (Left)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Thụt đầu dòng (mm)</label>
                  <input
                    type="number"
                    step="1"
                    min="0"
                    max="20"
                    value={currentProfile.first_line_indent_mm}
                    onChange={(e) => handleUpdateField('first_line_indent_mm', parseFloat(e.target.value))}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-sky-500"
                  />
                </div>
              </div>

              {/* Margins */}
              <div>
                <label className="block text-slate-400 mb-1">Lề trang (Margins mm)</label>
                <div className="grid grid-cols-2 gap-2">
                  <input
                    type="number"
                    placeholder="Trên"
                    value={currentProfile.margin_top_mm}
                    onChange={(e) => handleUpdateField('margin_top_mm', parseFloat(e.target.value))}
                    className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5 text-white"
                  />
                  <input
                    type="number"
                    placeholder="Dưới"
                    value={currentProfile.margin_bottom_mm}
                    onChange={(e) => handleUpdateField('margin_bottom_mm', parseFloat(e.target.value))}
                    className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5 text-white"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right: Smart Preview Frame */}
        <div className="flex-1 flex flex-col bg-slate-900/50 p-6 overflow-hidden">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-semibold text-slate-400 flex items-center space-x-1.5">
              <Eye className="w-3.5 h-3.5 text-sky-400" />
              <span>Smart Preview (Render mẫu đại diện)</span>
            </span>

            <div className="flex items-center space-x-2">
              <select
                value={sampleType}
                onChange={(e) => {
                  setSampleType(e.target.value);
                  loadPreview(currentProfile?.id, e.target.value);
                }}
                className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-white"
              >
                <option value="representative">Mẫu đại diện</option>
                <option value="first_pages">Chương mở đầu</option>
              </select>

              <button
                onClick={() => loadPreview(currentProfile?.id, sampleType)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300"
                title="Làm mới xem trước"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="flex-1 bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex items-center justify-center p-4">
            {loadingPreview ? (
              <div className="text-slate-500 text-xs flex items-center space-x-2">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Đang render trang mẫu...</span>
              </div>
            ) : (
              <iframe
                title="Reflow Preview"
                srcDoc={previewHtml}
                className="w-full h-full rounded-xl bg-white border-0"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

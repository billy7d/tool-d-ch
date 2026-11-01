import React, { useEffect, useState } from 'react';
import { Users, Wand2, Plus, Trash2, AlertTriangle } from 'lucide-react';
import { apiClient } from '../api/client';
import { AddressPair, AddressTerms, DocumentType, StyleRule } from '../types';
import type { CriticMode } from '../lib/styleGuide';

// Khớp DEFAULT_READER_TERMS ở backend/app/services/translation/address_terms.py.
const DEFAULT_READER: Partial<Record<DocumentType, string>> = {
  GENERAL: 'bạn',
  SELF_HELP: 'bạn',
  BUSINESS: 'bạn',
  TECHNICAL: 'bạn',
  FINANCE: 'bạn',
};

interface StyleConsistencyPanelProps {
  projectId: string;
  documentType: DocumentType;
  addressTerms: AddressTerms;
  onAddressTermsChange: (value: AddressTerms) => void;
  selectedModel: string;
  criticModel: string;
  onCriticModelChange: (value: string) => void;
  criticMode: CriticMode;
  onCriticModeChange: (value: CriticMode) => void;
  installedModels: string[];
  glossaryPresets: string[];
  onGlossaryPresetsChange: (value: string[]) => void;
  semanticCriticModel: string;
  onSemanticCriticModelChange: (value: string) => void;
}

export type { CriticMode } from '../lib/styleGuide';

// Khớp PRESETS trong backend/app/services/translation/glossary_presets.py.
export const GLOSSARY_PRESET_DOMAINS: { value: string; label: string }[] = [
  { value: 'LEGAL', label: 'Pháp lý' },
  { value: 'TECHNICAL', label: 'Kỹ thuật' },
  { value: 'FINANCE', label: 'Tài chính' },
  { value: 'BUSINESS', label: 'Kinh doanh' },
  { value: 'ACADEMIC', label: 'Học thuật' },
];

const inputClass =
  'w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-2 text-xs text-white focus:outline-none focus:border-sky-500';

export const StyleConsistencyPanel: React.FC<StyleConsistencyPanelProps> = ({
  projectId,
  documentType,
  addressTerms,
  onAddressTermsChange,
  selectedModel,
  criticModel,
  onCriticModelChange,
  criticMode,
  onCriticModeChange,
  installedModels,
  glossaryPresets,
  onGlossaryPresetsChange,
  semanticCriticModel,
  onSemanticCriticModelChange,
}) => {
  const [rules, setRules] = useState<StyleRule[]>([]);

  useEffect(() => {
    apiClient.getStyleRules(projectId).then(setRules).catch(() => setRules([]));
  }, [projectId]);

  const pairs = addressTerms.pairs || [];
  const characters = Object.entries(addressTerms.characters || {});
  const readerPlaceholder = DEFAULT_READER[documentType]
    ? `Mặc định: ${DEFAULT_READER[documentType]}`
    : 'Để trống: model tự chọn';

  const update = (patch: Partial<AddressTerms>) => onAddressTermsChange({ ...addressTerms, ...patch });

  const updatePair = (index: number, patch: Partial<AddressPair>) =>
    update({ pairs: pairs.map((pair, i) => (i === index ? { ...pair, ...patch } : pair)) });

  const updateCharacter = (index: number, name: string, term: string) => {
    const next = characters.map((entry, i) => (i === index ? [name, term] : entry));
    update({ characters: Object.fromEntries(next) });
  };

  const toggleRule = async (rule: StyleRule) => {
    // Quy tắc đang chờ có thể được người dùng bật thủ công mà không cần đợi lặp lại.
    const status = rule.status === 'ACTIVE' ? 'DISABLED' : 'ACTIVE';
    try {
      const updated = await apiClient.updateStyleRule(projectId, rule.id, status);
      setRules((items) => items.map((item) => (item.id === updated.id ? updated : item)));
    } catch (e) {
      alert('Không thể cập nhật quy tắc: ' + e);
    }
  };

  const selfReview = !criticModel || criticModel === selectedModel;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Users className="w-4 h-4 text-amber-400" />
            Sổ xưng hô
          </h3>
          <p className="text-[11px] text-slate-400 mt-1">
            Khóa cách xưng hô để mọi chương dùng nhất quán. Chỉ mục có tên xuất hiện trong đoạn đang dịch mới được đưa vào prompt.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-[11px] text-slate-300 mb-1">Gọi người đọc ("you")</label>
            <input
              value={addressTerms.reader ?? ''}
              placeholder={readerPlaceholder}
              onChange={(e) => update({ reader: e.target.value || undefined })}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-[11px] text-slate-300 mb-1">Người kể tự xưng ("I")</label>
            <input
              value={addressTerms.narrator ?? ''}
              placeholder="vd: tôi"
              onChange={(e) => update({ narrator: e.target.value || undefined })}
              className={inputClass}
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-slate-300">Cặp xưng hô giữa nhân vật</span>
            <button
              onClick={() => update({ pairs: [...pairs, { speaker: '', listener: '', self: '', other: '' }] })}
              className="text-[11px] text-sky-400 flex items-center gap-1"
            >
              <Plus className="w-3 h-3" /> Thêm cặp
            </button>
          </div>
          {pairs.map((pair, index) => (
            <div key={index} className="grid grid-cols-[1fr_1fr_0.7fr_0.7fr_auto] gap-1.5 items-center">
              <input value={pair.speaker} placeholder="Người nói" onChange={(e) => updatePair(index, { speaker: e.target.value })} className={inputClass} />
              <input value={pair.listener} placeholder="Người nghe" onChange={(e) => updatePair(index, { listener: e.target.value })} className={inputClass} />
              <input value={pair.self} placeholder="Tự xưng" onChange={(e) => updatePair(index, { self: e.target.value })} className={inputClass} />
              <input value={pair.other} placeholder="Gọi là" onChange={(e) => updatePair(index, { other: e.target.value })} className={inputClass} />
              <button onClick={() => update({ pairs: pairs.filter((_, i) => i !== index) })} className="text-slate-500 hover:text-rose-400" aria-label="Xóa cặp">
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-slate-300">Cách gọi nhân vật ở ngôi thứ ba</span>
            <button
              onClick={() => update({ characters: { ...(addressTerms.characters || {}), '': '' } })}
              className="text-[11px] text-sky-400 flex items-center gap-1"
            >
              <Plus className="w-3 h-3" /> Thêm nhân vật
            </button>
          </div>
          {characters.map(([name, term], index) => (
            <div key={index} className="grid grid-cols-[1fr_1fr_auto] gap-1.5 items-center">
              <input value={name} placeholder="Tên (vd: Alice)" onChange={(e) => updateCharacter(index, e.target.value, term)} className={inputClass} />
              <input value={term} placeholder="Gọi là (vd: cô)" onChange={(e) => updateCharacter(index, name, e.target.value)} className={inputClass} />
              <button
                onClick={() => update({ characters: Object.fromEntries(characters.filter((_, i) => i !== index)) })}
                className="text-slate-500 hover:text-rose-400"
                aria-label="Xóa nhân vật"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Wand2 className="w-4 h-4 text-violet-400" />
            Kiểm định văn phong
          </h3>
          <p className="text-[11px] text-slate-400 mt-1">
            Critic chấm độ tự nhiên nên dùng model khác model dịch để tránh tự chấm bài của chính mình.
          </p>
        </div>

        <div>
          <label className="block text-[11px] text-slate-300 mb-1">Thuật ngữ mẫu theo lĩnh vực (gợi ý mềm)</label>
          <div className="flex flex-wrap gap-1.5">
            {GLOSSARY_PRESET_DOMAINS.map(({ value, label }) => {
              const active = glossaryPresets.includes(value);
              return (
                <button
                  key={value}
                  onClick={() => onGlossaryPresetsChange(active ? glossaryPresets.filter((d) => d !== value) : [...glossaryPresets, value])}
                  className={`px-2 py-1 rounded-md text-[11px] border ${active ? 'bg-sky-500/20 border-sky-500/40 text-sky-200' : 'bg-slate-950 border-slate-800 text-slate-400'}`}
                >
                  {label}
                </button>
              );
            })}
          </div>
          <p className="text-[10px] text-slate-500 mt-1">Ví dụ "licensee" → "bên được cấp phép". Thuật ngữ bạn tự thêm luôn được ưu tiên.</p>
        </div>

        <div>
          <label className="block text-[11px] text-slate-300 mb-1">Cách kiểm độ tự nhiên</label>
          <select value={criticMode} onChange={(e) => onCriticModeChange(e.target.value as CriticMode)} className={inputClass}>
            <option value="AUTO">Tự động (lint; critic AI chỉ ở mức Xuất bản)</option>
            <option value="LINT_ONLY">Chỉ lint (nhanh nhất)</option>
            <option value="ALWAYS">Luôn dùng critic AI (chậm hơn ~30s/đoạn với model 7B)</option>
          </select>
        </div>

        <div>
          <label className="block text-[11px] text-slate-300 mb-1">Model critic độ tự nhiên</label>
          <select value={criticModel} onChange={(e) => onCriticModelChange(e.target.value)} className={inputClass + ' font-mono'}>
            <option value="">Giống model dịch ({selectedModel})</option>
            {installedModels.filter((m) => m !== selectedModel).map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          {selfReview && (
            <span className="text-[11px] text-amber-400/90 mt-1 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> Critic đang dùng chính model dịch; điểm có thể bị thiên vị.
            </span>
          )}
        </div>

        <div>
          <label className="block text-[11px] text-slate-300 mb-1">Model critic ngữ nghĩa</label>
          <select value={semanticCriticModel} onChange={(e) => onSemanticCriticModelChange(e.target.value)} className={inputClass + ' font-mono'}>
            <option value="">Giống model dịch ({selectedModel})</option>
            {installedModels.filter((m) => m !== selectedModel).map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <p className="text-[10px] text-slate-500 mt-1">Chỉ chạy với câu rủi ro (điều kiện, phủ định, lượng từ…). Model lớn hơn bắt lỗi sai nghĩa tốt hơn nhưng chậm hơn.</p>
        </div>

        <div className="space-y-2">
          <span className="text-[11px] text-slate-300">Quy tắc học từ chỉnh sửa của bạn ({rules.length})</span>
          {rules.length === 0 ? (
            <p className="text-[11px] text-slate-500">
              Chưa có. Khi bạn sửa và duyệt bản dịch ở Bước 6, những cụm bị sửa lặp lại ở từ 2 đoạn trở lên sẽ thành quy tắc.
            </p>
          ) : (
            <div className="border border-slate-800 rounded-xl max-h-[220px] overflow-y-auto divide-y divide-slate-800">
              {rules.map((rule) => (
                <div key={rule.id} className="flex items-center justify-between gap-2 px-3 py-2 text-xs">
                  <span className="text-slate-300">
                    <span className="line-through text-rose-300/80">{rule.avoid}</span>
                    {' → '}
                    <span className="text-emerald-300">{rule.prefer || '(lược bỏ)'}</span>
                    <span className="text-slate-500"> · {rule.occurrences} lần</span>
                  </span>
                  <button
                    onClick={() => toggleRule(rule)}
                    className={`shrink-0 px-2 py-0.5 rounded-md text-[10px] font-semibold border ${
                      rule.status === 'ACTIVE'
                        ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
                        : rule.status === 'DISABLED'
                        ? 'bg-slate-800 border-slate-700 text-slate-400'
                        : 'bg-sky-500/10 border-sky-500/30 text-sky-300'
                    }`}
                  >
                    {rule.status === 'ACTIVE' ? 'Đang áp dụng' : rule.status === 'DISABLED' ? 'Đã tắt' : 'Chờ lặp lại · bật'}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

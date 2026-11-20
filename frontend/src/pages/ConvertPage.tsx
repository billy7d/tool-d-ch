import React, { useEffect, useRef, useState } from 'react';
import {
  UploadCloud,
  FileText,
  Book,
  Tablet,
  Download,
  Loader2,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Trash2,
  Info,
  RefreshCw,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { ConversionJob, ConvertOptionsInfo } from '../types';

type OutputFormat = 'epub' | 'mobi';

const formatBytes = (bytes: number) => {
  if (!bytes) return '0 KB';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const STATUS_LABEL: Record<ConversionJob['status'], string> = {
  queued: 'Đang chờ',
  running: 'Đang chuyển',
  done: 'Hoàn tất',
  error: 'Lỗi',
};

export const ConvertPage: React.FC = () => {
  const [options, setOptions] = useState<ConvertOptionsInfo | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [format, setFormat] = useState<OutputFormat>('epub');
  const [device, setDevice] = useState<string>('kindle_pw');
  const [title, setTitle] = useState('');
  const [author, setAuthor] = useState('');
  const [language, setLanguage] = useState('auto');
  const [includeImages, setIncludeImages] = useState(true);
  const [makeCover, setMakeCover] = useState(true);
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [job, setJob] = useState<ConversionJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<ConversionJob[]>([]);
  const pollRef = useRef<number | null>(null);

  const loadHistory = async () => {
    try {
      setHistory(await apiClient.listConversions());
    } catch {
      // Không có lịch sử cũng không sao.
    }
  };

  useEffect(() => {
    apiClient
      .getConvertOptions()
      .then((data) => {
        setOptions(data);
        setDevice(data.default_device);
      })
      .catch(() => setError('Không tải được cấu hình chuyển đổi từ máy chủ.'));
    loadHistory();
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  const stopPolling = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const startPolling = (jobId: string) => {
    stopPolling();
    pollRef.current = window.setInterval(async () => {
      try {
        const latest = await apiClient.getConversionJob(jobId);
        setJob(latest);
        if (latest.status === 'done' || latest.status === 'error') {
          stopPolling();
          loadHistory();
        }
      } catch {
        stopPolling();
      }
    }, 1000);
  };

  const pickFile = (picked?: File | null) => {
    if (!picked) return;
    if (!picked.name.toLowerCase().endsWith('.pdf')) {
      setError('Chỉ nhận file PDF (.pdf).');
      return;
    }
    setError(null);
    setJob(null);
    setFile(picked);
  };

  const busy = uploadPercent !== null || job?.status === 'queued' || job?.status === 'running';

  const handleConvert = async () => {
    if (!file || busy) return;
    setError(null);
    setJob(null);
    setUploadPercent(0);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('output_format', format);
      form.append('device', device);
      if (title.trim()) form.append('title', title.trim());
      if (author.trim()) form.append('author', author.trim());
      form.append('language', language);
      form.append('include_images', String(includeImages));
      form.append('cover', String(makeCover));
      const created = await apiClient.startConversion(form, setUploadPercent);
      setJob(created);
      startPolling(created.id);
    } catch (err: any) {
      setError(err?.response?.data?.detail || `Không gửi được file: ${err}`);
    } finally {
      setUploadPercent(null);
    }
  };

  const handleDelete = async (jobId: string) => {
    try {
      await apiClient.deleteConversion(jobId);
      if (job?.id === jobId) setJob(null);
      loadHistory();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Không xóa được.');
    }
  };

  const mobiEngineNote = options?.calibre_available
    ? 'Máy đã có Calibre: MOBI được tạo bằng Calibre (chế độ MOBI 6 cho Kindle cũ).'
    : 'Không cần Calibre: dùng bộ ghi MOBI 6 tích hợp sẵn.';

  const result = job?.result;

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-bold text-white tracking-tight">Chuyển PDF sang EPUB / MOBI cho Kindle</h2>
        <p className="text-slate-400 text-xs leading-relaxed max-w-3xl">
          Dựng lại sách dạng chữ co giãn (reflow) từ PDF: tách đúng đoạn văn, nối chữ bị ngắt gạch nối cuối dòng, bỏ đầu
          trang/chân trang và số trang, nối đoạn bị cắt khi sang trang, giữ chữ nghiêng/đậm, trích dẫn, thơ căn giữa,
          ảnh minh họa, bìa và mục lục nhiều cấp. Tối ưu cho <strong>KOReader</strong> trên các dòng Kindle đời cũ.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Cột cấu hình */}
        <div className="lg:col-span-3 space-y-5">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              pickFile(e.dataTransfer.files?.[0]);
            }}
            className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
              dragOver ? 'border-sky-500 bg-sky-500/10' : 'border-slate-800 bg-slate-900/50 hover:border-slate-700'
            }`}
          >
            {file ? (
              <div className="flex items-center justify-center space-x-3">
                <FileText className="w-8 h-8 text-sky-400 shrink-0" />
                <div className="text-left min-w-0">
                  <span className="block text-sm font-semibold text-white truncate">{file.name}</span>
                  <span className="block text-[11px] text-slate-400">{formatBytes(file.size)}</span>
                </div>
              </div>
            ) : (
              <>
                <UploadCloud className="w-12 h-12 text-sky-400 mx-auto mb-3" />
                <h3 className="text-sm font-semibold text-white">Kéo thả file PDF vào đây</h3>
                <p className="text-slate-400 text-xs mt-1">PDF có lớp chữ (không phải ảnh scan) cho kết quả tốt nhất</p>
              </>
            )}
            <label className="cursor-pointer inline-flex items-center space-x-2 mt-4 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all">
              <span>{file ? 'Chọn file khác' : 'Chọn file PDF'}</span>
              <input
                type="file"
                accept=".pdf,application/pdf"
                disabled={busy}
                onChange={(e) => pickFile(e.target.files?.[0])}
                className="hidden"
              />
            </label>
          </div>

          {/* Định dạng */}
          <div className="space-y-2">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Định dạng xuất</span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {(['epub', 'mobi'] as OutputFormat[]).map((key) => {
                const info = options?.formats.find((f) => f.key === key);
                const active = format === key;
                const Icon = key === 'epub' ? Book : Tablet;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setFormat(key)}
                    className={`text-left p-4 rounded-xl border transition-all ${
                      active
                        ? key === 'epub'
                          ? 'border-emerald-500/60 bg-emerald-500/10'
                          : 'border-amber-500/60 bg-amber-500/10'
                        : 'border-slate-800 bg-slate-900 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center space-x-2">
                      <Icon className={`w-4 h-4 ${key === 'epub' ? 'text-emerald-400' : 'text-amber-400'}`} />
                      <span className="text-sm font-semibold text-white">{info?.label || key.toUpperCase()}</span>
                      {info?.recommended && (
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-semibold">
                          Khuyên dùng
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">{info?.description}</p>
                    {key === 'mobi' && <p className="text-[11px] text-amber-300/80 mt-1">{mobiEngineNote}</p>}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Dòng máy */}
          <div className="space-y-2">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Dòng máy Kindle</span>
            <div className="space-y-2">
              {(options?.devices || []).map((d) => (
                <label
                  key={d.key}
                  className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition-all ${
                    device === d.key ? 'border-sky-500/60 bg-sky-500/10' : 'border-slate-800 bg-slate-900 hover:border-slate-700'
                  }`}
                >
                  <input
                    type="radio"
                    name="device"
                    value={d.key}
                    checked={device === d.key}
                    onChange={() => setDevice(d.key)}
                    className="mt-0.5 accent-sky-500"
                  />
                  <div>
                    <span className="block text-xs font-semibold text-white">{d.label}</span>
                    <span className="block text-[11px] text-slate-400">{d.description}</span>
                  </div>
                </label>
              ))}
            </div>
            <p className="text-[11px] text-slate-500">Quyết định độ phân giải ảnh bìa và ảnh minh họa (nhẹ, giải mã nhanh trên máy RAM thấp).</p>
          </div>

          {/* Thông tin sách */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <label className="space-y-1 sm:col-span-2">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Tên sách</span>
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Tự lấy từ PDF"
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-white placeholder-slate-600 focus:border-sky-500 outline-none"
              />
            </label>
            <label className="space-y-1">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Ngôn ngữ</span>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-white focus:border-sky-500 outline-none"
              >
                <option value="auto">Tự nhận diện</option>
                <option value="vi">Tiếng Việt</option>
                <option value="en">English</option>
              </select>
            </label>
            <label className="space-y-1 sm:col-span-3">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Tác giả</span>
              <input
                value={author}
                onChange={(e) => setAuthor(e.target.value)}
                placeholder="Tự lấy từ PDF"
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-white placeholder-slate-600 focus:border-sky-500 outline-none"
              />
            </label>
          </div>

          <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-slate-300">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input type="checkbox" checked={includeImages} onChange={(e) => setIncludeImages(e.target.checked)} className="accent-sky-500" />
              <span>Giữ ảnh minh họa</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input type="checkbox" checked={makeCover} onChange={(e) => setMakeCover(e.target.checked)} className="accent-sky-500" />
              <span>Tạo ảnh bìa từ trang 1 (hiện trong thư viện KOReader)</span>
            </label>
          </div>

          <button
            onClick={handleConvert}
            disabled={!file || busy}
            className="w-full flex items-center justify-center space-x-2 py-3 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-sm font-semibold shadow-lg shadow-sky-500/20 transition-all disabled:opacity-50"
          >
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            <span>
              {uploadPercent !== null
                ? `Đang tải lên ${uploadPercent}%...`
                : busy
                  ? 'Đang chuyển đổi...'
                  : `Chuyển sang ${format.toUpperCase()}`}
            </span>
          </button>

          {error && (
            <div className="flex items-start space-x-2.5 p-4 rounded-xl bg-red-950/40 border border-red-800 text-red-300 text-xs">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Cột tiến độ & kết quả */}
        <div className="lg:col-span-2 space-y-5">
          {job && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-white truncate pr-2">{job.filename}</span>
                <span
                  className={`shrink-0 px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                    job.status === 'done'
                      ? 'bg-emerald-500/20 text-emerald-300'
                      : job.status === 'error'
                        ? 'bg-red-500/20 text-red-300'
                        : 'bg-sky-500/20 text-sky-300'
                  }`}
                >
                  {STATUS_LABEL[job.status]}
                </span>
              </div>

              {(job.status === 'queued' || job.status === 'running') && (
                <div className="space-y-1.5">
                  <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div className="h-full bg-sky-500 transition-all" style={{ width: `${Math.max(3, job.progress)}%` }} />
                  </div>
                  <span className="text-[11px] text-slate-400">
                    {job.stage} · {job.progress.toFixed(0)}%
                  </span>
                </div>
              )}

              {job.status === 'error' && (
                <div className="flex items-start space-x-2 text-xs text-red-300">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{job.error}</span>
                </div>
              )}

              {job.status === 'done' && result && (
                <div className="space-y-4">
                  <div className="flex items-center space-x-2 text-emerald-400 text-sm font-semibold">
                    <CheckCircle2 className="w-5 h-5" />
                    <span>Đã tạo {result.output_format.toUpperCase()}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    {[
                      ['Tên sách', result.title],
                      ['Tác giả', result.author || '—'],
                      ['Số trang PDF', result.page_count],
                      ['Mục lục', `${result.toc_entries} mục`],
                      ['Đoạn văn', result.paragraph_count],
                      ['Ảnh', result.image_count],
                      ['Kiểu đoạn', result.paragraph_style === 'indent' ? 'Thụt đầu dòng' : 'Cách đoạn'],
                      ['Dung lượng', formatBytes(result.file_size_bytes)],
                    ].map(([label, value]) => (
                      <div key={String(label)} className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 min-w-0">
                        <span className="text-slate-500 block">{label}</span>
                        <span className="text-white font-medium block truncate" title={String(value)}>
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>
                  {result.output_format === 'mobi' && (
                    <p className="text-[11px] text-slate-400">
                      Tạo bằng {result.engine === 'calibre' ? 'Calibre' : 'bộ ghi MOBI tích hợp'}.
                    </p>
                  )}
                  {result.warnings.length > 0 && (
                    <div className="space-y-1.5">
                      {result.warnings.map((w, i) => (
                        <div key={i} className="flex items-start space-x-2 text-[11px] text-amber-300/90">
                          <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                          <span>{w}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {job.download_url && (
                    <a
                      href={job.download_url}
                      download={job.output_name || undefined}
                      className="w-full flex items-center justify-center space-x-2 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white text-xs font-semibold shadow-lg shadow-emerald-500/20 transition-all"
                    >
                      <Download className="w-4 h-4" />
                      <span className="truncate">Tải xuống {job.output_name}</span>
                    </a>
                  )}
                </div>
              )}
            </div>
          )}

          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-2.5 text-[11px] text-slate-400 leading-relaxed">
            <div className="flex items-center space-x-2 text-slate-200 text-xs font-semibold">
              <Info className="w-4 h-4 text-sky-400" />
              <span>Đọc trên Kindle bằng KOReader</span>
            </div>
            <p>Cắm Kindle vào máy tính qua USB, chép file vào thư mục <code className="text-sky-300">documents</code>, rồi mở bằng trình duyệt file của KOReader.</p>
            <p>EPUB cho kết quả tốt nhất trong KOReader: mục lục nhiều cấp, ảnh, chữ nghiêng và cỡ chữ/font chỉnh tự do (menu chữ <strong>Aa</strong>).</p>
            <p>Sách tiếng Việt: nếu trình đọc gốc của Kindle hiện sai dấu với file MOBI, hãy mở bằng KOReader (có sẵn font đủ dấu).</p>
            {options && !options.tesseract_available && (
              <p className="text-amber-300/80">Máy chưa cài Tesseract OCR: trang scan (không có lớp chữ) sẽ được giữ dạng ảnh cả trang.</p>
            )}
          </div>

          {history.length > 0 && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Lượt chuyển gần đây</span>
              <div className="space-y-2">
                {history.slice(0, 8).map((h) => (
                  <div key={h.id} className="flex items-center justify-between gap-2 text-[11px]">
                    <div className="min-w-0">
                      <span className="block text-slate-200 truncate">{h.output_name || h.filename}</span>
                      <span className="block text-slate-500">
                        {h.output_format.toUpperCase()} · {STATUS_LABEL[h.status]}
                        {h.result ? ` · ${formatBytes(h.result.file_size_bytes)}` : ''}
                      </span>
                    </div>
                    <div className="flex items-center space-x-1 shrink-0">
                      {h.download_url && (
                        <a href={h.download_url} download className="p-1.5 rounded-lg hover:bg-slate-800 text-emerald-400" title="Tải xuống">
                          <Download className="w-3.5 h-3.5" />
                        </a>
                      )}
                      {(h.status === 'done' || h.status === 'error') && (
                        <button onClick={() => handleDelete(h.id)} className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-500 hover:text-red-400" title="Xóa">
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-slate-600">File kết quả được giữ 24 giờ; file PDF gốc bị xóa ngay sau khi chuyển.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

# Local AI Document Translator & Reflow Publisher (Version 1.0)

> Hệ thống Dịch thuật & Xuất bản Tài liệu AI Cục bộ dành cho Windows 10/11 theo kiến trúc **Reflow-first**.

---

## 🌟 Tổng quan sản phẩm

**Local AI Document Translator & Reflow Publisher** là ứng dụng web chạy hoàn toàn trên máy tính cá nhân (100% offline, local-first, không phụ thuộc API trả phí). Ứng dụng cho phép bạn đưa vào các tài liệu, sách tiếng Anh dài từ hàng trăm đến hơn **1.000 trang** (~300.000–500.000+ từ), tự động bóc tách cấu trúc, OCR selective, kiểm định và khóa cấu trúc, thiết lập văn phong dịch (Natural Vietnamese), dịch có bộ nhớ ngữ cảnh và xuất bản thành **PDF, EPUB, MOBI** chất lượng cao.

---

## 🔄 8 Bước Quy trình (Workflow)

```
1. Import ➔ 2. Phân tích / OCR ➔ 3. Kiểm tra cấu trúc ➔ 4. Thiết lập dịch ➔ 5. Dịch ➔ 6. QA & Chỉnh sửa ➔ 7. Layout & Preview ➔ 8. Export
```

1. **Import**: Kéo thả file PDF, EPUB, DOCX, TXT, Markdown. Nguồn gốc được bảo toàn bất biến (immutable).
2. **Phân tích / OCR**: Nhận diện trang scan vs trang text, thực hiện Selective OCR bằng Tesseract, nối dòng ngắt gạch nối (Hyphen Repair), bóc tách ảnh và lọc header/footer lặp lại.
3. **Kiểm tra cấu trúc (Structure Verification)**: Trình xem 2 cột (Cây cấu trúc & Bản xem trước gốc). Cho phép gộp/tách đoạn, đổi cấp tiêu đề H1-H4, xử lý bảng biểu và khóa cấu trúc trước khi dịch.
4. **Thiết lập dịch**: Chọn chế độ dịch (Natural Vietnamese, Balanced, Faithful, Academic, Technical, Custom), chọn mô hình Ollama cục bộ, tự động trích xuất bảng thuật ngữ (Glossary) và khóa các thuật ngữ quan trọng.
5. **Dịch (Translation Queue)**: Hàng đợi dịch bất đồng bộ với cơ chế **Immediate SQLite Commit** (lưu tức thì từng block) và **Crash Recovery** (chống mất tiến trình khi mất điện hoặc tắt máy đột ngột).
6. **QA & Chỉnh sửa (Translation Editor)**: Trình biên tập song song 3 cột (Outline | Source | Vietnamese) với tự động lưu (debounced autosave), dịch lại từng đoạn theo yêu cầu riêng, quét QA phát hiện thiếu số liệu/sai lệch độ dài và quét tính nhất quán thuật ngữ toàn văn bản.
7. **Layout & Preview (Reflow Renderer)**: Tách biệt nội dung và hình thức. Tùy biến khổ trang (A4, A5, 6x9, Custom), font chữ Việt (Noto Serif, Noto Sans), giãn dòng, lề trang và Smart Preview tức thì.
8. **Export**: Xuất bản ra PDF Reflow (searchable, nhúng font, bookmark), EPUB3 (reflowable, tối ưu cho KOReader) và MOBI 6 cho Kindle đời cũ (bộ ghi tích hợp sẵn; dùng Calibre nếu máy có cài).

---

## 📚 Chuyển PDF → EPUB / MOBI cho Kindle (KOReader)

Mục **Chuyển PDF → Ebook** ở thanh bên chuyển thẳng một file PDF (không dịch) thành ebook chữ co giãn:

- Tách đúng đoạn văn theo thụt đầu dòng, nối chữ bị ngắt gạch nối cuối dòng (dùng từ điển lấy từ chính cuốn sách), nối đoạn bị cắt khi sang trang/sang cột.
- Bỏ đầu trang, chân trang và số trang; giữ chữ nghiêng/đậm, trích dẫn, thơ và lời đề tặng căn giữa, dấu ngắt cảnh, chú thích cuối trang (dời về cuối chương).
- Ảnh minh họa nhúng sẵn (JPEG baseline/PNG không trong suốt, vừa độ phân giải máy), ảnh bìa từ trang 1, mục lục nhiều cấp từ outline của PDF hoặc từ tiêu đề.
- EPUB: CSS tối giản theo đơn vị em, không ép font nên KOReader tùy chỉnh font/cỡ chữ tự do và chữ Việt luôn đủ dấu. **Khuyên dùng cho KOReader.**
- MOBI 6 cổ điển cho trình đọc gốc của Kindle đời cũ; không cần Calibre.
- Chọn dòng máy: Kindle 6" đời cũ (600×800), Paperwhite 1/2 (758×1024), Kindle 300 ppi (1072×1448).

Trang scan không có lớp chữ được nhận dạng bằng **Tesseract OCR** (miễn phí) rồi dàn lại như sách thường, giữ cả hình minh họa trên trang scan; không có Tesseract thì trang scan được giữ dạng ảnh cả trang.

- Tesseract đặt sẵn trong `tools/tesseract/` (bản Windows 5.4 của UB Mannheim, chỉ bung `tesseract.exe` + DLL, không chạy bộ cài nên không ghi gì vào ổ C:/registry), dữ liệu ngôn ngữ ở `tools/tesseract/tessdata/`: `eng`, `vie` bản *fast* (đo trên trang scan: chính xác ngang bản *best* — ~99,9% từ với tiếng Anh, ~96–97% với tiếng Việt — mà nhanh gần gấp đôi).
- Máy khác: bung bộ cài `tesseract-ocr-w64-setup-*.exe` bằng 7-Zip vào `tools/tesseract`, chép `vie.traineddata` từ https://github.com/tesseract-ocr/tessdata_fast vào `tessdata/`; hoặc đặt `TOOL_DICH_TESSERACT` trỏ tới `tesseract.exe`; hoặc dùng Tesseract có sẵn trên PATH.
- OCR chạy song song mỗi trang một tiến trình 1 luồng (nhanh ~5 lần so với để Tesseract tự chia luồng), tự chọn tiếng Việt/tiếng Anh theo nội dung sách.
- Sau OCR tiếng Việt, app tự sửa lỗi dấu theo ngữ cảnh (Tesseract đọc đúng mặt chữ nhưng hay nhầm dấu: "trước dây" → "trước đây", "cái bãy" → "cái bẫy", "niểm tin" → "niềm tin", "lôi" đầu câu → "Tôi"). Bộ sửa chọn biến thể cùng mặt chữ theo tần suất cặp từ trong từ điển Anh–Việt `data/dictionaries/en-vi` và trong chính cuốn sách, dùng độ tin cậy Tesseract của từng từ và cả các phương án thay thế từng chữ cái mà Tesseract đã cân nhắc ("tiển": Tesseract chấm "ể" 94 điểm, "ề" 64 điểm → "tiền"); chỉ sửa khi bằng chứng rõ, không đụng tên riêng, từ viết tắt, đường dẫn. Đầu trang/chân trang được bỏ trước khi sửa để lỗi OCR lặp lại ở đầu trang không làm sai thống kê của sách. Không có từ điển thì bỏ qua bước này.
- Dòng bị đọc vỡ (chữ in nghiêng như "USA Today" → "US⁄1 Toảzy", tên sách/tạp chí tiếng Anh như "the" → "tbe") được cắt ra đọc lại bằng mô hình `eng` và bằng `vie` trên ảnh đã nắn thẳng chữ nghiêng; từng chữ chỉ được thay khi cách đọc mới là chữ có thật (từ điển Việt hoặc mục từ tiếng Anh của từ điển Anh–Việt) và giống chữ gốc hoặc được hai cách đọc đồng thuận. Tên riêng đọc sai lẻ tẻ được thống nhất theo cách viết phổ biến trong sách ("Scahawks" → "Seahawks"), chữ hoa lớn đầu chương bị bỏ sót được đọc riêng ("ăm 26 tuổi" → "Năm 26 tuổi").
- Bố cục sách scan: trang "Mục lục" in sẵn được bỏ số trang (kể cả số bị đọc sai như "l4", "ó8"), tách/nối mục dính hoặc xuống dòng, và dùng để sửa tiêu đề chương in phông trang trí ("Cuộc đời là một ván poker, không phủ" + "¡ mộồt ván cờ vud" → tên chương trong mục lục); chú thích vắt sang trang sau không còn chen vào giữa đoạn văn; văn xuôi canh trái ở bìa sau không bị dàn như thơ; sơ đồ/biểu đồ có chữ được giữ thành hình.
- Kiểm tra bắt buộc từng chữ sau OCR: mỗi chữ phải là âm tiết/từ có trong từ điển, số, ngày, mã, đường dẫn, tên riêng hợp lệ, hoặc chữ chính cuốn sách dùng nhiều lần với độ tin cậy cao. Chữ không đạt (ký tự lạ "øƒ", chữ lẫn số "7e", chữ cái lẻ, chữ dính "máigiữa", hoa thường lẫn lộn "JournAl") được sửa ngay nếu chắc chắn; còn lại thì cắt riêng dòng đó và quét lại theo từng lượt (tối đa 4 lượt × 3 cách đọc: phóng to, nắn chữ nghiêng nhiều góc, nhị phân hóa, làm đậm nét; mô hình `vie`, `eng`, `vie+eng`) cho tới khi hết lỗi. Mỗi vùng chỉ được thay khi các cách đọc đồng thuận rõ ràng: dấu tiếng Việt chỉ lấy từ mô hình `vie`, tên lạ phải có ở chỗ khác trong sách, câu tiếng Việt không nhận từ chỉ có trong tiếng Anh ("giá" không thành "gid"); chữ hiếm mà các lần quét lại đều đọc giống nhau thì giữ ("premortem", "xèng"). Hết lượt mà vẫn hỏng: lấy cách đọc được cả mô hình Việt lẫn Anh cùng đọc ra, không có thì bỏ chuỗi ký tự rác; các chỗ đó được ghi theo số trang trong cảnh báo để soát lại.
- Đo trên sách scan thật 322 trang (12 trang có đáp án đọc từ ảnh, ~4.400 chữ): còn 9 lỗi trên 56 lỗi OCR ban đầu, không sửa hỏng chữ đúng nào; cả cuốn ~840 lần sửa dấu và ~1.170 chữ in nghiêng/tiếng Anh được đọc lại. Bước kiểm tra từng chữ sau đó bắt ~600 chữ không đạt, sửa ~360 (soát ngẫu nhiên 48 chữ: đúng mặt chữ cả 48), bỏ ~14 cụm ký tự rác; số chữ còn ký tự lạ/lẫn số trong sách giảm từ 337 xuống 15 (khoảng một nửa là đường dẫn, khoảng số hợp lệ). Cả quá trình khoảng 4 phút (bước kiểm tra và quét lại chiếm ~1 phút).

---

## 🚀 Khởi chạy ứng dụng

Chỉ cần chạy file `start.bat` trên Windows:

```cmd
start.bat
```

Script sẽ tự động:
1. Tạo môi trường ảo Python `venv` nếu chưa có.
2. Cài đặt các thư viện cần thiết.
3. Build giao diện React / Vite / TailwindCSS.
4. Khởi chạy FastAPI backend và mở trình duyệt tại `http://127.0.0.1:8765`.

---

## 💻 Tech Stack

- **Backend**: Python 3.11+, FastAPI, SQLite (WAL mode), SQLAlchemy, PyMuPDF (`fitz`), pytesseract, ReportLab / WeasyPrint, python-docx, EbookLib.
- **Frontend**: React, TypeScript, Vite, Tailwind CSS, Lucide Icons.
- **Local AI Provider**: Ollama REST API (mặc định với `qwen2.5:7b` / `qwen2.5:14b`), OpenAI-compatible local fallback (LM Studio / vLLM) và MockProvider dành cho kiểm thử.
#   t o o l - d - c h  
 
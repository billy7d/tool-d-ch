import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from app.services.translation.citation_policy import citation_passthrough, is_bibliographic_citation
from app.services.translation.glossary_validator import GlossaryValidator
from app.services.translation.language_validator import validate_target_language
from app.services.translation.numeric_validator import NumericValidator
from app.services.translation.polarity_validator import PolarityValidator
from app.services.translation.reference_validator import ReferenceValidator


URL_PATTERN = re.compile(r"https?://[^\s)\]}]+|www\.[^\s)\]}]+", re.IGNORECASE)
QUOTE_PATTERN = re.compile(r"[\"“”]")
OPENING_QUOTE_LOST = re.compile(r"^[^\"“”]+[,.!?…]\s*[\"”]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
# Câu ngắn trùng nhau là bình thường ("Anh gật đầu."); chỉ coi là chép khi câu đủ dài và gần như trùng hẳn.
CONTEXT_LEAK_MIN_WORDS = 6
CONTEXT_LEAK_MIN_WORDS_SAME_LENGTH = 12
CONTEXT_LEAK_OVERLAP = 0.8
# Câu chỉ dẫn trong prompt bị model chép vào bản dịch (chạy sách thật: một chú thích ra
# "20.\n\nNhắc lại: chỉ dùng chữ Quốc ngữ, ... Theo tôi được biết, từ prefeel ...").
PROMPT_ECHO = re.compile(
    r"Nhắc lại:\s*chỉ dùng chữ Quốc ngữ|chỉ trả về bản dịch|Thuật ngữ bắt buộc\s*:|Dịch đoạn tiếng Anh sau|"
    r"không dùng chữ Hán|dịch hết mọi từ tiếng Anh",
    re.IGNORECASE,
)


# Token phải giữ nguyên từng ký tự: đường dẫn ổ đĩa/Unix, lời gọi hàm có dấu chấm, code inline.
# Chạy thật thấy qwen2.5:7b làm rơi "C:" khỏi "C:\workspace\app" mà không gate nào bắt.
TECHNICAL_TOKEN_PATTERN = re.compile(
    r"`[^`\n]+`"
    r"|[A-Za-z]:\\[^\s,;\"'“”()]+"
    r"|(?<![\w.])/(?:[\w.-]+/)+[\w.-]*"
    r"|\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\(\)"
)


# Từ tiếng Anh viết thường không trùng âm tiết tiếng Việt nào: còn trong bản dịch là chưa dịch hết.
# Chạy thật thấy qwen2.5:7b để lại "nobody chạm vào piano". Tránh các từ trùng âm tiết Việt
# ("to", "do", "in", "me", "an"...), từ kỹ thuật hay giữ nguyên ("build", "client") và chữ viết hoa.
ENGLISH_RESIDUE_WORDS = {
    "the", "and", "of", "with", "that", "this", "these", "those", "is", "are", "was", "were", "been",
    "have", "has", "had", "would", "could", "should", "nobody", "somebody", "everybody", "anybody",
    "nothing", "something", "everything", "anything", "never", "always", "said", "because", "which",
    "what", "when", "where", "there", "their", "they", "she", "his", "her", "him", "from", "into",
    # Không có "than": cũng là từ tiếng Việt (than thở, than phiền).
    "about", "then", "just", "very", "only", "also", "not", "however", "therefore", "although",
}
ENGLISH_WORD = re.compile(r"(?<![\w.`/\\-])[a-z]+(?![\w.`/\\-])")
EITHER_PATTERN = re.compile(r"\beither\b", re.IGNORECASE)
BOTH_PATTERN = re.compile(r"\bboth\b", re.IGNORECASE)
CA_HAI_PATTERN = re.compile(r"\bcả hai\b", re.IGNORECASE)
UNTRANSLATED_LABEL = re.compile(r"\b(?:Chapter|Section|Figure|Table|Appendix|Part)\s+\d+(?:\.\d+)*\b")


# Từ chức năng tiếng Anh viết hoa ở đầu câu mà từ kế tiếp viết thường: chạy trên sách thật thấy
# "Since sinh và tử dường như…". Từ kế tiếp viết hoa (tên sách "The Road Less Traveled") thì bỏ qua.
SENTENCE_START_ENGLISH = re.compile(
    r"(?:^|(?<=[.!?…:;]\s)|(?<=[“\"(]))"
    r"(Since|The|However|Although|Because|When|While|If|But|And|This|That|These|Those|There|It|What|Which|Nobody|Nothing|Everything)"
    r"\s+(?=[a-zà-ỹđ])"
)


_TOKEN = re.compile(r"[^\s“”\"()\[\]—–,;:.!?]+")


def _english_residue(text: str) -> List[str]:
    prose = URL_PATTERN.sub(" ", TECHNICAL_TOKEN_PATTERN.sub(" ", text or ""))
    tokens = [token.strip("'‘’") for token in _TOKEN.findall(prose)]
    tokens = [token for token in tokens if token]
    words = set()

    def neighbour(index: int, step: int) -> str:
        # Bỏ qua các từ chức năng liền nhau ("of the") để tới từ nội dung gần nhất.
        index += step
        while 0 <= index < len(tokens) and tokens[index] in ENGLISH_RESIDUE_WORDS:
            index += step
        return tokens[index] if 0 <= index < len(tokens) else ""

    for index, token in enumerate(tokens):
        if token not in ENGLISH_RESIDUE_WORDS:
            continue
        if neighbour(index, -1)[:1].isupper() and neighbour(index, 1)[:1].isupper():
            # Từ chức năng giữa hai từ viết hoa là một phần tên riêng/tựa sách giữ nguyên tiếng Anh
            # ("Psychology and Religion", "Journey of the Magi").
            continue
        words.add(token)
    words.update(match.group(1) for match in SENTENCE_START_ENGLISH.finditer(prose))
    return sorted(words)


# Chạy sách thật: qwen2.5:7b để lọt từ nội dung tiếng Anh ("strong", "passively", "parable", "discipline
# thường là…") và cả từ méo ("individul", "Therapeut") mà danh sách hư từ ở trên không bắt được.
# Từ chỉ gồm chữ không dấu mà không ghép được thành một âm tiết tiếng Việt (phụ âm đầu + vần) thì
# không phải tiếng Việt. Từ có dấu tiếng Việt ("Kitô", "ôxy") là phiên âm, bỏ qua.
_VI_INITIAL = "(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|b|c|d|g|h|k|l|m|n|p|r|s|t|v|x)?"
_VI_SYLLABLE = re.compile(f"^{_VI_INITIAL}[aeiouy]+(?:ng|nh|ch|c|m|n|p|t)?$")
_ASCII_WORD = re.compile(r"(?<![\w@#$%&+=/\\.-])[A-Za-z]+(?![\w@/\\-]|\.\w)")
_PARENTHESIZED = re.compile(r"\([^()]*\)")
# Một từ đứng riêng trong ngoặc là chữ đang được bàn tới (từ “conscious” có gốc Latinh…), không phải lọt.
_QUOTED_SINGLE_WORD = re.compile(r"[“\"‘']([A-Za-z]+)[”\"’']")
_LOWER_ROMAN = re.compile(r"^x{0,3}(?:ix|iv|v?i{0,3})$")
# Từ được chính nguồn giải nghĩa phải giữ nguyên: "the Latin con, meaning “with,” and scire, meaning “to know.”"
_DEFINED_WORD = re.compile(r"\b([A-Za-z]+)\b[,\"”'’]*\s+(?:meaning|means)\b")
# Cả cụm trong ngoặc được giải nghĩa cũng giữ nguyên: ‘‘Dilige et quod vis fac,” meaning “If you are loving…”.
_DEFINED_PHRASE = re.compile(r"[“\"‘']+([^”\"’]{2,80}?)[,.]?[”\"’']+\s+(?:meaning|means)\b")
# Model đôi khi bình luận về chính bản dịch ("… không phù hợp với ngữ cảnh, nên cần điều chỉnh thành…", "(Note: …").
# Trên 908 đoạn đã dịch của sách thật mẫu này không bắt nhầm đoạn nào.
MODEL_COMMENTARY = re.compile(
    r"\(\s*(?:Note|Lưu ý|Ghi chú)\s*:|(?:^|[.!?…:”\"]\s+)(?:Note|Lưu ý|Ghi chú)\s*:"
    r"|\bdưới đây là bản dịch\b|\bbản dịch (?:được |đã )?(?:sửa|điều chỉnh|đúng|chính xác)|\bdịch lại (?:thành|sang|như)\b"
    r"|\bdịch sang tiếng Việt\b|\bbản gốc\b|\bnguyên văn\b|\bcải chính\b|\bđiều chỉnh thành\b|\bsửa lại đúng\b"
    r"|\bkhông phù hợp với ngữ cảnh\b",
    re.IGNORECASE,
)
# Model bỏ dở câu rồi xả dấu câu: "bị đánh đập,, ——..", "tắm rửa cho.. “”..;.?...", "từ “”. Có phải".
# Trên 908 đoạn đã dịch của sách thật mẫu này bắt 19 đoạn, đọc lại cả 19 đều là rác thật.
PUNCTUATION_GARBAGE = re.compile(
    r",\s*,|;;|—\s*—|–\s*–|“\s*”|\.\s*\?|\?\s*\.(?!\.)|(?<!v\.v)(?<!\.)\.\.(?!\.)|[,;:]\s*[.!?](?!\.)"
)
# Nguồn đang nói về chính việc dịch hay ghi chú thì các từ trên có thể là nội dung thật.
COMMENTARY_EXEMPT_SOURCE = re.compile(r"translat|\boriginal\b|\bnote\b|\bverbatim\b", re.IGNORECASE)
# Từ mượn và đơn vị đã thành thông dụng trong văn viết tiếng Việt.
VIETNAMESE_LOANWORDS = {
    "ok", "email", "internet", "online", "offline", "video", "web", "website", "blog", "game", "file", "app",
    "wifi", "pizza", "taxi", "radio", "piano", "guitar", "violin", "jazz", "rock", "pop", "golf", "tennis",
    "menu", "show", "shop", "laptop", "smartphone", "robot", "virus", "vitamin", "protein", "sandwich",
    "hamburger", "cocktail", "selfie", "livestream", "marketing", "startup", "logo", "slogan", "poster",
    "bar", "sofa", "opera", "yoga", "acid", "axit", "ballet", "karaoke", "logic", "ego", "tivi",
    "electron", "proton", "neutron", "entropy", "alpha", "beta", "gamma", "delta", "omega",
    "kg", "km", "cm", "mm", "ml", "mg", "kw", "kwh", "mb", "gb", "tb", "kb", "hz", "mhz", "ghz", "ppm",
    "gram", "inch", "feet", "ounce", "pound", "yard", "mile",
    # Từ mượn thường giữ nguyên trong sách tiếng Việt (dịch thử Stumbling on Happiness).
    "cholesterol", "macaron", "bingo", "noir",
    # Viết tắt trong trích dẫn, chú thích.
    "tr", "ed", "eds", "trans", "vol", "pp", "ibid", "etc", "vs",
    # Tiền tố đo lường viết liền như tiếng Việt vẫn viết ("mili giây", "công nghệ nano") và đơn vị thời gian.
    "mili", "micro", "nano", "kilo", "mega", "giga", "ms", "ns",
}
# Tên ngôn ngữ lập trình/công cụ: trong văn bản kỹ thuật, nguồn hay viết thường ("rust library", "python deployments")
# còn bản dịch viết hoa đúng tên ("thư viện Rust"); cả hai đều không phải chữ chưa dịch.
_PROGRAMMING_NAMES = {
    "python", "rust", "ruby", "java", "perl", "bash", "linux", "docker", "kotlin", "swift", "scala", "haskell",
    "javascript", "typescript", "golang", "php", "lua", "nodejs", "node", "git", "github", "gitlab",
}


# Sau chữ có dấu, vần tiếng Việt chỉ còn tối đa 3 chữ cái ("ô"+"ng", "ế"+"ng"). Dài hơn là chữ bị dính với
# tiếng Anh: chạy sách thật thấy "khôngscious", "Củasciousness", "tráiDependency" lọt qua vì kiểm tra từ
# ASCII không thấy biên từ. Chiều ngược lại, trước chữ có dấu tiếng Việt có tối đa 4 chữ cái ("nghi"+"êng",
# "thuy"+"ết"); từ 5 chữ trở lên là tiếng Anh dính vào trước: lượt sửa câu viết "coldhearted" thành "coldlòng".
_FUSED_WORD = re.compile(r"\w*(?:(?![A-Za-z0-9_])\w)[A-Za-z]{5,}\w*|\w*[A-Za-z]{5,}(?:(?![A-Za-z0-9_])\w)\w*")


_VI_SYLLABLES = re.compile(f"^(?:{_VI_INITIAL}[aeiouy]+(?:ng|nh|ch|c|m|n|p|t)?)+$")
_LETTER_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
# Sau NFD: dấu móc của ơ/ư, dấu trăng của ă, dấu hỏi, dấu nặng, cùng chữ đ chỉ có trong chữ Việt.
_VIETNAMESE_ONLY = re.compile("[đĐ" + "".join(chr(code) for code in (0x031B, 0x0306, 0x0309, 0x0323)) + "]")


def _strip_vietnamese_marks(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").lower()


def _malformed_words(source_text: str, text: str) -> List[str]:
    """Chữ có dấu tiếng Việt mà không tách được thành các âm tiết tiếng Việt: chữ ngoại dính vào chữ Việt.

    Dịch thử William James: "the Absolute" thành "Đếextrême" mà kiểm chữ dính cũ (cần 5 chữ ASCII liền) bỏ lọt.
    Phiên âm viết liền ("Giêsu", "Kitô") vẫn tách được thành âm tiết nên không bị bắt. Trên 908 đoạn đã dịch của
    sách thật, kiểm này bắt 40 chỗ, cả 40 đều là chữ rác ("khôngconscious", "phépiddles").
    """
    source = source_text or ""
    found = []
    for match in _LETTER_TOKEN.finditer(text or ""):
        word = match.group(0)
        if word.isascii() or word in source:
            continue
        if not all(unicodedata.normalize("NFD", char)[0].isascii() or char in "đĐ" for char in word):
            continue  # bộ chữ khác (Hán, Kirin…) do kiểm bộ chữ xử lý
        if _VI_SYLLABLES.match(_strip_vietnamese_marks(word)) or word in found:
            continue
        # Phiên âm tên riêng có thể có cụm phụ âm ("Mátxcơva") nhưng vẫn mang chữ chỉ tiếng Việt mới có (ơ, ư, ă, đ,
        # dấu hỏi, dấu nặng). Chữ rác là một âm tiết Việt trọn vẹn dính với một chữ ngoại không có các dấu đó.
        if any(
            _VI_SYLLABLE.match(_strip_vietnamese_marks(word[:cut]))
            and not _VIETNAMESE_ONLY.search(unicodedata.normalize("NFD", word[cut:]))
            for cut in range(1, len(word) - 2)
        ):
            found.append(word)
    return found


def _title_like(source_text: str) -> bool:
    """Nguồn là tiêu đề viết hoa từng chữ ("Miscellaneous Objections to the Theory of Natural Selection")."""
    source = (source_text or "").strip()
    # Trích dẫn sách ("* The Prophet (New York: Knopf, 1951)") và lời thoại có tên người không phải tiêu đề.
    if not source or len(source.split()) > 12 or re.search(r"[\d“”\"]", source) or source.endswith("."):
        return False
    words = re.findall(r"[A-Za-z]{4,}", source)
    return bool(words) and sum(word[0].isupper() for word in words) >= 0.6 * len(words)


def _english_common_word(word: str) -> bool:
    try:
        from app.services.translation.vocabulary_hints import VocabularyHints
        return bool(VocabularyHints().senses(word))
    except Exception:
        return False


_DICTIONARY_PROBE: Optional[bool] = None


def _dictionary_ready() -> bool:
    """Không có từ điển thì mọi từ đều là "không có trong từ điển": luật dựa trên từ điển phải tắt."""
    global _DICTIONARY_PROBE
    if _DICTIONARY_PROBE is None:
        _DICTIONARY_PROBE = _english_common_word("water") and _english_common_word("library")
    return _DICTIONARY_PROBE


# Dấu hiệu đoạn nguồn là văn bản kỹ thuật: code, URL, tên có chữ số (libxml2, V2, HTTP/1.1), camelCase/PascalCase
# (ElementTree, iPhone; trừ họ người như McDonald), snake_case, C++.
_TECHNICAL_HINT = re.compile(
    r"`|https?://|\b[A-Za-z]+\d\w*|\b\w+/\d|\b[a-z]+[A-Z]\w*"
    r"|\b(?!(?:Mc|Mac|De|Di|Da|Du|La|Le|Van|Von)[A-Z])[A-Z][a-z]+[A-Z][a-z]\w*|\b[a-z]+_\w+|[A-Za-z]\+\+"
)
# Khóa tham chiếu của link markdown ("[ASGI specification][asgi]", "[contrib]: https://…") phải giữ nguyên.
_MARKDOWN_REFERENCE_KEY = re.compile(r"\]\[([^\]\n]+)\]|^\s*\[([^\]\n]+)\]:", re.MULTILINE)
_LONE_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.+-]*")
_URL_HOST = re.compile(r"^(?:https?://)?[^/\s]+", re.IGNORECASE)


def _technical_document(locked_glossary) -> bool:
    """Tài liệu loại TECHNICAL: bộ thuật ngữ gate (glossary_validation_terms) mang preset lĩnh vực TECHNICAL.

    Bộ này được truyền giống nhau tới mọi nơi chạy gate (engine, lưu bản dịch, bộ nhớ dịch, QA), nên đây là tín hiệu loại
    tài liệu duy nhất mà mọi lượt kiểm đều thấy như nhau.
    """
    if not isinstance(locked_glossary, (list, tuple)):
        return False
    return any(isinstance(entry, dict) and str(entry.get("domain") or "").upper() == "TECHNICAL" for entry in locked_glossary)


def _source_identifiers(source_text: str, technical_document: bool = False) -> Set[str]:
    """Tên thư viện, gói, khóa link trong nguồn mà bản dịch phải giữ nguyên, dù viết thường.

    Dịch thử README của 61 thư viện Python: đoạn đúng bị chặn vì "pydantic", "lxml" (tiêu đề và đầu câu), "libxslt",
    khóa link "[contrib]". Chữ viết thường không có trong từ điển chỉ được coi là tên khi đoạn nguồn mang dấu hiệu kỹ thuật:
    sách văn học ("mọi thứ chúng ta cathect") thì chữ không có trong từ điển vẫn là chữ chưa dịch.
    """
    source = (source_text or "").strip()
    found: Set[str] = set()
    if _LONE_IDENTIFIER.fullmatch(source):
        # Tiêu đề chỉ có một chữ viết thường là tên gói/lệnh ("pydantic", "requests"); tiêu đề thường viết hoa.
        found.add(source)
    for match in _MARKDOWN_REFERENCE_KEY.finditer(source):
        key = match.group(1) or match.group(2) or ""
        found.update(word.lower() for word in re.findall(r"[A-Za-z]+", key))
    technical = technical_document or bool(_TECHNICAL_HINT.search(source))
    if technical:
        found.update(word for word in re.findall(r"[a-z]+", source.lower()) if word in _PROGRAMMING_NAMES)
    if technical and _dictionary_ready():
        prose = TECHNICAL_TOKEN_PATTERN.sub(" ", URL_PATTERN.sub(" ", source))
        for word in set(re.findall(r"(?<![\w.-])[a-z]{2,}(?![\w-])", prose)):
            if word not in ENGLISH_RESIDUE_WORDS and not _english_common_word(word):
                found.add(word)
    return found


def _beside_name(text: str, start: int, end: int) -> bool:
    """Từ viết hoa đứng cạnh một từ viết hoa không dấu khác là một phần của tên ("Adam Smith")."""
    before = re.search(r"([A-Za-z]+)\W*$", text[:start])
    after = re.match(r"\W*([A-Za-z]+)", text[end:])
    return any(
        neighbour and neighbour.group(1)[0].isupper() and not _VI_SYLLABLE.match(neighbour.group(1).lower())
        for neighbour in (before, after)
    )


def _short_source_expanded(source: str, target: str) -> bool:
    """Tiêu đề "Entropy and Original Sin" thành tiêu đề cộng cả đoạn giải thích entropy (đã được duyệt ở sách thật).

    Trên 978 đoạn đã dịch (sách thật và dịch thử 7 cuốn), kiểm này bắt 4 đoạn, cả 4 đều là nội dung model tự viết thêm.
    """
    if not source or "\n" in source or len(source.split()) > 15:
        return False
    lines = [line for line in target.splitlines() if line.strip()]
    if len(lines) > 1:
        return True
    # Tiêu đề một hai từ thành cả câu định nghĩa ("Packfiles" -> "Packfiles là tập tin chứa dữ liệu Git được nén.").
    # Trên 1.118 cặp đã dịch, không tiêu đề ngắn nào dịch đúng mà dài quá gấp đôi cộng 18 ký tự.
    is_label = not re.search(r"[.!?…:;”\"’)]$", source)
    if is_label and len(source.split()) <= 6 and len(target) > len(source) + 25 and len(target) > 2.5 * len(source):
        return True
    return len(target) > max(3 * len(source), len(source) + 40)


def _fused_words(source_text: str, text: str) -> List[str]:
    source = source_text or ""
    # Tên riêng có dấu trong nguồn ("Éluard") vẫn được giữ nguyên.
    return [match.group(0) for match in _FUSED_WORD.finditer(text or "") if match.group(0) not in source]


def _non_vietnamese_words(source_text: str, text: str, locked_glossary=None) -> List[str]:
    prose = TECHNICAL_TOKEN_PATTERN.sub(" ", URL_PATTERN.sub(" ", text or ""))
    # "học máy (machine learning)": giữ thuật ngữ gốc trong ngoặc là cách viết chuẩn.
    prose = _PARENTHESIZED.sub(" ", prose)
    prose = _QUOTED_SINGLE_WORD.sub(
        lambda match: " " if match.group(1).lower() in (source_text or "").lower() else match.group(0), prose,
    )
    # Tên miền không tính: "Kozea (https://kozea.fr/)" vẫn là tên riêng. Đường dẫn thì vẫn tính, vì chữ link để nguyên
    # tiếng Anh thường lặp lại đúng đường dẫn đó ("[Contributing to Pydantic](https://docs.pydantic.dev/contributing/)").
    source_lower_words = set(re.findall(
        r"\b[a-z]+\b", URL_PATTERN.sub(lambda match: " " + _URL_HOST.sub(" ", match.group(0)), source_text or ""),
    ))
    allowed = {
        word.lower()
        for _, target, _, variants in GlossaryValidator._entries(locked_glossary or {})
        for value in [target, *variants]
        for word in re.findall(r"[A-Za-z]+", value or "")
    }
    allowed.update(word.lower() for word in _DEFINED_WORD.findall(source_text or ""))
    allowed.update(_source_identifiers(source_text, _technical_document(locked_glossary)))
    # Tên lệnh, tên hàm trong code của nguồn (`git svn rebase`) giữ nguyên là đúng, kể cả khi model bỏ dấu `.
    for span in re.findall(r"`([^`\n]+)`", source_text or ""):
        allowed.update(word.lower() for word in re.findall(r"[A-Za-z]+", span))
    for phrase in _DEFINED_PHRASE.findall(source_text or ""):
        allowed.update(word.lower() for word in re.findall(r"[A-Za-z]+", phrase))
    # Tên khoa học hai phần chép đúng từ nguồn ("Aplysia parvula", "Macaca mulatta"): phần sau viết thường nhưng không
    # phải chữ tiếng Anh cần dịch.
    for genus, species in re.findall(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b", source_text or ""):
        if f"{genus} {species}" in (text or "") and not (
            _english_common_word(genus.lower()) or _english_common_word(species)
        ):
            allowed.add(species)
    found = list(dict.fromkeys(_fused_words(source_text, prose) + _malformed_words(source_text, prose)))
    title_like = _title_like(source_text or "")
    for match in _ASCII_WORD.finditer(prose):
        word = match.group(0)
        lower = word.lower()
        if (
            len(word) < 2 or lower in VIETNAMESE_LOANWORDS or lower in allowed or lower in ENGLISH_RESIDUE_WORDS
            or _VI_SYLLABLE.match(lower) or _LOWER_ROMAN.match(word)
        ):
            continue  # hư từ tiếng Anh đã có _english_residue xử lý (kể cả trong tựa sách)
        if any(char.isupper() for char in word[1:]):
            continue  # viết tắt, số La Mã, camelCase/iPhone
        if word[0].isupper() and lower not in source_lower_words:
            # Tiêu đề viết hoa từng chữ: mọi từ đều viết hoa nên không phân biệt được tên riêng theo chữ hoa.
            # Dịch thử Darwin: "Miscellaneous Objections…" thành "Các Objection Miễn cưỡng" mà gate cho qua.
            # Khi đó từ có trong từ điển và không đứng cạnh một tên khác thì vẫn là chữ sót.
            if not (title_like and _english_common_word(lower) and not _beside_name(prose, match.start(), match.end())):
                continue  # tên riêng, tựa sách giữ nguyên ("Freud", "How People Change"); "Strong" đầu câu vẫn bắt
        if word not in found:
            found.append(word)
    return found


def _technical_tokens(text: str) -> List[str]:
    return [token.rstrip(".,;:") for token in TECHNICAL_TOKEN_PATTERN.findall(text or "")]


def _word_set(text: str) -> Set[str]:
    return {word.casefold() for word in WORD_PATTERN.findall(text or "")}


def _expand_counter(values: Counter) -> List[str]:
    """Trả về các giá trị theo số lần xuất hiện để thông báo không làm mất bản sao."""
    expanded: List[str] = []
    for value, count in values.items():
        expanded.extend([value] * count)
    return expanded


def _url_counts(text: str) -> Counter:
    """Chuẩn hóa dấu câu kết câu để URL hợp lệ không bị coi là URL khác."""
    return Counter(url.rstrip(".,;:!?") for url in URL_PATTERN.findall(text or ""))


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    hard_fail: bool
    score: float
    issues: List[Dict[str, str]] = field(default_factory=list)


class TranslationQualityGate:
    """Tập trung các kiểm tra bắt buộc trước khi lưu bản dịch và TM."""

    @staticmethod
    def _issue(code: str, severity: str, message: str) -> Dict[str, str]:
        return {"code": code, "severity": severity, "message": message}

    @staticmethod
    def _content_addition_issues(source: str, target: str, context_translations: Sequence[str]) -> List[Dict[str, str]]:
        """Bắt các dạng chèn nội dung đã gặp khi chạy qwen2.5:7b thật.

        Model nhỏ hay kéo câu từ đoạn trước vào bản dịch (thêm câu hỏi, chép lại câu đã dịch)
        và làm rơi dấu ngoặc kép mở đầu lời thoại.
        """
        issues: List[Dict[str, str]] = []
        echo = PROMPT_ECHO.search(target)
        if echo and not PROMPT_ECHO.search(source):
            issues.append(TranslationQualityGate._issue(
                "PROMPT_ECHO", "ERROR", f"Bản dịch chép lại chỉ dẫn của prompt: \"{echo.group(0)}\".",
            ))
        source_questions = URL_PATTERN.sub(" ", source).count("?")
        target_questions = URL_PATTERN.sub(" ", target).count("?")
        if target_questions > source_questions:
            issues.append(TranslationQualityGate._issue(
                "CONTENT_ADDITION", "ERROR",
                f"Bản dịch có {target_questions} câu hỏi trong khi nguồn có {source_questions}; có thể đã chèn nội dung.",
            ))
        source_quotes = len(QUOTE_PATTERN.findall(source))
        target_quotes = len(QUOTE_PATTERN.findall(target))
        # Nguồn mở đầu bằng lời thoại mà bản dịch lại bắt đầu bằng chữ rồi mới gặp dấu đóng ngoặc:
        # model đã rơi dấu mở (số dấu vẫn chẵn nếu rơi cả cặp nên kiểm đếm không bắt được).
        opening_lost = (
            source[:1] in "\"“" and target[:1] not in "\"“-–—" and OPENING_QUOTE_LOST.match(target)
        )
        if opening_lost or (target_quotes % 2 == 1 and source_quotes % 2 == 0):
            issues.append(TranslationQualityGate._issue(
                "QUOTE_UNBALANCED", "ERROR", "Dấu ngoặc kép trong bản dịch bị lệch (thiếu mở hoặc đóng lời thoại).",
            ))
        previous_sentences = [
            _word_set(sentence) for text in context_translations for sentence in SENTENCE_SPLIT.split(text or "")
        ]
        target_sentences = SENTENCE_SPLIT.split(target)
        # Bản dịch không dài hơn nguồn thì câu ngắn trùng đoạn trước là do tác giả lặp ý ("You do have a problem."),
        # không phải model chép thêm: chạy sách thật thấy bị chặn oan cả đoạn. Câu dài trùng hẳn vẫn bị bắt.
        added_sentences = len(target_sentences) > len(SENTENCE_SPLIT.split(source))
        minimum_words = CONTEXT_LEAK_MIN_WORDS if added_sentences else CONTEXT_LEAK_MIN_WORDS_SAME_LENGTH
        for sentence in target_sentences:
            words = _word_set(sentence)
            if len(words) < minimum_words:
                continue
            if any(len(words & other) / len(words | other) >= CONTEXT_LEAK_OVERLAP for other in previous_sentences if other):
                issues.append(TranslationQualityGate._issue(
                    "CONTEXT_LEAK", "ERROR", f"Câu \"{sentence.strip()[:80]}\" lặp lại bản dịch của đoạn trước.",
                ))
                break
        return issues

    def validate(
        self,
        source_text: str,
        translated_text: str,
        locked_glossary: Dict[str, str],
        context_translations: Optional[Sequence[str]] = None,
    ) -> QualityGateResult:
        source = (source_text or "").strip()
        target = (translated_text or "").strip()
        issues: List[Dict[str, str]] = []
        if target and target == citation_passthrough(source).strip() and is_bibliographic_citation(source):
            # Mục chú thích chỉ là trích dẫn tài liệu được giữ nguyên văn (citation_policy).
            return QualityGateResult(True, False, 1.0, [])

        if not target:
            issues.append(self._issue("EMPTY_TRANSLATION", "ERROR", "Bản dịch đang để trống."))
        else:
            language = validate_target_language(source, target)
            if not language.passed:
                issues.append(self._issue(language.reason or "WRONG_TARGET_LANGUAGE", "ERROR", "Bản dịch không đạt yêu cầu ngôn ngữ đích."))

            numeric = NumericValidator.validate(source, target)
            if numeric.missing:
                missing_numbers = ", ".join(token.raw for token in numeric.missing)
                issues.append(self._issue("NUMBER_MISMATCH", "ERROR", f"Thiếu hoặc sai số liệu từ nguồn: {missing_numbers}"))
            if numeric.unexpected:
                unexpected_numbers = ", ".join(token.raw for token in numeric.unexpected)
                issues.append(self._issue("NUMBER_ADDITION", "ERROR", f"Bản dịch bổ sung số liệu không có trong nguồn: {unexpected_numbers}"))

            source_url_counts = _url_counts(source)
            target_url_counts = _url_counts(target)
            missing_urls = _expand_counter(source_url_counts - target_url_counts)
            unexpected_urls = _expand_counter(target_url_counts - source_url_counts)
            if missing_urls:
                issues.append(self._issue("URL_MISMATCH", "ERROR", f"Thiếu URL từ nguồn: {', '.join(missing_urls)}"))
            if unexpected_urls:
                issues.append(self._issue("URL_ADDITION", "ERROR", f"Bản dịch bổ sung URL không có trong nguồn: {', '.join(unexpected_urls)}"))

            references = ReferenceValidator.validate(source, target)
            if references.missing:
                missing_refs = ", ".join(token.raw for token in references.missing)
                issues.append(self._issue("REFERENCE_MISMATCH", "ERROR", f"Thiếu hoặc sai tham chiếu: {missing_refs}"))
            if references.unexpected:
                unexpected_refs = ", ".join(token.raw for token in references.unexpected)
                issues.append(self._issue("REFERENCE_ADDITION", "ERROR", f"Bản dịch bổ sung tham chiếu không có trong nguồn: {unexpected_refs}"))

            garbage = next((m.group(0) for m in PUNCTUATION_GARBAGE.finditer(target) if m.group(0) not in source), None)
            if garbage:
                issues.append(self._issue(
                    "PUNCTUATION_GARBAGE", "ERROR", f"Bản dịch có chuỗi dấu câu vô nghĩa ({garbage!r}), thường là câu bị bỏ dở.",
                ))

            if _short_source_expanded(source, target):
                issues.append(self._issue(
                    "SHORT_SOURCE_EXPANSION", "ERROR",
                    "Nguồn chỉ là một dòng ngắn (tiêu đề, nhãn) nhưng bản dịch có nhiều dòng hoặc dài gấp nhiều lần: "
                    "model đã viết thêm nội dung.",
                ))

            if len(source) > 100:
                ratio = len(target) / max(len(source), 1)
                if ratio < 0.25 or ratio > 3.5:
                    issues.append(self._issue("LENGTH_ANOMALY", "WARNING", f"Tỷ lệ độ dài bất thường: {ratio:.2f}."))

            glossary = GlossaryValidator.validate(source, target, locked_glossary or {})
            for violation in glossary.violations:
                issues.append(self._issue(
                    "GLOSSARY_MISMATCH",
                    "ERROR",
                    f"Thuật ngữ khóa '{violation.source_term}' phải dùng '{violation.expected_target}'.",
                ))
            for warning in glossary.warnings:
                allowed = ", ".join(warning.allowed_variants) or warning.expected_target
                issues.append(self._issue(
                    "GLOSSARY_PREFERENCE",
                    "WARNING",
                    f"Thuật ngữ mềm '{warning.source_term}' ưu tiên '{warning.expected_target}' (cho phép: {allowed}).",
                ))

            issues.extend(self._content_addition_issues(source, target, context_translations or []))

            commentary = None if COMMENTARY_EXEMPT_SOURCE.search(source) else MODEL_COMMENTARY.search(target)
            if commentary:
                issues.append(self._issue(
                    "MODEL_COMMENTARY", "ERROR",
                    f"Bản dịch có lời bình của model thay vì nội dung nguồn: \"{commentary.group(0).strip()}\".",
                ))

            if EITHER_PATTERN.search(source) and not BOTH_PATTERN.search(source) and CA_HAI_PATTERN.search(target):
                issues.append(self._issue(
                    "QUANTIFIER_SHIFT", "ERROR",
                    "Nguồn nói 'either' (bất kỳ/một trong hai) nhưng bản dịch thành 'cả hai'.",
                ))

            residue = _english_residue(target)
            residue += [word for word in _non_vietnamese_words(source, target, locked_glossary) if word not in residue]
            if residue:
                issues.append(self._issue(
                    "ENGLISH_RESIDUE", "ERROR",
                    f"Còn từ tiếng Anh chưa dịch trong văn xuôi: {', '.join(residue[:5])}",
                ))
            untranslated_labels = [
                match.group(0) for match in UNTRANSLATED_LABEL.finditer(target)
                if not UNTRANSLATED_LABEL.search(source) or match.group(0) in source
            ]
            if untranslated_labels:
                issues.append(self._issue(
                    "UNTRANSLATED_LABEL", "ERROR",
                    f"Nhãn tham chiếu chưa dịch: {', '.join(untranslated_labels[:3])} (vd 'Chapter 1' -> 'Chương 1').",
                ))

            missing_tokens = [token for token in _technical_tokens(source) if token not in target]
            if missing_tokens:
                issues.append(self._issue(
                    "TECHNICAL_TOKEN_MISMATCH", "ERROR",
                    f"Thiếu hoặc bị sửa token kỹ thuật phải giữ nguyên: {', '.join(missing_tokens[:5])}",
                ))

            polarity = PolarityValidator.validate(source, target)
            if polarity.explicit_negation_lost:
                issues.append(self._issue("NEGATION_LOSS", "ERROR", "Bản dịch đã làm mất phủ định tường minh của nguồn."))
            elif polarity.ambiguous_negation_risk:
                issues.append(self._issue("NEGATION_RISK", "WARNING", "Bản dịch có nguy cơ làm mất ý phủ định."))

        hard_fail = any(issue["severity"] == "ERROR" for issue in issues)
        passed = not hard_fail
        penalty = sum(0.25 if issue["severity"] == "ERROR" else 0.1 for issue in issues)
        return QualityGateResult(passed, hard_fail, max(0.0, round(1.0 - penalty, 2)), issues)

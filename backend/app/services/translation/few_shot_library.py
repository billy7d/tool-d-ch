"""Thư viện few-shot đã biên tập cho các miền dịch tiếng Việt.

Các ví dụ trong file này chỉ mô tả cách xử lý câu và văn phong. Bộ chọn runtime
không xem chúng là dữ kiện, thuật ngữ khóa hay nguồn sự thật cho tài liệu hiện tại.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple


FEW_SHOT_LIBRARY_VERSION = "curated-few-shots-v1"
SUPPORTED_DOMAINS = (
    "GENERAL", "BUSINESS", "FINANCE", "SELF_HELP",
    "TECHNICAL", "ACADEMIC", "LEGAL", "LITERATURE",
)


@dataclass(frozen=True)
class FewShotExample:
    example_id: str
    domain: str
    node_type: str
    source: str
    target: str
    patterns: Tuple[str, ...] = ()
    modes: Tuple[str, ...] = ("NATURAL", "BALANCED")
    register: str = "neutral"

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["patterns"] = list(self.patterns)
        value["modes"] = list(self.modes)
        return value


def _items(
    domain: str,
    values: Iterable[Tuple[str, str, str, Tuple[str, ...]]],
    register: str = "neutral",
) -> List[FewShotExample]:
    return [
        FewShotExample(
            example_id=f"{domain.lower()}-{index:02d}",
            domain=domain,
            node_type=node_type,
            source=source,
            target=target,
            patterns=patterns,
            register=register,
        )
        for index, (source, target, node_type, patterns) in enumerate(values, start=1)
    ]


# Mỗi miền có 20 ví dụ nhỏ, đủ đa dạng để bộ chọn deterministic nhận diện mẫu câu.
CURATED_FEW_SHOTS: Dict[str, List[FewShotExample]] = {
    "GENERAL": _items("GENERAL", [
        ("It is important to note that the plan was changed in order to reduce delays.", "Cần lưu ý rằng kế hoạch đã được điều chỉnh để giảm chậm trễ.", "paragraph", ("nominalization",)),
        ("The issue was resolved after the team reviewed the evidence.", "Vấn đề được giải quyết sau khi nhóm xem xét các bằng chứng.", "paragraph", ("passive", "causality")),
        ("The new policy for employees working across several regional offices was announced yesterday.", "Chính sách mới dành cho nhân viên làm việc tại nhiều văn phòng khu vực đã được công bố hôm qua.", "paragraph", ("long_noun_phrase", "passive")),
        ("The report, which was published in June, is available online.", "Báo cáo được công bố vào tháng Sáu hiện đã có trên mạng.", "paragraph", ("relative_clause",)),
        ("If the weather improves, the event will move outdoors.", "Nếu thời tiết khá hơn, sự kiện sẽ được chuyển ra ngoài trời.", "paragraph", ("conditional",)),
        ("You may contact the library if you need a larger copy.", "Bạn có thể liên hệ với thư viện nếu cần bản sao lớn hơn.", "paragraph", ("modality", "conditional")),
        ("Because the road was closed, we took a different route.", "Vì con đường bị đóng, chúng tôi đã đi theo một tuyến khác.", "paragraph", ("causality",)),
        ("Alice thanked Bob after she received the revised file.", "Alice cảm ơn Bob sau khi nhận được tệp đã chỉnh sửa.", "paragraph", ("pronoun_chain",)),
        ("The short exercise helped break the ice before the meeting.", "Bài tập ngắn giúp mọi người bớt ngại ngùng trước cuộc họp.", "paragraph", ("idiom",)),
        ("The committee will look into the complaint next week.", "Ủy ban sẽ xem xét khiếu nại vào tuần tới.", "paragraph", ("phrasal_verb",)),
        ("The instructions are long, so we divided them into two paragraphs.", "Phần hướng dẫn khá dài nên chúng tôi chia thành hai đoạn.", "paragraph", ("sentence_split", "causality")),
        ("She opened the letter and immediately called her sister.", "Cô mở thư rồi lập tức gọi cho chị gái.", "paragraph", ("sentence_merge",)),
        ("However, the result does not answer the central question.", "Tuy nhiên, kết quả này chưa trả lời câu hỏi cốt lõi.", "paragraph", ("discourse_connector", "negation")),
        ("The package arrived on 12 March 2026 at 10:30 a.m.", "Gói hàng đến vào lúc 10 giờ 30 sáng ngày 12 tháng 3 năm 2026.", "paragraph", ("number",)),
        ("The proposal does not require additional funding.", "Đề xuất này không cần thêm kinh phí.", "paragraph", ("negation",)),
        ("\"Are you ready?\" he asked.", "\"Anh sẵn sàng chưa?\" anh hỏi.", "dialogue", ("pronoun_chain",)),
        ("First, save the file. Then close the application.", "Trước hết, hãy lưu tệp. Sau đó, đóng ứng dụng.", "list_item", ("discourse_connector", "sentence_split")),
        ("The room was quiet, and everyone waited for the announcement.", "Căn phòng yên tĩnh, mọi người đều chờ thông báo.", "paragraph", ("sentence_merge",)),
        ("This approach is simpler than the previous one.", "Cách tiếp cận này đơn giản hơn cách trước.", "paragraph", ("comparison",)),
        ("The changes affect both new and existing users.", "Các thay đổi ảnh hưởng đến cả người dùng mới lẫn người dùng hiện tại.", "paragraph", ("long_noun_phrase",)),
    ]),
    "BUSINESS": _items("BUSINESS", [
        ("The company has taken significant steps toward improving operational efficiency.", "Công ty đã có nhiều bước tiến đáng kể trong việc nâng cao hiệu quả hoạt động.", "paragraph", ("business_collocation", "nominalization")),
        ("We need to align the sales team with the new strategy.", "Chúng ta cần thống nhất đội ngũ kinh doanh với chiến lược mới.", "paragraph", ("business_collocation",)),
        ("The board approved the proposal after a detailed review.", "Hội đồng quản trị đã phê duyệt đề xuất sau khi xem xét kỹ lưỡng.", "paragraph", ("business_collocation", "causality")),
        ("The project was delivered on time despite several setbacks.", "Dự án được bàn giao đúng hạn dù gặp một số trở ngại.", "paragraph", ("passive", "discourse_connector")),
        ("The customer, who had used the service for years, renewed the contract.", "Khách hàng đã sử dụng dịch vụ nhiều năm đó đã gia hạn hợp đồng.", "paragraph", ("relative_clause", "business_collocation")),
        ("If demand continues to grow, the company may expand production.", "Nếu nhu cầu tiếp tục tăng, công ty có thể mở rộng sản xuất.", "paragraph", ("conditional", "modality")),
        ("The new pricing model is intended to reduce churn.", "Mô hình định giá mới nhằm giảm tỷ lệ khách hàng rời bỏ.", "paragraph", ("business_collocation", "nominalization")),
        ("The delay resulted from a shortage of qualified staff.", "Sự chậm trễ bắt nguồn từ tình trạng thiếu nhân sự đủ năng lực.", "paragraph", ("causality", "business_collocation")),
        ("The manager asked whether the team could follow up with the client.", "Quản lý hỏi liệu nhóm có thể liên hệ lại với khách hàng hay không.", "paragraph", ("phrasal_verb", "modality")),
        ("The two departments are on the same page about the launch.", "Hai bộ phận đã thống nhất về việc ra mắt.", "paragraph", ("idiom", "business_collocation")),
        ("However, the quarter ended below our original forecast.", "Tuy nhiên, quý này kết thúc với kết quả thấp hơn dự báo ban đầu.", "paragraph", ("discourse_connector", "business_collocation")),
        ("The company does not plan to reduce its research budget.", "Công ty không có kế hoạch cắt giảm ngân sách nghiên cứu.", "paragraph", ("negation", "business_collocation")),
        ("A clear owner was assigned to each workstream.", "Mỗi luồng công việc đều được giao cho một người phụ trách rõ ràng.", "paragraph", ("passive", "business_collocation")),
        ("Revenue grew 8% in the first half of the year.", "Doanh thu tăng 8% trong nửa đầu năm.", "paragraph", ("number", "business_collocation")),
        ("The partnership will create new opportunities for both sides.", "Quan hệ hợp tác sẽ tạo ra những cơ hội mới cho cả hai bên.", "paragraph", ("business_collocation",)),
        ("The team reduced the scope so that it could meet the deadline.", "Nhóm đã thu hẹp phạm vi để có thể đáp ứng thời hạn.", "paragraph", ("causality", "business_collocation")),
        ("The presentation was concise, but it covered the main risks.", "Bài trình bày ngắn gọn nhưng vẫn bao quát các rủi ro chính.", "paragraph", ("sentence_merge", "business_collocation")),
        ("The client asked us to set up a second review call.", "Khách hàng đề nghị chúng tôi sắp xếp thêm một cuộc gọi rà soát.", "paragraph", ("phrasal_verb", "business_collocation")),
        ("The initiative supports small businesses in regional markets.", "Sáng kiến này hỗ trợ các doanh nghiệp nhỏ tại những thị trường khu vực.", "paragraph", ("long_noun_phrase", "business_collocation")),
        ("The results were shared with stakeholders before the decision.", "Kết quả được chia sẻ với các bên liên quan trước khi đưa ra quyết định.", "paragraph", ("passive", "business_collocation")),
    ], register="professional"),
    "FINANCE": _items("FINANCE", [
        ("Revenue rose 12%, while the debt-to-equity ratio remained at 0.8.", "Doanh thu tăng 12%, trong khi hệ số nợ trên vốn chủ sở hữu vẫn ở mức 0,8.", "paragraph", ("number", "business_collocation")),
        ("The interest rate may remain high through the second quarter.", "Lãi suất có thể vẫn ở mức cao trong suốt quý II.", "paragraph", ("polysemy", "modality")),
        ("The investor expressed interest in the company after reviewing its return profile.", "Nhà đầu tư bày tỏ sự quan tâm đến công ty sau khi xem xét đặc điểm lợi nhuận của công ty.", "paragraph", ("polysemy", "long_noun_phrase")),
        ("The fund returned 6.5% in 2025.", "Quỹ đạt mức sinh lời 6,5% trong năm 2025.", "paragraph", ("polysemy", "number")),
        ("The gross margin improved from 31% to 34%.", "Biên lợi nhuận gộp cải thiện từ 31% lên 34%.", "paragraph", ("polysemy", "number")),
        ("The security is backed by a diversified portfolio.", "Chứng khoán này được bảo đảm bằng một danh mục đa dạng hóa.", "paragraph", ("polysemy", "passive")),
        ("The analyst changed the position after the earnings release.", "Chuyên viên phân tích đã thay đổi vị thế sau khi công bố kết quả kinh doanh.", "paragraph", ("polysemy", "causality")),
        ("The company reduced its exposure to foreign-currency risk.", "Công ty đã giảm mức độ phơi nhiễm với rủi ro ngoại tệ.", "paragraph", ("polysemy", "business_collocation")),
        ("If inflation accelerates, real returns could decline.", "Nếu lạm phát tăng tốc, lợi nhuận thực có thể giảm.", "paragraph", ("conditional", "modality")),
        ("The loan must be repaid within 30 days.", "Khoản vay phải được hoàn trả trong vòng 30 ngày.", "paragraph", ("modality", "number")),
        ("The report, which covers three markets, excludes private placements.", "Báo cáo bao quát ba thị trường nhưng không bao gồm các đợt phát hành riêng lẻ.", "paragraph", ("relative_clause", "negation")),
        ("The loss was caused by a one-time impairment charge.", "Khoản lỗ do một khoản chi phí suy giảm giá trị một lần gây ra.", "paragraph", ("causality", "polysemy")),
        ("However, cash flow remained positive during the period.", "Tuy nhiên, dòng tiền vẫn dương trong kỳ.", "paragraph", ("discourse_connector", "business_collocation")),
        ("The balance sheet does not include the pending acquisition.", "Bảng cân đối kế toán không bao gồm thương vụ mua lại đang chờ hoàn tất.", "paragraph", ("negation", "business_collocation")),
        ("The central bank raised the benchmark rate by 25 basis points.", "Ngân hàng trung ương tăng lãi suất chuẩn thêm 25 điểm cơ bản.", "paragraph", ("number", "business_collocation")),
        ("The valuation depends on assumptions about future growth.", "Định giá phụ thuộc vào các giả định về tăng trưởng trong tương lai.", "paragraph", ("nominalization", "business_collocation")),
        ("The portfolio manager took profits after the rally.", "Nhà quản lý danh mục chốt lời sau đợt tăng giá.", "paragraph", ("phrasal_verb", "business_collocation")),
        ("The market moved sharply, so the margin requirement was revised.", "Thị trường biến động mạnh nên yêu cầu ký quỹ đã được điều chỉnh.", "paragraph", ("causality", "passive")),
        ("The note matures on 15 September 2028.", "Trái phiếu này đáo hạn vào ngày 15 tháng 9 năm 2028.", "paragraph", ("number",)),
        ("The results suggest a modest improvement, not a reversal of the trend.", "Kết quả cho thấy mức cải thiện khiêm tốn, chứ chưa phải sự đảo chiều của xu hướng.", "paragraph", ("academic_hedging", "negation")),
    ], register="precise"),
    "SELF_HELP": _items("SELF_HELP", [
        ("Taking action is often more useful than waiting for the perfect moment.", "Hành động thường hữu ích hơn là chờ đến thời điểm hoàn hảo.", "paragraph", ("comparison",)),
        ("You do not have to solve everything today.", "Bạn không cần phải giải quyết mọi việc ngay hôm nay.", "paragraph", ("negation", "modality")),
        ("If you feel overwhelmed, start with one small task.", "Nếu cảm thấy quá tải, hãy bắt đầu bằng một việc nhỏ.", "paragraph", ("conditional",)),
        ("The habit was built through small, repeated choices.", "Thói quen được hình thành từ những lựa chọn nhỏ lặp đi lặp lại.", "paragraph", ("passive",)),
        ("Notice what your body is telling you before you push harder.", "Hãy để ý những tín hiệu cơ thể gửi đến trước khi cố gắng hơn nữa.", "paragraph", ("pronoun_chain",)),
        ("A short walk can clear your head after a difficult day.", "Một cuộc đi bộ ngắn có thể giúp bạn thư thái đầu óc sau một ngày khó khăn.", "paragraph", ("collocation",)),
        ("The exercise is designed to bring your attention back to the present.", "Bài tập này giúp bạn đưa sự chú ý trở lại với hiện tại.", "paragraph", ("nominalization",)),
        ("You may need to set boundaries with people you care about.", "Bạn có thể cần đặt ra ranh giới với những người mình quan tâm.", "paragraph", ("modality", "phrasal_verb")),
        ("She wrote down the thought and then let it go.", "Cô ghi lại suy nghĩ đó rồi buông bỏ nó.", "paragraph", ("sentence_merge", "pronoun_chain")),
        ("However, progress is rarely visible from one day to the next.", "Tuy nhiên, tiến bộ hiếm khi nhìn thấy được chỉ sau một ngày.", "paragraph", ("discourse_connector", "passive")),
        ("The goal is not to become a different person overnight.", "Mục tiêu không phải là biến thành một con người khác chỉ sau một đêm.", "paragraph", ("negation", "idiom")),
        ("When the plan fails, learn from it and begin again.", "Khi kế hoạch thất bại, hãy rút kinh nghiệm rồi bắt đầu lại.", "paragraph", ("conditional", "discourse_connector")),
        ("It is okay to ask for help before the problem grows.", "Bạn có thể nhờ giúp đỡ trước khi vấn đề trở nên nghiêm trọng hơn.", "paragraph", ("modality", "causality")),
        ("The advice is simple, but applying it takes patience.", "Lời khuyên đơn giản, nhưng áp dụng được lại cần sự kiên nhẫn.", "paragraph", ("sentence_merge",)),
        ("You can look back without living in the past.", "Bạn có thể nhìn lại mà không sống mãi trong quá khứ.", "paragraph", ("phrasal_verb", "negation")),
        ("Give yourself room to change your mind.", "Hãy cho bản thân không gian để thay đổi suy nghĩ.", "paragraph", ("idiom",)),
        ("The first step may feel uncomfortable, and that is normal.", "Bước đầu tiên có thể khiến bạn không thoải mái, và điều đó là bình thường.", "paragraph", ("modality", "sentence_merge")),
        ("Write three things you can control today.", "Hãy viết ra ba điều bạn có thể kiểm soát hôm nay.", "list_item", ("number", "modality")),
        ("The routine works best when it fits your real life.", "Thói quen này hiệu quả nhất khi phù hợp với cuộc sống thực tế của bạn.", "paragraph", ("relative_clause",)),
        ("Small steps still count, even when they feel slow.", "Những bước nhỏ vẫn có ý nghĩa, ngay cả khi chúng có vẻ chậm chạp.", "paragraph", ("discourse_connector",)),
    ], register="accessible"),
    "TECHNICAL": _items("TECHNICAL", [
        ("Call the HTTP API with curl, then parse the JSON response in Python.", "Gọi API HTTP bằng curl, sau đó phân tích phản hồi JSON trong Python.", "paragraph", ("phrasal_verb", "sentence_merge")),
        ("A cache is a component that stores frequently requested data for faster access.", "Bộ nhớ đệm là thành phần lưu dữ liệu thường được yêu cầu để truy cập nhanh hơn.", "paragraph", ("technical_definition", "relative_clause")),
        ("The service must validate the token before it handles the request.", "Dịch vụ phải xác thực token trước khi xử lý yêu cầu.", "paragraph", ("technical_definition", "modality")),
        ("If the connection drops, the client will retry with exponential backoff.", "Nếu kết nối bị ngắt, máy khách sẽ thử lại với thời gian chờ tăng dần theo cấp số nhân.", "paragraph", ("conditional", "technical_definition")),
        ("The response body is compressed before it is sent over the network.", "Phần thân phản hồi được nén trước khi gửi qua mạng.", "paragraph", ("passive", "technical_definition")),
        ("The function returns an empty list when no records match the filter.", "Hàm trả về danh sách rỗng khi không có bản ghi nào khớp với bộ lọc.", "paragraph", ("conditional",)),
        ("The schema defines how clients should represent an error.", "Schema định nghĩa cách máy khách nên biểu diễn lỗi.", "paragraph", ("technical_definition", "modality")),
        ("Do not expose the secret key in logs or error messages.", "Không được để lộ khóa bí mật trong nhật ký hoặc thông báo lỗi.", "paragraph", ("modality", "negation")),
        ("The worker reads the queue and hands each job to a separate handler.", "Worker đọc hàng đợi rồi chuyển từng tác vụ cho một bộ xử lý riêng.", "paragraph", ("sentence_merge",)),
        ("The migration failed because the column already existed.", "Migration thất bại vì cột này đã tồn tại.", "paragraph", ("causality",)),
        ("The module, which is loaded lazily, avoids a circular import.", "Module được nạp lười nên tránh được việc import vòng.", "paragraph", ("relative_clause", "technical_definition")),
        ("The command-line flag controls whether the cache is refreshed.", "Cờ dòng lệnh điều khiển việc có làm mới bộ nhớ đệm hay không.", "paragraph", ("technical_definition",)),
        ("The test suite caught a regression after the dependency was upgraded.", "Bộ kiểm thử phát hiện lỗi hồi quy sau khi dependency được nâng cấp.", "paragraph", ("causality", "passive")),
        ("The timeout is set to 30 seconds by default.", "Theo mặc định, thời gian chờ được đặt là 30 giây.", "paragraph", ("number", "passive")),
        ("Use a stable identifier rather than the display label.", "Hãy dùng mã định danh ổn định thay vì nhãn hiển thị.", "paragraph", ("technical_definition",)),
        ("The patch fixes the race condition without changing the public API.", "Bản vá sửa lỗi tranh chấp mà không thay đổi API công khai.", "paragraph", ("negation", "technical_definition")),
        ("The parser splits the input at sentence boundaries when the limit is exceeded.", "Bộ phân tích tách đầu vào tại ranh giới câu khi vượt quá giới hạn.", "paragraph", ("sentence_split", "technical_definition")),
        ("The implementation is small, but its behavior is covered by integration tests.", "Phần triển khai nhỏ, nhưng hành vi của nó được bao phủ bởi các kiểm thử tích hợp.", "paragraph", ("passive", "sentence_merge")),
        ("The endpoint accepts either a project ID or a workspace ID.", "Endpoint chấp nhận ID dự án hoặc ID workspace.", "paragraph", ("technical_definition",)),
        ("The client should fall back to the local file when the network is unavailable.", "Máy khách nên chuyển sang tệp cục bộ khi mạng không khả dụng.", "paragraph", ("phrasal_verb", "conditional")),
    ], register="precise"),
    "ACADEMIC": _items("ACADEMIC", [
        ("The optimization of the structure resulted in a measurable improvement.", "Việc tối ưu cấu trúc mang lại mức cải thiện có thể đo lường được.", "paragraph", ("nominalization",)),
        ("The findings suggest that the intervention may reduce variability.", "Các phát hiện cho thấy biện pháp can thiệp có thể làm giảm độ biến thiên.", "paragraph", ("academic_hedging", "modality")),
        ("Although the sample was small, the pattern was consistent across groups.", "Mặc dù mẫu nghiên cứu nhỏ, mô hình này vẫn nhất quán giữa các nhóm.", "paragraph", ("discourse_connector", "passive")),
        ("The model, which was trained on public data, performed well on the test set.", "Mô hình được huấn luyện trên dữ liệu công khai đạt kết quả tốt trên tập kiểm thử.", "paragraph", ("relative_clause", "passive")),
        ("If this assumption holds, the estimate should remain unbiased.", "Nếu giả định này đúng, ước lượng sẽ không bị chệch.", "paragraph", ("conditional", "modality")),
        ("The relationship is not necessarily causal.", "Mối quan hệ này không nhất thiết mang tính nhân quả.", "paragraph", ("academic_hedging", "negation")),
        ("The result can be explained by differences in baseline exposure.", "Kết quả có thể được giải thích bằng sự khác biệt về mức độ phơi nhiễm ban đầu.", "paragraph", ("academic_hedging", "passive")),
        ("The analysis controls for age, income, and education.", "Phân tích kiểm soát các yếu tố tuổi, thu nhập và học vấn.", "paragraph", ("nominalization",)),
        ("However, the evidence remains insufficient to support a broad conclusion.", "Tuy nhiên, bằng chứng vẫn chưa đủ để ủng hộ một kết luận khái quát.", "paragraph", ("discourse_connector", "academic_hedging")),
        ("The two measures capture related but distinct dimensions.", "Hai thước đo phản ánh những khía cạnh có liên quan nhưng khác biệt.", "paragraph", ("sentence_merge",)),
        ("The procedure was repeated three times for each condition.", "Quy trình được lặp lại ba lần cho mỗi điều kiện.", "paragraph", ("passive", "number")),
        ("The authors do not claim that the method generalizes to every setting.", "Các tác giả không khẳng định rằng phương pháp này khái quát cho mọi bối cảnh.", "paragraph", ("negation", "academic_hedging")),
        ("The observed effect was small, yet statistically reliable.", "Hiệu ứng quan sát được nhỏ nhưng vẫn có ý nghĩa thống kê đáng tin cậy.", "paragraph", ("sentence_merge",)),
        ("This distinction matters because the variables are measured differently.", "Sự phân biệt này quan trọng vì các biến được đo lường theo những cách khác nhau.", "paragraph", ("causality", "passive")),
        ("The framework provides a basis for comparing the three approaches.", "Khung này tạo cơ sở để so sánh ba cách tiếp cận.", "paragraph", ("nominalization", "number")),
        ("The results are broadly consistent with earlier work.", "Kết quả nhìn chung phù hợp với các nghiên cứu trước đây.", "paragraph", ("academic_hedging",)),
        ("The interpretation should be treated with caution.", "Cần thận trọng khi diễn giải kết quả này.", "paragraph", ("modality", "passive")),
        ("The study addresses a gap in the existing literature.", "Nghiên cứu này giải quyết một khoảng trống trong các tài liệu hiện có.", "paragraph", ("nominalization",)),
        ("The evidence points to a gradual rather than immediate change.", "Bằng chứng cho thấy sự thay đổi diễn ra dần dần chứ không ngay lập tức.", "paragraph", ("academic_hedging", "comparison")),
        ("The appendix reports additional robustness checks.", "Phụ lục trình bày thêm các kiểm tra độ vững.", "heading", ("technical_definition",)),
    ], register="scholarly"),
    "LEGAL": _items("LEGAL", [
        ("The licensee may terminate this Agreement only if written notice is provided 30 days in advance.", "Bên được cấp phép chỉ có thể chấm dứt Thỏa thuận này nếu gửi thông báo bằng văn bản trước 30 ngày.", "paragraph", ("legal_obligation", "conditional", "number")),
        ("The supplier shall deliver the goods within ten business days.", "Nhà cung cấp phải giao hàng trong vòng mười ngày làm việc.", "paragraph", ("legal_obligation", "number")),
        ("The customer must not disclose confidential information.", "Khách hàng không được tiết lộ thông tin mật.", "paragraph", ("legal_obligation", "negation")),
        ("This clause applies to all services provided under the Agreement.", "Điều khoản này áp dụng cho mọi dịch vụ được cung cấp theo Thỏa thuận.", "paragraph", ("legal_obligation", "passive")),
        ("If either party breaches this section, the other party may seek remedies.", "Nếu một trong hai bên vi phạm mục này, bên còn lại có thể yêu cầu biện pháp khắc phục.", "paragraph", ("conditional", "modality", "legal_obligation")),
        ("The term includes its successors and permitted assigns.", "Thuật ngữ này bao gồm các bên kế nhiệm và bên nhận chuyển nhượng được phép.", "paragraph", ("legal_obligation",)),
        ("No amendment is effective unless it is signed by both parties.", "Không sửa đổi nào có hiệu lực trừ khi được cả hai bên ký.", "paragraph", ("legal_obligation", "negation", "passive")),
        ("The parties agree to keep the dispute confidential.", "Các bên đồng ý giữ bí mật tranh chấp.", "paragraph", ("legal_obligation",)),
        ("The notice must identify the affected order and the requested relief.", "Thông báo phải xác định đơn hàng bị ảnh hưởng và biện pháp khắc phục được yêu cầu.", "paragraph", ("legal_obligation", "long_noun_phrase")),
        ("The provision survives termination of this Agreement.", "Điều khoản này vẫn có hiệu lực sau khi Thỏa thuận chấm dứt.", "paragraph", ("legal_obligation",)),
        ("However, nothing in this section limits the parties' statutory rights.", "Tuy nhiên, không nội dung nào trong mục này hạn chế các quyền theo luật định của các bên.", "paragraph", ("discourse_connector", "legal_obligation")),
        ("The claim shall be brought within one year after the relevant event.", "Khiếu nại phải được đưa ra trong vòng một năm kể từ sự kiện liên quan.", "paragraph", ("legal_obligation", "number")),
        ("The recipient may use the material solely for the stated purpose.", "Bên nhận chỉ có thể sử dụng tài liệu cho mục đích đã nêu.", "paragraph", ("legal_obligation", "modality")),
        ("The agreement does not create a partnership, agency, or employment relationship.", "Thỏa thuận này không tạo lập quan hệ hợp danh, đại lý hoặc lao động.", "paragraph", ("legal_obligation", "negation")),
        ("The parties shall cooperate in good faith to resolve the issue.", "Các bên phải hợp tác thiện chí để giải quyết vấn đề.", "paragraph", ("legal_obligation", "causality")),
        ("The foregoing obligations remain binding after delivery.", "Các nghĩa vụ nêu trên vẫn có tính ràng buộc sau khi giao hàng.", "paragraph", ("legal_obligation",)),
        ("Where required by law, the provider shall retain the records.", "Khi pháp luật yêu cầu, nhà cung cấp phải lưu giữ hồ sơ.", "paragraph", ("legal_obligation", "conditional")),
        ("The defined term has the meaning set out in Section 2.", "Thuật ngữ được định nghĩa có nghĩa như quy định tại Mục 2.", "paragraph", ("legal_obligation", "number")),
        ("The court may order the losing party to pay reasonable costs.", "Tòa án có thể yêu cầu bên thua kiện thanh toán các chi phí hợp lý.", "paragraph", ("legal_obligation", "modality")),
        ("Except as expressly stated, the warranty is disclaimed.", "Trừ khi được nêu rõ, bảo hành bị từ chối.", "paragraph", ("legal_obligation", "passive")),
    ], register="formal"),
    "LITERATURE": _items("LITERATURE", [
        ('"You came back," she said, barely above a whisper.', '“Anh đã về,” cô khẽ nói, giọng chỉ như thì thầm.', "dialogue", ("pronoun_chain",)),
        ("Rain tapped against the window long after the house fell silent.", "Mưa gõ lách tách lên khung cửa sổ rất lâu sau khi căn nhà chìm vào im lặng.", "paragraph", ("personification",)),
        ("He carried the old photograph as if it were a fragile promise.", "Anh mang theo tấm ảnh cũ như thể đó là một lời hứa mong manh.", "paragraph", ("idiom", "long_noun_phrase")),
        ("The street, which had seemed endless at noon, was empty now.", "Con phố vốn tưởng như bất tận vào buổi trưa giờ đã vắng tanh.", "paragraph", ("relative_clause",)),
        ("If she opened the door, the past would enter with her.", "Nếu mở cánh cửa, cô sẽ để quá khứ bước vào cùng mình.", "paragraph", ("conditional", "personification")),
        ("He might have forgotten the name, but not the voice.", "Có thể anh đã quên cái tên, nhưng không quên giọng nói ấy.", "paragraph", ("modality", "negation")),
        ("She looked away, and the silence answered for her.", "Cô ngoảnh mặt đi, còn sự im lặng đã trả lời thay cô.", "paragraph", ("sentence_merge", "personification")),
        ("However, the memory returned whenever the train slowed.", "Tuy nhiên, ký ức ấy lại trở về mỗi khi đoàn tàu chạy chậm lại.", "paragraph", ("discourse_connector",)),
        ("He finally gave in to the laughter rising around him.", "Cuối cùng anh cũng chịu bật cười theo tiếng cười đang dâng lên quanh mình.", "paragraph", ("phrasal_verb",)),
        ("The child held her hand as though the night could take it away.", "Đứa trẻ nắm tay cô như thể màn đêm có thể cuốn nó đi.", "paragraph", ("conditional", "pronoun_chain")),
        ("The room smelled of dust, oranges, and forgotten summers.", "Căn phòng có mùi bụi, cam và những mùa hè bị lãng quên.", "paragraph", ("sensory_detail",)),
        ("He did not answer; he only folded the letter twice.", "Anh không trả lời, chỉ gấp lá thư làm đôi hai lần.", "paragraph", ("negation", "sentence_merge")),
        ("She knew the road by heart, though she had never walked it before.", "Cô thuộc lòng con đường ấy dù chưa từng đi qua.", "paragraph", ("idiom", "discourse_connector")),
        ("At dawn, the village began to gather its voices.", "Lúc bình minh, ngôi làng bắt đầu gom nhặt những âm thanh của mình.", "paragraph", ("personification",)),
        ("The letter was written in a hurried hand.", "Lá thư được viết bằng nét chữ vội vã.", "paragraph", ("passive",)),
        ("\"Wait for me.\"\n\"I have been waiting all along.\"", "“Đợi tôi nhé.”\n“Em vẫn luôn chờ.”", "dialogue", ("pronoun_chain", "sentence_split")),
        ("The moon slipped behind the clouds, leaving the garden without a witness.", "Mặt trăng khuất sau những đám mây, để khu vườn không còn ai chứng kiến.", "paragraph", ("personification",)),
        ("She set the key down and walked away without looking back.", "Cô đặt chìa khóa xuống rồi bước đi, không ngoái lại.", "paragraph", ("phrasal_verb", "negation")),
        ("The house was smaller than he remembered, but warmer.", "Ngôi nhà nhỏ hơn anh nhớ, nhưng ấm áp hơn.", "paragraph", ("comparison", "sentence_merge")),
        ("For a moment, neither of them knew what to say.", "Trong một khoảnh khắc, cả hai đều không biết phải nói gì.", "paragraph", ("pronoun_chain", "negation")),
    ], register="voice-driven"),
}


def curated_examples(domain: str) -> List[FewShotExample]:
    """Trả về bản sao danh sách để caller không làm biến đổi thư viện tĩnh."""
    normalized = (domain or "GENERAL").upper()
    return list(CURATED_FEW_SHOTS.get(normalized, CURATED_FEW_SHOTS["GENERAL"]))


def curated_examples_as_dict(domain: str) -> List[Dict[str, object]]:
    return [example.to_dict() for example in curated_examples(domain)]


def library_counts() -> Dict[str, int]:
    return {domain: len(CURATED_FEW_SHOTS.get(domain, [])) for domain in SUPPORTED_DOMAINS}

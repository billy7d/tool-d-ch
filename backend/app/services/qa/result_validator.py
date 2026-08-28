from typing import Any, Dict


def qa_error(message: str) -> Dict[str, Any]:
    return {
        "status": "ERROR",
        "is_passed": False,
        "score": None,
        "issues": [],
        "suggested_revision": "",
        "error": message,
    }


def validate_qa_result(value: Any) -> Dict[str, Any]:
    """Chuẩn hóa contract QA và đóng lỗi khi schema không hợp lệ."""
    if not isinstance(value, dict):
        return qa_error("QA response không phải JSON object")
    if value.get("status") == "ERROR" or value.get("error"):
        # Không ghi đè chi tiết lỗi provider bằng lỗi schema chung chung.
        return qa_error(str(value.get("error") or "QA provider trả về lỗi."))
    if not isinstance(value.get("is_passed"), bool):
        return qa_error("QA response thiếu is_passed hợp lệ")
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
        return qa_error("QA response có score ngoài khoảng 0..1")
    issues = value.get("issues")
    if not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
        return qa_error("QA response có issues không hợp lệ")
    passed = value["is_passed"]
    return {
        "status": "PASS" if passed else "FAIL",
        "is_passed": passed,
        "score": float(score),
        "issues": issues,
        "suggested_revision": value.get("suggested_revision", "") if isinstance(value.get("suggested_revision", ""), str) else "",
        "error": None,
    }

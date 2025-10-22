import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Sequence, Iterator


OPERATORS = {"+", "-", "*", "/"}


def _count_operators(equation: str, operators: Iterable[str] = OPERATORS) -> int:
    """
    Count arithmetic operators in an equation string by token. Assumes SVAMP formatting
    where tokens are space-separated and parentheses are spaced, e.g.,
    "( ( 36.0 - 12.0 ) - 8.0 )".
    """
    if not equation:
        return 0
    tokens = equation.replace("(", " ").replace(")", " ").split()
    return sum(1 for token in tokens if token in operators)


def _load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"SVAMP json not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected SVAMP.json to contain a list of items.")
    return data


def filter_items_by_operator_count(
    items: List[Dict],
    *,
    min_ops: Optional[int] = None,
    max_ops: Optional[int] = None,
    exact_ops: Optional[int] = None,
    operators: Iterable[str] = OPERATORS,
) -> List[Dict]:
    """
    Filter SVAMP items by number of arithmetic operators in the 'Equation' field.

    - exact_ops: keep items with exactly this many operators
    - otherwise apply min_ops/max_ops bounds if provided
    """
    if exact_ops is not None:
        min_ops = exact_ops
        max_ops = exact_ops

    result: List[Dict] = []
    for item in items:
        eq = item.get("Equation", "")
        num_ops = _count_operators(eq, operators)
        if min_ops is not None and num_ops < min_ops:
            continue
        if max_ops is not None and num_ops > max_ops:
            continue
        result.append(item)
    return result


# Deprecated: multi-operation loader removed per project scope (two-ops only)


def load_svamp_two_operation(
    json_path: str = "SVAMP.json",
    *,
    operators: Iterable[str] = OPERATORS,
    save_to: Optional[str] = None,
) -> List[Dict]:
    """
    Load SVAMP and return items whose 'Equation' has exactly two operators.
    """
    items = _load_json(json_path)
    filtered = filter_items_by_operator_count(
        items, exact_ops=2, operators=operators
    )
    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
    return filtered


__all__ = [
    "load_svamp_two_operation",
    "filter_items_by_operator_count",
    "SVAMPQuestion",
    "SVAMPDataset",
]



@dataclass
class SVAMPQuestion:
    """
    Basit soru-cevap şeması:
    - question: Body + " " + Question (concat)
    - answer: Answer alanı (stringe çevrilmiş)
    - meta: opsiyonel ek bilgileri taşıyacak sözlük
    """
    question: str
    answer: str
    meta: Dict[str, object]


class SVAMPDataset:
    """
    SVAMP iki-operatörlü örneklerden oluşan dataset arabirimi.
    Varsayılan olarak tam 2 operatörlü örnekler yüklenir ve soru=Body+Question, cevap=Answer olur.
    """

    def __init__(
        self,
        json_path: str = "SVAMP.json",
        *,
        exact_ops: int = 2,
    ) -> None:
        raw_items = _load_json(json_path)
        items = filter_items_by_operator_count(raw_items, exact_ops=exact_ops)
        self._examples: List[SVAMPQuestion] = []
        for it in items:
            body = it.get("Body", "")
            q = it.get("Question", "")
            question_text = (body.strip() + " " + q.strip()).strip()
            ans = it.get("Answer")
            answer_text = str(ans) if ans is not None else ""
            meta = {
                "ID": it.get("ID"),
                "Equation": it.get("Equation"),
                "Type": it.get("Type"),
            }
            self._examples.append(SVAMPQuestion(question=question_text, answer=answer_text, meta=meta))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> SVAMPQuestion:
        return self._examples[idx]

    def __iter__(self) -> Iterator[SVAMPQuestion]:
        return iter(self._examples)

    def to_list(self) -> List[SVAMPQuestion]:
        return list(self._examples)


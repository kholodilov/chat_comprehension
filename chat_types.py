from typing import List

class MaybeReplies:
    def __init__(self, start_id: int, end_id: int):
        self.start_id = start_id
        self.end_id = end_id


class Message:
    def __init__(self, id: int, text: str, reply_to: int, replies: List[int] = None, maybe_replies: MaybeReplies = None):
        self.id = id
        self.text = text
        self.reply_to = reply_to
        self.replies = replies or []
        self.maybe_replies = maybe_replies


class MessageHighlight:
    def __init__(self, id: int, start: int, len: int, score: float = None):
        self.id = id
        self.start = start
        self.len = len
        self.score = score


class QA:
    def __init__(self, question_id: int, answers: List[MessageHighlight], maybe_answers: List[MessageHighlight]):
        self.question_id = question_id
        self.answers = answers
        self.maybe_answers = maybe_answers


class ModelResults:
    def __init__(self, question_ids, answer_candidate_ids, highlights, start_pos, scores):
        self.question_ids = question_ids
        self.answer_candidate_ids = answer_candidate_ids
        self.highlights = highlights
        self.start_pos = start_pos
        self.scores = scores

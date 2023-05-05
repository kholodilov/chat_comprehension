import itertools, operator
from typing import List, Dict
from deeppavlov.core.common.chainer import Chainer

from chat_types import *


def run_model_on_replies(messages: Dict[int, Message], model: Chainer) -> ModelResults:
    question_ids: List[int] = []
    questions_text: List[str] = []
    answer_candidate_ids: List[int] = []
    answer_candidates_text: List[str] = []
    for message in messages.values():
        num_answers = len(message.replies)
        if '?' in message.text and num_answers > 0:
            question_ids.extend(itertools.repeat(message.id, num_answers))
            questions_text.extend(itertools.repeat(message.text, num_answers))
            answer_candidate_ids.extend(message.replies)
            answer_candidates_text.extend([messages[id].text for id in message.replies])

    (highlights, start_pos, scores) = model(answer_candidates_text, questions_text)
    return ModelResults(question_ids, answer_candidate_ids, highlights, start_pos, scores)


def run_model_on_maybe_replies(messages: Dict[int, Message], model: Chainer) -> ModelResults:
    question_ids: List[int] = []
    questions_text: List[str] = []
    answer_candidate_ids: List[int] = []
    answer_candidates_text: List[str] = []
    for message in messages.values():
        if '?' in message.text:
            maybe_answer_ids: List[int] = []
            for maybe_answer_id in range(message.maybe_replies.start_id, message.maybe_replies.end_id+1):
                if maybe_answer_id in messages and maybe_answer_id not in message.replies:
                    maybe_answer_ids.append(maybe_answer_id)
            num_answers = len(maybe_answer_ids)
            if num_answers > 0:
                question_ids.extend(itertools.repeat(message.id, num_answers))
                questions_text.extend(itertools.repeat(message.text, num_answers))
                answer_candidate_ids.extend(maybe_answer_ids)
                answer_candidates_text.extend([messages[id].text for id in maybe_answer_ids])

    (highlights, start_pos, scores) = model(answer_candidates_text, questions_text)
    return ModelResults(question_ids, answer_candidate_ids, highlights, start_pos, scores)


def prepare_qa_from_model_results(model_results: ModelResults, model_results_maybe_replies: ModelResults) -> List[QA]:
    questions_and_answers: Dict[int, QA] = {}
    results = list(zip(
        model_results.question_ids, model_results.answer_candidate_ids, model_results.highlights,
        model_results.start_pos, model_results.scores))
    results_maybe_replies = list(zip(
        model_results_maybe_replies.question_ids, model_results_maybe_replies.answer_candidate_ids,
        model_results_maybe_replies.highlights, model_results_maybe_replies.start_pos, model_results_maybe_replies.scores))
    for question_id, raw_answers in itertools.groupby(results, operator.itemgetter(0)):
        answers = [
            MessageHighlight(id, start, len(highlight), score)
            for _, id, highlight, start, score in raw_answers
            if start >= 0
        ]
        if len(answers) > 0:
            questions_and_answers[question_id] = QA(question_id, answers, [])

    for question_id, raw_answers in itertools.groupby(results_maybe_replies, operator.itemgetter(0)):
        answers = [
            MessageHighlight(id, start, len(highlight), score)
            for _, id, highlight, start, score in raw_answers
            if start >= 0
        ]
        if len(answers) > 0:
            if question_id in questions_and_answers:
                qa = questions_and_answers[question_id]
                qa.maybe_answers = answers
            else:
                questions_and_answers[question_id] = QA(question_id, [], answers)

    return list(questions_and_answers.values())

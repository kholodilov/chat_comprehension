# Extract questions with answers from chat
#
# 1) Identify question in chat
# + all messages with question mark (which are not replies to other messages with question marks?)
# 2) Identify messages with potential answers
# + Replies to question message
# - and all messages in <X minutes/Y messages> window after question message
# 3) Extract/score potential answers, output the best ones
# + score text of all replies together
# - score each answer separately, select ones with highest threshold

import itertools, operator
from typing import List

import telethon
from telethon import TelegramClient, events, sync
from chat_types import *
import chat_utils

from deeppavlov import build_model, configs

api_id = -1
api_hash = ''
client = TelegramClient('test_session', api_id, api_hash)
client.start()

dialogs = client.get_dialogs()
chat = [c for c in dialogs if c.name == "Лондон чат для русскоговорящих"][0]

raw_messages = client.get_messages(chat, limit=150)
messages = {}
for msg in reversed(raw_messages):
    if type(msg) is telethon.tl.types.Message and msg.message is not None:
        reply_to = msg.reply_to.reply_to_msg_id if msg.reply_to is not None else None
        messages[msg.id] = Message(msg.id, msg.message, reply_to)
        if reply_to in messages:
            messages[reply_to].replies.append(msg.id)

message_ids = list(messages.keys())
WINDOW_SIZE = 10
window_start = len(message_ids)-1
window_end = window_start
for id in reversed(message_ids[:-1]):
    messages[id].maybe_replies = MaybeReplies(message_ids[window_start], message_ids[window_end])
    window_start -= 1
    if window_end - window_start >= WINDOW_SIZE:
        window_end -= 1

model = build_model(configs.squad.squad_ru_rubert_infer, download=True)
model_noans = build_model(configs.squad.multi_squad_ru_retr_noans_rubert_infer, download=True)

model_results = chat_utils.run_model_on_replies(messages, model)
model_results_maybe_replies = chat_utils.run_model_on_maybe_replies(messages, model_noans)
questions_and_answers = chat_utils.prepare_qa_from_model_results(model_results, model_results_maybe_replies)

for qa in questions_and_answers:
    print("Question: " + messages[qa.question_id].text + "\n")
    for answer in qa.answers:
        text = messages[answer.id].text
        start = answer.start
        end = answer.start+answer.len
        print("Answer (" + str(answer.score) + "): " + "*".join([text[:start], text[start:end], text[end:]]) + "\n")
    for answer in qa.maybe_answers:
        text = messages[answer.id].text
        start = answer.start
        end = answer.start+answer.len
        print("Maybe answer (" + str(answer.score) + "): " + "*".join([text[:start], text[start:end], text[end:]]) + "\n")
    print("\n")

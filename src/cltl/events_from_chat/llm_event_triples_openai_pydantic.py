import copy
import json
from datetime import datetime
import os
import re
from openai import OpenAI
from typing import Optional
from pydantic import BaseModel
import prompts as prompts
import data_type as data_type

#https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat

#predicates = ["sem:hasActor", "sem:hasTimne", "sem:hasPlace"]

def _load_key() -> str:
    env = os.environ.get("OPENAI_API_KEY")
    if not env:
        raise SystemExit("OPENAI_API_KEY environment variable not set.")
    return env


# Loaded when the first extractor without its own api_key is made, not at import, so a local
# server (see LLM_EventExtraction's base_url) works without OPENAI_API_KEY.
key = None

ACTIVITY_ID_PATTERN = re.compile(r"^chat(\d+)\.(\d+)$")


def locate_phrase(utterance, phrase):
    """Find `phrase` as a substring of `utterance`: exact match first, then
    case-insensitive. Returns (offset, length), or (None, len(phrase)) if `phrase`
    does not occur anywhere in `utterance`. Mirrors locatePhrase() in
    src/cltl/annotation/annotation_tool.html, so LLM output and manual annotation
    fall back to the same offset-recovery strategy."""
    if not utterance or not phrase:
        return None, len(phrase or "")
    idx = utterance.find(phrase)
    if idx == -1:
        idx = utterance.lower().find(phrase.lower())
    return (idx, len(phrase)) if idx != -1 else (None, len(phrase))


class Perspective(BaseModel):
    model_config = {"json_schema_mode": "validation"}

    emotion: data_type.EmotionLabel
    factuality: data_type.Factuality
    certainty: data_type.Certainty


class Activity(BaseModel):
    model_config = {"json_schema_mode": "validation"}

    activity_id: str
    value: Optional[str] = None
    offset: Optional[int] = None
    length: Optional[int] = None
    type: Optional[data_type.ActivityType] = None


class RoleSpan(BaseModel):
    model_config = {"json_schema_mode": "validation"}

    value: str
    type: data_type.RoleType
    offset: int
    length: int


class ResultSpan(BaseModel):
    model_config = {"json_schema_mode": "validation"}

    value: str
    type: data_type.ResultType
    offset: int
    length: int


class TimeSpan(BaseModel):
    model_config = {"json_schema_mode": "validation"}

    value: str
    offset: int
    length: int


class TimeResolved(BaseModel):
    model_config = {"json_schema_mode": "validation"}

    time_expression: str
    temporal_type: data_type.TemporalType
    absolute_date: Optional[str] = None
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    recurrence_pattern: Optional[data_type.RecurrencePattern] = None


class SRLAnnotation(BaseModel):
    model_config = {"json_schema_mode": "validation"}

    perspective: Perspective
    activity: Activity
    agent: Optional[list[RoleSpan]] = []
    patient: Optional[list[RoleSpan]] = []
    agent_patient: Optional[list[RoleSpan]] = []
    experiencer: Optional[list[RoleSpan]] = []
    participant: Optional[list[RoleSpan]] = []
    qualification: Optional[list[RoleSpan]] = []
    instrument: Optional[list[RoleSpan]] = []
    location: Optional[list[RoleSpan]] = []
    result: Optional[list[ResultSpan]] = []
    time: Optional[list[TimeSpan]] = []
    time_resolved: Optional[list[TimeResolved]] = []


class SRLAnnotations(BaseModel):
    model_config = {"json_schema_mode": "validation"}

    extractions: list[SRLAnnotation]

    @classmethod
    def to_openai_function(cls):
        """Convert the Pydantic model to OpenAI function definition for tool use."""
        return {
            "type": "function",
            "function": {
                "name": "extract_events",
                "description": "Extract offset-anchored SRL annotations with activity identifiers and speaker perspective from text, matching the annotations.json format.",
                "parameters": cls.model_json_schema()
            }
        }


def _weekday(date) -> Optional[str]:
    """The weekday of a conversation date as the sessions write it ("2026,Nov,05", or ISO
    "2026-11-05"), or None when it is not such a date."""
    for fmt in ("%Y,%b,%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(date), fmt).strftime("%A")
        except ValueError:
            continue
    return None


def _anchored_activity_schema() -> dict:
    """SRLAnnotations' JSON schema with every activity's value, offset, length and type required
    and non-null -- for LLM_EventExtraction(tool_calls=False), whose output a local server
    (llama-server) constrains to the schema with a grammar. With the plain schema, where those
    four are optional (a bare reference to an earlier activity has only its activity_id), a
    grammar-constrained model took the shortest way out and left them all null: every extraction
    came back without an activity. The cost: a bare reference cannot be expressed this way, so
    the model anchors it in the current utterance instead."""
    schema = copy.deepcopy(SRLAnnotations.model_json_schema())
    activity = schema["$defs"]["Activity"]
    for key in ("value", "offset", "length", "type"):
        prop = activity["properties"][key]
        non_null = [branch for branch in prop.get("anyOf", []) if branch.get("type") != "null"]
        if non_null:
            rest = {k: v for k, v in prop.items() if k not in ("anyOf", "default")}
            activity["properties"][key] = {**non_null[0], **rest}
    activity["required"] = sorted(set(activity.get("required", [])) | {"value", "offset", "length", "type"})
    return schema


# Added to the extraction instructions for a smaller (local) model, with extra_rules=True: the
# mistakes Qwen3-30B made in a real session (see tests/extraction_cases.json) -- negations stored
# as confirmed, "I think" as certain, the object of an activity left out, and a new activity
# folded into the previous one under its activity_id.
EXTRA_RULES = """

Extra rules -- check every extraction against them:
1. Negation: when the speaker says something did NOT happen ("I didn't take", "I skipped", "I
   forgot", "I never", "no ...", "not ..."), the factuality is "deny", not "confirm".
2. Hedging: "I think", "maybe", "probably", "I guess", "not sure" make the certainty "uncertain".
3. Plans and expectations ("I will", "I'm going to", "tomorrow I") have factuality "expect".
4. Always extract what an activity is about as its patient: the medicine ("insulin"), the food
   or drink ("tea", "pasta"), the measurement ("blood sugar"). Do not leave it out.
5. A new activity gets a NEW activity_id (the next free number in this chat). Reuse an earlier
   activity_id ONLY when the speaker clearly talks about that same activity again, and then with
   exactly the same type. Eating, drinking, taking medicine and a condition such as hunger or
   tiredness are always different activities.
6. Resolve relative dates ("yesterday", "on Monday", "last week") from the conversation date and
   its weekday given with each input; "on Monday" is the most recent Monday on or before it."""

DEFAULT_TIMEOUT = 60.0  # seconds -- see the `timeout` parameter docstring below
DEFAULT_MODEL = "gpt-5.1"


class LLM_EventExtraction:

    def __init__(self, timeout: float = DEFAULT_TIMEOUT, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, model: str = DEFAULT_MODEL,
                 tool_calls: bool = True, history_turns: Optional[int] = None,
                 temperature: Optional[float] = None, extra_rules: bool = False):
        """
        :param temperature: sampling temperature of the extraction calls; None leaves the
            server's default. 0 makes a local server (llama-server samples at ~0.8 by default)
            give the same extraction for the same input.
        :param extra_rules: append EXTRA_RULES to the extraction instructions, for a model that
            needs them spelled out (a local one).
        :param api_key, base_url: another OpenAI-compatible server, such as llama.cpp's
            llama-server (base_url "http://localhost:8080/v1"), and its key. Default:
            OPENAI_API_KEY and OpenAI itself.
        :param model: the chat model.
        :param tool_calls: False asks for the extractions as JSON following SRLAnnotations'
            schema (response_format json_schema) instead of as a forced tool call, for a server
            without tool calling (llama-server without --jinja).
        :param history_turns: keep only this many earlier user turns (with the replies to them)
            in the history sent along, for a model with a small context window. None keeps all.
        :param timeout: per-request timeout (seconds) for this instance's OpenAI client,
            passed straight through to openai.OpenAI(timeout=...). Without this, a stuck or
            very slow request has no ceiling and can block process_input() indefinitely (this
            is what happened before it was added -- a live chat turn once hung for 5+ minutes
            with no error). max_retries=0 too, so a timeout surfaces immediately as
            openai.APITimeoutError instead of the SDK silently retrying (by default, up to 2
            more times) before finally raising -- chat_sessions.KgChatSession's
            _call_openai() catches that and turns it into a message the human sees.
        """
        global key
        if api_key is None and key is None:
            key = _load_key()
        self._client = OpenAI(api_key=api_key or key, base_url=base_url, timeout=timeout, max_retries=0)
        self._model = model
        self._tool_calls = tool_calls
        self._history_turns = history_turns
        self._sampling = {} if temperature is None else {"temperature": temperature}
        self._history = []
        instructions = prompts.prompt_conversational_srl_annotation + (EXTRA_RULES if extra_rules else "")
        self._instruct = [{"role": "system", "content": instructions}]
        self._known_activities = {}

    def process_input(self, turn):
        # 2. Define the OpenAI function call parameters
        function_schema = SRLAnnotations.to_openai_function()

        self._history.append({"role": "user", "content": "Input: {}".format(turn)})
        self._trim_history()

        # 3. Prepare the messages
        messages = self._instruct+self._history

        if not self._tool_calls:
            return self._process_as_json(messages)

        # 4. Call the OpenAI API with the tool definition
        # You can set tool_choice to "required" if you insist the model calls the function,
        # or "auto" to let it decide.
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=[function_schema],
            tool_choice = "required",  # Force function call otherwise use :auto"
            **self._sampling,
        )

        # 5. Process the response
        response_message = response.choices[0].message

        tool_calls = response_message.tool_calls

        # Append the model's own reply to history too, not just our per-turn inputs. Without
        # this, every call sent a run of consecutive "user" messages with no assistant replies
        # in between, which gives the model no clear signal for which message is the CURRENT
        # turn -- it increasingly pulled content from earlier turns into the latest extraction
        # as the conversation grew (visible as HALLUCINATION: N tags in evaluate.py's mismatch
        # log, almost always pointing at an earlier turn).
        assistant_message = {"role": "assistant", "content": response_message.content}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
        self._history.append(assistant_message)

        if not tool_calls:
            # Fallback if the model didn't call the function
            return response_message.content

        # Every tool_call on an assistant message must be answered by a matching "tool"
        # message before the next API call, or the request is rejected -- we're not actually
        # executing a tool, just using function-calling for structured output, so the content
        # is a placeholder acknowledgment rather than a real result.
        for tool_call in tool_calls:
            self._history.append({"role": "tool", "tool_call_id": tool_call.id, "content": "Extraction recorded."})

        # The model invoked the function
        available_functions = {
            "extract_events": SRLAnnotations,
        }
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            # Parse the response into a Pydantic model
            try:
                structured_output = function_to_call(**function_args)
            except Exception as e:
                print("Error parsing function output:", e)
                structured_output = None
            if structured_output is not None:
                return structured_output.extractions
            else:
                return None

    def _trim_history(self):
        """Drop all but the last `history_turns` user turns (and what follows each of them)."""
        if self._history_turns is None:
            return
        user_turns = [i for i, m in enumerate(self._history) if m["role"] == "user"]
        if len(user_turns) > self._history_turns + 1:
            self._history = self._history[user_turns[-(self._history_turns + 1)]:]

    def _process_as_json(self, messages):
        """process_input() for a server without tool calling: the extractions come back as the
        message content, constrained to SRLAnnotations' JSON schema."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": {
                "name": "extract_events", "schema": _anchored_activity_schema(), "strict": True}},
            **self._sampling,
        )
        content = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": content})
        try:
            return SRLAnnotations(**json.loads(content)).extractions
        except Exception as e:
            print("Error parsing JSON output:", e)
            return None

    def _split_reused_ids(self, chat_number, extractions):
        """An activity_id from an earlier turn, reused with a DIFFERENT type, is a new activity
        the model folded into an earlier one (a local model did: "I drink tea" became part of
        "I didn't take insulin", "I eat three times a day" part of "I'm hungry"). Such an
        extraction gets the next free id of this chat instead, in place. Reuse with the same type
        is left alone: that is the same activity talked about again."""
        used = [int(m.group(2)) for m in (ACTIVITY_ID_PATTERN.match(i) for i in self._known_activities) if m]
        used += [int(m.group(2)) for m in (ACTIVITY_ID_PATTERN.match(e.activity.activity_id or "")
                                           for e in extractions) if m]
        next_number = max(used, default=0) + 1
        for extraction in extractions:
            activity = extraction.activity
            known_type = self._known_activities.get(activity.activity_id)
            if activity.value is None or activity.type is None or known_type is None or activity.type == known_type:
                continue
            new_id = f"chat{chat_number}.{next_number}"
            print(f"Activity '{activity.value}' ({activity.type.value}) reused {activity.activity_id} "
                  f"({known_type.value}); made it {new_id}")
            activity.activity_id = new_id
            next_number += 1

    def check_compliance(self, chat_number, utterance, extractions):
        """Validate a turn's extractions against the annotations.json Output-entry schema.

        An activity entry is either:
        - a first-time (or anchored-reference) mention: value/offset/length/type all present.
          A first-time mention mints a new activity_id; an anchored reference reuses an
          activity_id already seen in this conversation and MUST reuse that activity's type
          exactly (mirrors the annotation tool: reusing an id also reuses its type).
        - a bare (anchorless) reference: activity_id only, value/offset/length/type all None.

        Also checks that every value's offset/length matches the turn's utterance text. When
        they don't, this re-locates the value via locate_phrase() (exact then case-insensitive
        substring search, mirroring the annotation tool's own offset-recovery fallback) and
        corrects the offset/length IN PLACE on the extraction, since the LLM is unreliable at
        raw character counting even though it almost always gets the value text itself right.
        A span whose value doesn't occur anywhere in the utterance can't be auto-corrected and
        is reported as a genuine issue (typically because the model pulled it from a different
        turn's utterance).
        Returns a list of human-readable violation strings (empty if fully compliant); entries
        describing an auto-correction are included too, so corrections stay visible in logs even
        though the extraction itself has already been fixed by the time this returns."""
        issues = []
        for i, extraction in enumerate(extractions):
            prefix = f"extraction[{i}]"
            activity = extraction.activity

            match = ACTIVITY_ID_PATTERN.match(activity.activity_id)
            if not match:
                issues.append(f"{prefix}: activity_id '{activity.activity_id}' does not match 'chat<N>.<M>'")
            elif int(match.group(1)) != chat_number:
                issues.append(f"{prefix}: activity_id '{activity.activity_id}' does not belong to chat {chat_number}")

            spans = []
            if activity.value is not None:
                if activity.offset is None or activity.length is None or activity.type is None:
                    issues.append(f"{prefix}: activity '{activity.value}' is missing offset, length or type")
                else:
                    spans.append(("activity", activity))
                    known_type = self._known_activities.get(activity.activity_id)
                    if known_type is None:
                        self._known_activities[activity.activity_id] = activity.type
                    elif activity.type != known_type:
                        issues.append(f"{prefix}: activity_id '{activity.activity_id}' reused with type '{activity.type.value}' but was first introduced as '{known_type.value}'")
            else:
                if activity.activity_id not in self._known_activities:
                    issues.append(f"{prefix}: activity_id '{activity.activity_id}' references an activity not introduced earlier in this conversation")
                if activity.offset is not None or activity.length is not None or activity.type is not None:
                    issues.append(f"{prefix}: bare reference to '{activity.activity_id}' should not carry offset, length or type without a value")

            for role_name in ("agent", "patient", "agent_patient", "experiencer", "participant", "qualification", "instrument", "location", "result", "time"):
                for role in getattr(extraction, role_name):
                    spans.append((role_name, role))

            for role_name, span in spans:
                offset, length, value = span.offset, span.length, span.value
                valid = (
                    offset is not None and length is not None
                    and 0 <= offset and offset + length <= len(utterance)
                    and utterance[offset:offset + length] == value
                )
                if not valid:
                    new_offset, new_length = locate_phrase(utterance, value)
                    if new_offset is not None:
                        span.offset, span.length = new_offset, new_length
                        issues.append(f"{prefix}: {role_name} offset auto-corrected for '{value}': ({offset},{length}) -> ({new_offset},{new_length})")
                    else:
                        issues.append(f"{prefix}: {role_name} value '{value}' not found anywhere in the utterance — cannot auto-correct (was ({offset},{length}))")

            time_values = {t.value for t in extraction.time}
            for resolved in extraction.time_resolved:
                if resolved.time_expression not in time_values:
                    issues.append(f"{prefix}: time_resolved expression '{resolved.time_expression}' has no matching entry in time")

        return issues

    def annotate_all_turns_in_conversation(self, input={}):
        annotations = []
        self._history = []
        self._known_activities = {}
        print("Annotating a conversation with {} utterances".format(len(input['turns'])))
        for index, turn in enumerate(input['turns']):
            turn_with_context = {
                "chat": input['chat'],
                "human": input['human'],
                "date": input['date'],
                "turn": turn['turn'],
                "speaker": turn['speaker'],
                "utterance": turn['utterance'],
            }
            print('turn', turn_with_context)
            response = self.process_input(turn_with_context)
            if response:
                issues = self.check_compliance(input['chat'], turn['utterance'], response)
                if issues:
                    print(f"Compliance issues for chat {input['chat']} turn {turn['turn']}:")
                    for issue in issues:
                        print(" -", issue)
                annotation={"chat": input['chat'], "date": input["date"], "human": input["human"], "Input": turn, "Output": response}
                annotations.append(annotation)
        return annotations

    def reset_conversation(self):
        """Clear this extractor's running self._history/self._known_activities, e.g. before
        starting a new conversation on a reused LLM_EventExtraction instance. Mirrors what
        annotate_all_turns_in_conversation() does automatically at the start of every call;
        annotate_new_turn() does NOT reset on its own, since its whole point is to keep that
        state alive across calls."""
        self._history = []
        self._known_activities = {}

    def annotate_new_turn(self, chat, human, date, turn):
        """Incremental sibling of annotate_all_turns_in_conversation(): annotates exactly ONE
        new turn, using this extractor's own running self._history/self._known_activities as
        context instead of resetting and re-sending every prior turn on every call. Call this
        once per new turn, in order, for turns of the SAME conversation (same chat/human/date)
        -- call reset_conversation() first when starting a different conversation on a reused
        instance (a fresh instance already starts with empty state, so that's optional there).

        :param chat: chat identifier, same for every turn of this conversation.
        :param human: the human speaker's name, same for every turn of this conversation.
        :param date: date the conversation took place.
        :param turn: {"turn": <turn id>, "speaker": ..., "utterance": ...}
        :return: the same {"chat", "date", "human", "Input", "Output"} annotation entry
            annotate_all_turns_in_conversation() would produce for this turn, or None if the
            LLM returned no extractions for it.
        """
        turn_with_context = {
            "chat": chat,
            "human": human,
            "date": date,
            # For "on Monday": the model cannot work out the weekday of a date reliably itself.
            **({"weekday": _weekday(date)} if _weekday(date) else {}),
            "turn": turn['turn'],
            "speaker": turn['speaker'],
            "utterance": turn['utterance'],
        }
        print('turn', turn_with_context)
        response = self.process_input(turn_with_context)
        if not response:
            return None
        self._split_reused_ids(chat, response)
        issues = self.check_compliance(chat, turn['utterance'], response)
        if issues:
            print(f"Compliance issues for chat {chat} turn {turn['turn']}:")
            for issue in issues:
                print(" -", issue)
        return {"chat": chat, "date": date, "human": human, "Input": turn, "Output": response}

if __name__ == "__main__":
    llm_extractor  = LLM_EventExtraction()

    chats = [{
        "chat": 254,
        "human": "Mehmet",
        "date": "2013,Jan,31",
        "turns": [
            {
                "turn": 1,
                "speaker": "Mehmet",
                "utterance": "I've been reading about how different diets can impact my blood sugar levels. Recently, I've heard about the Mediterranean diet being beneficial. Do you think it's suitable for me?"
            },
            {
                "turn": 2,
                "speaker": "agent",
                "utterance": "The Mediterranean diet can indeed be beneficial for managing Type 2 Diabetes, Mehmet. Its emphasis on whole grains, healthy fats, fruits, vegetables, and lean proteins can help stabilize blood sugar levels. Have you tried integrating any of its elements into your diet already?"
            },
            {
                "turn": 3,
                "speaker": "Mehmet",
                "utterance": "Not entirely, but I've been incorporating more olive oil and fish into my meals. Is there anything specific I should avoid or be cautious about?"
            },
            {
                "turn": 4,
                "speaker": "agent",
                "utterance": "That's a great start. You might want to also focus on portion control to maintain a balanced calorie intake and avoid added sugars and refined grains. This diet’s high fiber content can particularly help in managing your blood sugar."
            },
            {
                "turn": 5,
                "speaker": "Mehmet",
                "utterance": "I see, but I enjoy having traditional Turkish meals, especially during family gatherings. Can these fit within the Mediterranean guidelines?"
            },
            {
                "turn": 6,
                "speaker": "agent",
                "utterance": "Absolutely. Many traditional Turkish dishes already align well with the Mediterranean diet. You can focus on dishes that include plenty of vegetables, legumes, and lean meats, and try to swap out less healthy ingredients for more wholesome options where possible."
            },
            {
                "turn": 7,
                "speaker": "Mehmet",
                "utterance": "That sounds manageable. I'll explore more recipes and perhaps even share them with my family for our gatherings. Thanks for the advice."
            },
            {
                "turn": 8,
                "speaker": "agent",
                "utterance": "You're welcome, Mehmet! It's wonderful that you’re involving your family in your health journey. This can make managing your condition more enjoyable and sustainable."
            }
        ]
    },
        {
            "chat": 32,
            "human": "Rudolf",
            "date": "2014,Apr,08",
            "turns": [
                {
                    "turn": 1,
                    "speaker": "agent",
                    "utterance": "Rudolf, let's talk about your meals today. What was your breakfast like?"
                },
                {
                    "turn": 2,
                    "speaker": "Rudolf",
                    "utterance": "For breakfast, I had a bowl of oatmeal along with an apple."
                },
                {
                    "turn": 3,
                    "speaker": "agent",
                    "utterance": "That's a healthy choice, Rudolf. How about lunch and dinner? Did you include lean proteins and vegetables?"
                },
                {
                    "turn": 4,
                    "speaker": "Rudolf",
                    "utterance": "Yes, for lunch, I had grilled chicken with salad, and for dinner, I cooked a fish with some steamed vegetables."
                },
                {
                    "turn": 5,
                    "speaker": "agent",
                    "utterance": "Sounds good. Including lean proteins and vegetables in your meals is beneficial for managing your blood sugar levels. However, don't forget to also include whole grains and fiber-rich foods for balanced meals."
                },
                {
                    "turn": 6,
                    "speaker": "Rudolf",
                    "utterance": "I will remember to do that. Also, I need advice on snacks. What would you recommend?"
                },
                {
                    "turn": 7,
                    "speaker": "agent",
                    "utterance": "Snacking on fruits and nuts or seeds is a good option, they can help prevent spikes in your blood sugar levels. Remember to consume in moderation, especially nuts and seeds, as they are high in calories. Drink plenty of water throughout the day as well."
                },
                {
                    "turn": 8,
                    "speaker": "agent",
                    "utterance": "Good morning, Rudolf. I noticed that you haven't logged your glucose levels in the app for a couple of days. Is everything alright?"
                },
                {
                    "turn": 9,
                    "speaker": "Rudolf",
                    "utterance": "I got caught up with work and forgot. Also, it's a bit of a hassle to do it every day."
                },
                {
                    "turn": 10,
                    "speaker": "agent",
                    "utterance": "It's understandable that it can feel tedious, Rudolf, but remember, regular monitoring is essential in managing your diabetes effectively."
                }
            ]
        }
    ]
    for chat in chats:
        annotations = llm_extractor.annotate_all_turns_in_conversation(chat)
        for annotation in annotations:
            print(annotation['Output'])

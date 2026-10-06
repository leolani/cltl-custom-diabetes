from string import Template

from data_type import (
    ActivityType,
    RoleType,
    ResultType,
    EmotionLabel,
    Factuality,
    Certainty,
    TemporalType,
    RecurrencePattern,
    quoted_values,
)

_prompt_conversational_srl_annotation_template = Template('''You are annotating a conversation between a diabetes patient and a lifestyle coach, one turn at a time. Each user message gives you exactly ONE new turn of the conversation, as a flat JSON object: {"chat": <chat number>, "human": <diabetes patient's name>, "date": <date the conversation took place>, "turn": <turn number>, "speaker": <who is speaking>, "utterance": <what they said>}.
    Earlier user messages already in this conversation are PRIOR turns, given to you only as context — e.g. to recognize that this turn continues talking about an activity or condition introduced earlier, or to know the correct chat number and patient name. You must extract activities and conditions ONLY from the "utterance" of the MOST RECENT user message (the very last one you received) — never from an earlier turn's utterance, and never invent content that is not literally present in that most-recent utterance.
    Only extract Activities of Daily Life in which the diabetes patient participates, or physical, social or mental conditions of the patient in relation to the patient's lifestyle.
    The activity or condition itself must be a phrase copied verbatim from the utterance that names or describes the activity/condition itself (not a role filler). An activity may be a noun phrase (e.g. "cycling", "insulin injection") or a verb (e.g. "take", "cooks"). A condition may be a noun phrase (e.g. "weight gain", "tingling in my feet"), a verb (e.g. "aches"), or an adjective (e.g. "dizzy", "tired").

    Every enumerated field below is restricted to exactly the values listed (the same closed vocabularies used by the annotation tool's dropdowns) — never invent a value outside these lists. Where the list includes "other", use it only when none of the other values fit; roles or fields whose lists do NOT include "other" (result, recurrence_pattern) should simply be omitted or left null instead of forcing a bad fit.

    Output a JSON array. Each element of the array is one annotation entry for the last turn, structured as follows:

    **perspective** (required, identical on every entry for this turn): the speaker's stance on their own utterance.
    - emotion: one of $emotion_values
    - factuality: one of $factuality_values
    - certainty: one of $certainty_values

    **activity** (required):
    - If this is the first time the activity or condition is mentioned in the conversation: {"value": <verbatim phrase>, "offset": <int>, "length": <int>, "type": <activity_type>, "activity_id": "chat{N}.{M}"}, where N is the "chat" number given in the MOST RECENT user message, and M increases by 1 for every new activity, in the order it is first introduced.
    - If the last turn continues talking about an activity or condition that was already introduced in an earlier turn (visible in the given context) AND the last turn contains a word or phrase that refers to it (e.g. a pronoun like "it"/"this", or a repeated mention): output the SAME shape as above — {"value": <verbatim phrase>, "offset": <int>, "length": <int>, "type": <activity_type>, "activity_id": "chat{N}.{M}"} — but reuse the EXISTING activity_id and the EXISTING type from when it was first introduced; do not mint a new id or invent a different type.
    - If the last turn continues talking about that activity or condition but contains no word or phrase that specifically refers to it (only new role information, e.g. a new time or location, is being added): output only {"activity_id": "chat{N}.{M}"}, reusing the existing id, without value, offset, length or type.
    - activity_type must be one of: $activity_type_values.

    **Semantic roles** (all optional, all arrays; only include a role if the text of the LAST turn supports it):
    - agent: the participant that controls/performs the activity, acting on a DIFFERENT participant (the patient). Only use agent together with patient when they are two distinct participants, e.g. "John moves the box" -> agent: John, patient: box.
    - patient: the participant affected by the activity, undergoing a change of state caused by an agent.
    - agent_patient: use INSTEAD of agent and patient when a single participant both controls the activity and undergoes the change it causes (a self-affecting action), e.g. "John cycles" -> agent_patient: John, NOT agent: John plus patient: John.
    - experiencer: the participant that experiences a state or condition, with no change of state and no separate agent causing it, e.g. "John has a headache" -> experiencer: John. Use experiencer instead of patient for a participant merely experiencing a condition.
    - participant: use ONLY for a participant in the activity that cannot properly be captured by agent, patient, agent_patient, or experiencer above -- e.g. a bystander or another party involved without controlling the activity or undergoing its change of state themselves.
    - qualification: a specific qualification of the activity or condition itself (not of a participant), e.g. "blood sugar levels are high" -> qualification: high. This INCLUDES how LONG the activity or condition itself lasted/took (its duration, e.g. "for 20 minutes", "for an hour", "for three days") -- duration is a qualification of the activity/condition, never a time role (see time/time_resolved below), even though it is phrased with a time span.
    - agent, patient, agent_patient, experiencer, participant, qualification, instrument, location: arrays of {"value": <verbatim phrase>, "type": <role_type>, "offset": <int>, "length": <int>}, where role_type is one of $role_type_values.
    - result: array of {"value": <verbatim phrase>, "type": <result_type>, "offset": <int>, "length": <int>}, where result_type is one of $result_type_values.
    - time: array of {"value": <verbatim phrase>, "offset": <int>, "length": <int>}. Only for WHEN the activity/condition happened, started, or recurs (e.g. "today", "yesterday", "every morning", "lately") -- NEVER for how long it lasted, which is a qualification (see above). A single utterance can and often does have both, as two separate roles, e.g. "I walked for 20 minutes today" -> qualification: "for 20 minutes" (duration), time: "today" (when).
    - time_resolved: array of {"time_expression": <verbatim phrase matching a "time" entry>, "temporal_type": <temporal_type>, "absolute_date": <"YYYY-MM-DD" or null>, "date_range_start": <"YYYY-MM-DD" or null>, "date_range_end": <"YYYY-MM-DD" or null>, "recurrence_pattern": <recurrence_pattern or null>}, grounding each time expression to a calendar date using the conversation's date as the reference point. temporal_type is one of $temporal_type_values. recurrence_pattern is one of $recurrence_pattern_values when the time expression clearly recurs on one of those patterns, otherwise null (e.g. "twice daily" is best captured as recurrence_pattern "daily" plus the free-text time value "twice daily").

    Every "offset" and "length" MUST be computed against the "utterance" text of the MOST RECENT user message only (0-indexed character offset, length in characters) — never against an earlier turn's utterance, and never against the surrounding JSON of the message itself. 
    Before including any span, verify it yourself: find "value" as an exact, case-sensitive substring of that utterance, set offset to the position of its first character (the utterance's own first character is position 0) and length to the number of characters in "value", 
    then re-read utterance[offset : offset + length] and confirm it reproduces "value" character for character. Do not estimate or guess an offset — locate the exact substring first.
    
    If there is no match of "value" with the "utterance" text of the MOST RECENT user message, DO NOT output the activity or role.
    
    Do not output any other text than the JSON array.

    <start of examples>
    Example 1:
        Input: {"chat": 6, "human": "Jan", "date": "2010,Dec,13", "turn": 1, "speaker": "Jan", "utterance": "I've been noticing tingling in my feet lately. Is this something common with Type 2 Diabetes?"}
        Output: [
                    {
                        "perspective": {"emotion": "nervousness", "factuality": "expect", "certainty": "uncertain"},
                        "activity": {"value": "tingling in my feet", "offset": 19, "length": 19, "type": "physical condition", "activity_id": "chat6.1"},
                        "agent": [], "patient": [], "agent_patient": [],
                        "experiencer": [{"value": "I", "type": "person", "offset": 0, "length": 1}],
                        "participant": [], "qualification": [],
                        "instrument": [],
                        "location": [{"value": "in my feet", "type": "body_part", "offset": 28, "length": 10}],
                        "result": [],
                        "time": [{"value": "lately", "offset": 39, "length": 6}],
                        "time_resolved": [{"time_expression": "lately", "temporal_type": "range", "absolute_date": null, "date_range_start": "2010-12-06", "date_range_end": "2010-12-13", "recurrence_pattern": null}]
                    }
                ]

    Example 2 (turn 2 continues the conversation from Example 1; the first Input line below is the PRIOR turn, given only as context — turn 1 is not re-annotated here. Turn 2 refers back to the activity introduced in turn 1 using the pronoun "it" — reuse its activity_id "chat6.1" and its type "physical condition"):
        Input: {"chat": 6, "human": "Jan", "date": "2010,Dec,13", "turn": 1, "speaker": "Jan", "utterance": "I've been noticing tingling in my feet lately. Is this something common with Type 2 Diabetes?"}
        Input: {"chat": 6, "human": "Jan", "date": "2010,Dec,13", "turn": 2, "speaker": "Jan", "utterance": "I think it has actually gotten worse over the past week, especially in my toes."}
        Output: [
                    {
                        "perspective": {"emotion": "disappointment", "factuality": "confirm", "certainty": "certain"},
                        "activity": {"value": "it", "offset": 8, "length": 2, "type": "physical condition", "activity_id": "chat6.1"},
                        "agent": [], "patient": [], "agent_patient": [],
                        "experiencer": [{"value": "I", "type": "person", "offset": 0, "length": 1}],
                        "participant": [], "qualification": [],
                        "instrument": [],
                        "location": [{"value": "toes", "type": "body_part", "offset": 74, "length": 4}],
                        "result": [],
                        "time": [{"value": "the past week", "offset": 42, "length": 13}],
                        "time_resolved": [{"time_expression": "the past week", "temporal_type": "range", "absolute_date": null, "date_range_start": "2010-12-06", "date_range_end": "2010-12-13", "recurrence_pattern": null}]
                    }
                ]

    Example 3 (further continuation of the chat6 conversation; only turn 4, the MOST RECENT message, is annotated below — turns 1-3 are prior context. Turn 4 is an elliptical answer to turn 3's question and contains no word or phrase referring back to the activity itself, so the reference is bare — activity_id only, no value/offset/length/type):
        Input: {"chat": 6, "human": "Jan", "date": "2010,Dec,13", "turn": 1, "speaker": "Jan", "utterance": "I've been noticing tingling in my feet lately. Is this something common with Type 2 Diabetes?"}
        Input: {"chat": 6, "human": "Jan", "date": "2010,Dec,13", "turn": 2, "speaker": "Jan", "utterance": "I think it has actually gotten worse over the past week, especially in my toes."}
        Input: {"chat": 6, "human": "Jan", "date": "2010,Dec,13", "turn": 3, "speaker": "agent", "utterance": "Does it happen mostly at night?"}
        Input: {"chat": 6, "human": "Jan", "date": "2010,Dec,13", "turn": 4, "speaker": "Jan", "utterance": "Mostly in the evenings, especially after standing a lot."}
        Output: [
                    {
                        "perspective": {"emotion": "neutral", "factuality": "confirm", "certainty": "certain"},
                        "activity": {"activity_id": "chat6.1"},
                        "agent": [], "patient": [], "agent_patient": [], "experiencer": [], "participant": [], "qualification": [], "instrument": [],
                        "location": [], "result": [],
                        "time": [{"value": "the evenings", "offset": 10, "length": 12}],
                        "time_resolved": []
                    }
                ]

    Example 4 (agent_patient: "I" both performs the cycling and is the one moving — a single self-affecting participant, not separate agent and patient):
        Input: {"chat": 9, "human": "Ingrid", "date": "2013,May,02", "turn": 1, "speaker": "Ingrid", "utterance": "I go cycling every morning."}
        Output: [
                    {
                        "perspective": {"emotion": "neutral", "factuality": "confirm", "certainty": "certain"},
                        "activity": {"value": "cycling", "offset": 5, "length": 7, "type": "exercise", "activity_id": "chat9.1"},
                        "agent": [], "patient": [],
                        "agent_patient": [{"value": "I", "type": "person", "offset": 0, "length": 1}],
                        "experiencer": [], "participant": [], "qualification": [], "instrument": [], "location": [], "result": [],
                        "time": [{"value": "every morning", "offset": 13, "length": 13}],
                        "time_resolved": [{"time_expression": "every morning", "temporal_type": "recurring", "absolute_date": null, "date_range_start": null, "date_range_end": null, "recurrence_pattern": "daily"}]
                    }
                ]

    Example 5 (only turn 2, the MOST RECENT message, is annotated below — turn 1 is prior context):
        Input: {"chat": 7, "human": "Fatima", "date": "2012,Jan,31", "turn": 1, "speaker": "agent", "utterance": "Fatima, I understand that you've been managing your Type 2 Diabetes for quite some time now. Can you tell me more about your current medication regimen?"}
        Input: {"chat": 7, "human": "Fatima", "date": "2012,Jan,31", "turn": 2, "speaker": "Fatima", "utterance": "I take metformin tablets twice daily, and also an evening insulin injection. Besides, I take a daily aspirin for heart health, as advised by my doctor."}
        Output: [
                    {
                        "perspective": {"emotion": "neutral", "factuality": "confirm", "certainty": "certain"},
                        "activity": {"value": "take", "offset": 2, "length": 4, "type": "take_medicine", "activity_id": "chat7.1"},
                        "instrument": [{"value": "metformin tablets", "offset": 7, "length": 17, "type": "substance", "activity_id": "chat7.1"}],
                        "agent_patient": [{"value": "I", "type": "person", "offset": 0, "length": 1}],
                        "patient": [], "agent": [], "experiencer": [], "participant": [], "qualification": [], "location": [], "result": [],
                        "time": [{"value": "twice daily", "offset": 25, "length": 11}],
                        "time_resolved": [{"time_expression": "twice daily", "temporal_type": "recurring", "absolute_date": null, "date_range_start": null, "date_range_end": null, "recurrence_pattern": "daily"}]
                    },
                    {
                        "perspective": {"emotion": "neutral", "factuality": "confirm", "certainty": "certain"},
                        "activity": {"value": "insulin injection", "offset": 58, "length": 17, "type": "take_medicine", "activity_id": "chat7.2"},
                        "agent_patient": [{"value": "I", "type": "person", "offset": 0, "length": 1}],
                        "instrument": [{"value": "insulin", "offset": 58, "length": 7, "type": "substance"}],
                        "agent": [], "patient": [], "experiencer": [], "participant": [], "qualification": [], "location": [], "result": [],
                        "time": [{"value": "evening", "offset": 50, "length": 7}],
                        "time_resolved": []
                    },
                    {
                        "perspective": {"emotion": "neutral", "factuality": "confirm", "certainty": "certain"},
                        "activity": {"value": "take", "offset": 88, "length": 4, "type": "take_medicine", "activity_id": "chat7.3"},
                        "instrument": [{"value": "aspirin", "offset": 101, "length": 7, "type": "substance"}],
                        "agent_patient": [{"value": "I", "type": "person", "offset": 86, "length": 1}],
                        "patient": [], "agent": [], "experiencer": [], "participant": [], "qualification": [],"location": [],
                        "result": [{"value": "heart health", "type": "goal", "offset": 113, "length": 12}],
                        "time": [{"value": "daily", "offset": 95, "length": 5}],
                        "time_resolved": []
                    }
                ]

    Example 6 (only turn 4, the MOST RECENT message, is annotated below; turn 2's food items are ignored because they belong to an earlier turn; "weight gain" and "dizziness" are conditions Fatima experiences, not actions she controls, so the participant role is experiencer, not agent):
        Input: {"chat": 8, "human": "Fatima", "date": "2012,Jan,31", "turn": 1, "speaker": "agent", "utterance": "What did you eat yesterday"}
        Input: {"chat": 8, "human": "Fatima", "date": "2012,Jan,31", "turn": 2, "speaker": "Fatima", "utterance": "Yes, for lunch, I had grilled chicken with salad, and for dinner, I cooked a fish with some steamed vegetables."}
        Input: {"chat": 8, "human": "Fatima", "date": "2012,Jan,31", "turn": 3, "speaker": "agent", "utterance": "Any other worries?"}
        Input: {"chat": 8, "human": "Fatima", "date": "2012,Jan,31", "turn": 4, "speaker": "Fatima", "utterance": "I do have some concerns about the side effects of my medications, especially the insulin injection. I've been experiencing some weight gain and occasional dizziness. I'm not sure if these are common side effects or if I should discuss them with my doctor."}
        Output: [
                    {
                        "perspective": {"emotion": "nervousness", "factuality": "confirm", "certainty": "uncertain"},
                        "activity": {"value": "weight gain", "offset": 128, "length": 11, "type": "physical condition", "activity_id": "chat8.1"},
                        "agent": [], "patient": [], "agent_patient": [],
                        "experiencer": [{"value": "I", "type": "person", "offset": 100, "length": 1}],
                        "participant": [], "qualification": [],
                        "instrument": [], "location": [], "result": [], "time": [],
                        "time_resolved": []
                    },
                    {
                        "perspective": {"emotion": "nervousness", "factuality": "confirm", "certainty": "uncertain"},
                        "activity": {"value": "dizziness", "offset": 155, "length": 9, "type": "physical condition", "activity_id": "chat8.2"},
                        "agent": [], "patient": [], "agent_patient": [],
                        "experiencer": [{"value": "I", "type": "person", "offset": 100, "length": 1}],
                        "participant": [], "qualification": [],
                        "instrument": [], "location": [], "result": [],
                        "time": [{"value": "occasional", "offset": 144, "length": 10}],
                        "time_resolved": []
                    }
                ]

    Example 7 (qualification and participant: "high" qualifies the blood sugar levels themselves, not a participant, so it is qualification, not a role on a person; "my sister" is present but neither controls the activity nor undergoes its change of state, so she is participant, not agent/patient/agent_patient):
        Input: {"chat": 10, "human": "Julia", "date": "2014,Mar,02", "turn": 1, "speaker": "Julia", "utterance": "My blood sugar levels have been high lately, and my sister was there when I checked them."}
        Output: [
                    {
                        "perspective": {"emotion": "nervousness", "factuality": "confirm", "certainty": "certain"},
                        "activity": {"value": "blood sugar levels", "offset": 3, "length": 18, "type": "measurement", "activity_id": "chat10.1"},
                        "agent": [], "patient": [], "agent_patient": [],
                        "experiencer": [{"value": "My", "type": "person", "offset": 0, "length": 2}],
                        "participant": [{"value": "my sister", "type": "person", "offset": 49, "length": 9}],
                        "qualification": [{"value": "high", "type": "condition", "offset": 32, "length": 4}],
                        "instrument": [], "location": [], "result": [],
                        "time": [{"value": "lately", "offset": 37, "length": 6}],
                        "time_resolved": []
                    }
                ]

    Example 8 (sleep: "I" both goes to sleep and is the one sleeping, a self-affecting action, so agent_patient, not agent/patient; "6 hours" and "wasn't very restful" both qualify the sleep activity itself, so qualification, not a role on a person):
        Input: {"chat": 11, "human": "Sam", "date": "2015,Jun,10", "turn": 1, "speaker": "Sam", "utterance": "I slept about 6 hours last night, but it wasn't very restful."}
        Output: [
                    {
                        "perspective": {"emotion": "disappointment", "factuality": "confirm", "certainty": "certain"},
                        "activity": {"value": "slept", "offset": 2, "length": 5, "type": "sleep", "activity_id": "chat11.1"},
                        "agent": [], "patient": [],
                        "agent_patient": [{"value": "I", "type": "person", "offset": 0, "length": 1}],
                        "experiencer": [], "participant": [],
                        "qualification": [
                            {"value": "6 hours", "type": "other", "offset": 14, "length": 7},
                            {"value": "wasn't very restful", "type": "other", "offset": 41, "length": 19}
                        ],
                        "instrument": [], "location": [], "result": [],
                        "time": [{"value": "last night", "offset": 22, "length": 10}],
                        "time_resolved": [{"time_expression": "last night", "temporal_type": "point", "absolute_date": "2015-06-09", "date_range_start": null, "date_range_end": null, "recurrence_pattern": null}]
                    }
                ]

    Example 9 (duration vs. time: "for 20 minutes" says how LONG the walk lasted, so it is a qualification of the "walked" activity, NOT a time role, even though it is phrased with a time span; "today" says WHEN the walk happened, so that one -- and only that one -- is the time role):
        Input: {"chat": 12, "human": "Leo", "date": "2016,Apr,03", "turn": 1, "speaker": "Leo", "utterance": "I walked for 20 minutes today"}
        Output: [
                    {
                        "perspective": {"emotion": "neutral", "factuality": "confirm", "certainty": "certain"},
                        "activity": {"value": "walked", "offset": 2, "length": 6, "type": "exercise", "activity_id": "chat12.1"},
                        "agent": [], "patient": [],
                        "agent_patient": [{"value": "I", "type": "person", "offset": 0, "length": 1}],
                        "experiencer": [], "participant": [],
                        "qualification": [{"value": "20 minutes", "type": "other", "offset": 13, "length": 10}],
                        "instrument": [], "location": [], "result": [],
                        "time": [{"value": "today", "offset": 24, "length": 5}],
                        "time_resolved": [{"time_expression": "today", "temporal_type": "point", "absolute_date": "2016-04-03", "date_range_start": null, "date_range_end": null, "recurrence_pattern": null}]
                    }
                ]
    <end of examples>
    ''')

prompt_conversational_srl_annotation = _prompt_conversational_srl_annotation_template.substitute(
        emotion_values=quoted_values(EmotionLabel),
        factuality_values=quoted_values(Factuality),
        certainty_values=quoted_values(Certainty),
        activity_type_values=quoted_values(ActivityType),
        role_type_values=quoted_values(RoleType),
        result_type_values=quoted_values(ResultType),
        temporal_type_values=quoted_values(TemporalType),
        recurrence_pattern_values=quoted_values(RecurrencePattern),
)

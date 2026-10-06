import random
from cltl.commons.discrete import UtteranceType
from datetime import date, datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
# from perspective.emotion_extraction import GoEmotionDetector
#
# model_path = "AnasAlokla/multilingual_go_emotions"
# #  Languages: Arabic, English, French, Spanish, Dutch, Turkish
# emotion_detector = GoEmotionDetector(model=model_path)
#
negation_words = [    "not", "never", "nobody", "no", "none", "neither", "nor", "hardly", "scarcely", "rarely", "seldom", "little", "few"]
certainty_words = ["certain", "sure", "definitely", "absolutely", "certainly", "surely"]
uncertainty_words = ["think", "believe", "might", "maybe", "could", "perhaps"]
low_level_words = ["difficult", "problem", "cannot", "unable", "bad", "impossible"]
high_level_words = ["well", "can", "easy", "good", "possible"]

def prune_neutral_emotion(emotion_values):
    for emotion in emotion_values:
        if emotion == "neutral":
            emotion_values.remove(emotion)

def get_utterance_perspective(utterance: str, emotion_detector):

    perspective= {"certainty": 1, "polarity": 1, "sentiment": 0}
    go_emotions, ekman_emotions, sentiments = emotion_detector.extract_text_emotions(utterance, threshold=0.6)
    if len(sentiments)>0:
        for sentiment in sentiments:
            if sentiment.value == "positive":
                perspective["sentiment"] = 1
            elif sentiment.value == "negative":
                perspective["sentiment"] = -1
    if len(go_emotions)>0:
        go_values = []
        for go_emotion in go_emotions:
            go_values.append(go_emotion.value)
        prune_neutral_emotion(go_values)
        if len(go_values)>0:
            perspective["emotion"] = go_values

    #### OR use ekman values
    # if len(ekman_emotions)>0:
    #     ekman_values = []
    #     for ekman_emotion in ekman_emotions:
    #         ekman_values.append(ekman_emotion.value)
    #     perspective["emotion"] = ekman_values

    utterance_lower = utterance.lower()
    utterance_tokens = utterance_lower.split(" ")
    for negation in negation_words:
        if negation in utterance_tokens:
            perspective["polarity"] = -1
            break
    certainty_score = 0.0
    for certainty in certainty_words:
        if certainty in utterance_tokens:
            certainty_score+=0.5
    for uncertainty in uncertainty_words:
        if uncertainty in utterance_tokens:
            certainty_score-=0.5
    perspective["certainty"] = certainty_score
    level_score = 2.0
    for level in high_level_words:
        if level in utterance_tokens:
            level_score+=1.0
    for level in low_level_words:
        if level in utterance_tokens:
            level_score-=1.0
    perspective["level"] = level_score
    return perspective

def get_triples_from_object(event, event_id):
    triples = []
    predicate_objects = []
    if event.activity:
        subject = event.activity
        subject_uri = "http://cltl.nl/leolani/n2mu/"+event.activity.replace(" ", "_")+str(event_id)
        if event.agent:
            if type(event.agent)==str:
                event.agent = [event.agent]
            elif type(event.agent)==tuple:
                event.agent = list(event.agent)
            for agent in event.agent:
                triple = {"subject": {"label": subject, "type": ["activity"], "uri": subject_uri},
                  "predicate": {"label": "agent", "uri": "http://cltl.nl/leolani/n2mu/agent"},
                  "object": {"label":agent, "type": ["agent"], "uri": ""}}
                triples.append(triple) 
        if event.patient:
            if type(event.patient)==str:
                event.patient = [event.patient]
            elif type(event.patient)==tuple:
                event.patient = list(event.patient)
            for patient in event.patient:
                triple = {"subject": {"label": subject, "type": ["activity"], "uri": subject_uri},
                  "predicate": {"label": "patient", "uri": "http://cltl.nl/leolani/n2mu/patient"},
                  "object": {"label": patient, "type": ["agent", "object"], "uri": ""}}
                triples.append(triple) 
        if event.manner:
            if type(event.manner)==str:
                event.manner = [event.manner]
            elif type(event.manner)==tuple:
                event.manner = list(event.manner)
            for manner in event.manner:
                triple = {"subject": {"label": subject, "type": ["activity"], "uri": subject_uri},
                  "predicate": {"label": "manner", "uri": "http://cltl.nl/leolani/n2mu/manner"},
                  "object": {"label": manner, "type": ["property"], "uri": ""}}
                triples.append(triple) 
        if event.instrument:
            if type(event.instrument)==str:
                event.instrument = [event.instrument]
            elif type(event.instrument)==tuple:
                event.instrument = list(event.instrument)
            for instrument in event.instrument:
                triple = {"subject": {"label": subject, "type": ["activity"], "uri": subject_uri},
                  "predicate": {"label": "instrument", "uri": "http://cltl.nl/leolani/n2mu/instrument"},
                  "object": {"label": instrument, "type": ["instrument"], "uri": ""}}
                triples.append(triple) 
        if event.location:
            if type(event.location)==str:
                event.location = [event.location]
            elif type(event.location)==tuple:
                event.location = list(event.location)
            for location in event.location:
                triple = {"subject": {"label": subject, "type": ["activity"], "uri": subject_uri},
                  "predicate": {"label": "location", "uri": "http://cltl.nl/leolani/n2mu/location"},
                  "object": {"label": location, "type": ["place"], "uri": ""}}
                triples.append(triple) 
        if event.time:
            if type(event.time)==str:
                event.time = [event.time]
            elif type(event.time)==tuple:
                event.time = list(event.time)
            for time in event.time:
                triple = {"subject": {"label": subject, "type": ["activity"], "uri": subject_uri},
                  "predicate": {"label": "time", "uri": "http://cltl.nl/leolani/n2mu/time"},
                  "object": {"label": time, "type": ["time"], "uri": ""}}
                triples.append(triple) 
    return triples

def get_triples(event, event_id):
    triples = []
    if 'activity' in event and not event['activity'] is None:
        subject = event['activity']
        subject_uri = "http://cltl.nl/leolani/n2mu/"+subject.replace(" ", "_")+str(event_id)
        activity_type = ["activity"]
        if 'activity_type' in event and not event['activity_type']==None:
            activity_type.append(event['activity_type'])
        if 'agent' in event and not event['agent'] is None:
            if type(event['agent'])==str:
                event['agent'] = [event['agent']]
            for agent in event['agent']:
                triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "agent", "uri": "http://cltl.nl/leolani/n2mu/agent"},
                  "object": {"label":agent, "type": ["agent"], "uri": ""}}
                triples.append(triple) 
        if 'patient' in event and not event['patient'] is None:
            if type(event['patient'])==str:
                event['patient'] = [event['patient']]
            for patient in event['patient']:
                triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "patient", "uri": "http://cltl.nl/leolani/n2mu/patient"},
                  "object": {"label": patient, "type": ["agent", "object"], "uri": ""}}
                triples.append(triple) 
        if 'manner' in event and not event['manner'] is None:
            if type(event['manner'])==str:
                event['manner'] = [event['manner']]
            for manner in event['manner']:
                triple = {"subject": {"label": subject, "type":activity_type, "uri": subject_uri},
                  "predicate": {"label": "manner", "uri": "http://cltl.nl/leolani/n2mu/manner"},
                  "object": {"label": manner, "type": ["property"], "uri": ""}}
                triples.append(triple) 
        if 'instrument' in event and not event['instrument'] is None:
            if type(event['instrument'])==str:
                event['instrument'] = [event['instrument']]
            for instrument in event['instrument']:
                triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "instrument", "uri": "http://cltl.nl/leolani/n2mu/instrument"},
                  "object": {"label": instrument, "type": ["instrument"], "uri": ""}}
                triples.append(triple) 
        if 'location' in event and not event['location'] is None:
            if type(event['location'])==str:
                event['location'] = [event['location']]
            for location in event['location']:
                triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "location", "uri": "http://cltl.nl/leolani/n2mu/location"},
                  "object": {"label": location, "type": ["place"], "uri": ""}}
                triples.append(triple) 
        if 'time' in event and not event['time'] is None:
            if type(event['time'])==str:
                event['time'] = [event['time']]
            for time in event['time']:
                triple = {"subject": {"label": subject, "type":activity_type, "uri": subject_uri},
                  "predicate": {"label": "time", "uri": "http://cltl.nl/leolani/n2mu/time"},
                  "object": {"label": time, "type": ["time"], "uri": ""}}
                triples.append(triple) 
    return triples

def get_triples_with_types(event, event_id, utterence_time:date):
    triples = []
    if 'activity' in event and not event['activity'] is None:
        subject = event['activity']
        subject_uri = "http://cltl.nl/leolani/n2mu/"+subject.replace(" ", "_")+str(event_id)
        activity_type = ["activity"]
        if 'activity_type' in event and not event['activity_type']==None:
            activity_type.append(event['activity_type'])
        if 'time_resolved' in event and not event['time_resolved'] is None:
            for time in event['time_resolved']:
                a_type = time["temporal_type"]
                if a_type == "recurring" or a_type=="vague":
                    activity_type.append("<https://cltl.nl/eckg/EventSeries")
                    break
        if 'agent' in event and not event['agent'] is None:
            if type(event['agent'])==str:
                event['agent'] = [event['agent']]
            for agent in event['agent']:
                a_label = agent["value"]
                a_type = agent["type"]
                triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "agent", "uri": "http://cltl.nl/leolani/n2mu/agent"},
                  "object": {"label":a_label, "type": [a_type], "uri": ""}}
                triples.append(triple)
        if 'patient' in event and not event['patient'] is None:
            if type(event['patient'])==str:
                event['patient'] = [event['patient']]
            for patient in event['patient']:
                a_label = patient["value"]
                a_type = patient["type"]
                triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "patient", "uri": "http://cltl.nl/leolani/n2mu/patient"},
                  "object": {"label":a_label, "type": [a_type], "uri": ""}}
                triples.append(triple)
        if 'manner' in event and not event['manner'] is None:
            if type(event['manner'])==str:
                event['manner'] = [event['manner']]
            for manner in event['manner']:
                triple = {"subject": {"label": subject, "type":activity_type, "uri": subject_uri},
                  "predicate": {"label": "manner", "uri": "http://cltl.nl/leolani/n2mu/manner"},
                  "object": {"label": manner, "type": ["property"], "uri": ""}}
                triples.append(triple)
        if 'instrument' in event and not event['instrument'] is None:
            if type(event['instrument'])==str:
                event['instrument'] = [event['instrument']]
            for instrument in event['instrument']:
                a_label = instrument["value"]
                a_type = instrument["type"]
                triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "instrument", "uri": "http://cltl.nl/leolani/n2mu/instrument"},
                  "object": {"label":a_label, "type": [a_type], "uri": ""}}
                triples.append(triple)
        if 'location' in event and not event['location'] is None:
            if type(event['location'])==str:
                event['location'] = [event['location']]
            for location in event['location']:
                a_label = location["value"]
                a_type = location["type"]
                triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "location", "uri": "http://cltl.nl/leolani/n2mu/location"},
                  "object": {"label":a_label, "type": [a_type], "uri": ""}}
                triples.append(triple)
        if 'time_resolved' in event and not event['time_resolved'] is None:
            for time in event['time_resolved']:
                a_label = time["time_expression"]
                a_type = time["temporal_type"]
                uri = ""
                time_type = "dateTime"
                if "date_range_start" in time and time["date_range_start"] is not None:
                    time_type = "rangeTime"
                    uri = "http://cltl.nl/leolani/n2mu/time/" + parser.parse(time["date_range_start"]).date().isoformat()
                elif "absolute_date" in time and time["absolute_date"] is not None:
                    time_type = "dateTime"
                    uri = "http://cltl.nl/leolani/n2mu/time/" + parser.parse(time["absolute_date"]).date().isoformat()
                elif a_type=="recurring":
                    ## We define a date 2 months ago as a baseline proxy for a series of recurring events
                    time_type = "recurringTime"
                    uri = "http://cltl.nl/leolani/n2mu/time/" + (utterence_time - relativedelta(months=2)).isoformat()
                elif a_type=="vague":
                    ## We define a date 1 month ago as a proxy for a vagualy defines series of events
                    time_type = "vagueTime"
                    uri = "http://cltl.nl/leolani/n2mu/time/" + (utterence_time - relativedelta(months=1)).isoformat()
                triple = {"subject": {"label": subject, "type":activity_type, "uri": subject_uri},
                  "predicate": {"label": "time", "uri": "http://cltl.nl/leolani/n2mu/time/"+time_type},
                  "object": {"label": a_label, "type": [a_type], "uri": uri}}
                triples.append(triple)
    return triples

# The roles carried by the current SRL schema (data/event_srl.json.zip) that behave like
# agent/patient/instrument/location above: a list of {"value", "type", "offset", "length"}
# dicts. "result" is handled separately below (its own ResultType vocabulary, not RoleType);
# "time" is handled via time_resolved only, matching get_triples_with_types; there is no
# "manner" role in this schema.
ROLE_FIELDS_WITH_TYPE = ["agent", "patient", "agent_patient", "experiencer", "participant", "qualification", "instrument", "location"]

# Closed sets of first-/second-person pronouns a role filler's verbatim phrase is resolved
# against (see _pronoun_identity() below) -- exactly the forms named for this: not "we"/"us"/
# "your" or other inflections, and not matched as a substring of a longer phrase, only as the
# filler's *entire* value.
FIRST_PERSON_PRONOUNS = {"i", "me", "mine", "myself"}
SECOND_PERSON_PRONOUNS = {"you", "yours", "yourself"}


def _pronoun_identity(value, speaker, other):
    """If `value` (a role filler's verbatim phrase) is exactly a first- or second-person
    pronoun, return the identity it refers to: `speaker` for I/me/mine/myself, `other` (the
    other party in the two-party conversation) for you/yours/yourself. Returns None for
    anything else (including when `speaker`/`other` themselves are unknown), so the caller can
    fall back to using the phrase itself."""
    if not value or (not speaker and not other):
        return None
    normalized = value.strip().lower()
    if normalized in FIRST_PERSON_PRONOUNS:
        return speaker
    if normalized in SECOND_PERSON_PRONOUNS:
        return other
    return None


def _role_filler_object(filler, speaker=None, other=None):
    """Build a role filler's RDF object dict. If its verbatim phrase is a first-/second-person
    pronoun, this resolves it to the identity it refers to (see _pronoun_identity()) and links
    it with the same "http://cltl.nl/leolani/friends/<name>" URI used for a turn's own author,
    instead of leaving the bare pronoun as an unlinked literal with no URI -- the knowledge
    graph should hold identities, not pronouns. Falls back to the phrase itself (unresolved,
    literal, no URI) when it isn't a pronoun, or speaker/other weren't given."""
    identity = _pronoun_identity(filler.get('value'), speaker, other)
    if identity:
        return {"label": identity, "type": ["person"], "uri": "http://cltl.nl/leolani/friends/" + identity}
    return {"label": filler.get('value'), "type": [filler.get('type')], "uri": ""}


# Semantic roles that identify WHO performed/experienced an activity -- if none of these got a
# filler from the SRL extraction, add_speaker_as_agent() defaults the turn's own speaker as the
# implicit agent instead of leaving the activity with no agent-like role at all.
AGENT_LIKE_ROLES = ("agent", "agent_patient", "participant", "experiencer")


def add_speaker_as_agent(subject, subject_uri, speaker: str):
    """Default `speaker` (the turn's own speaker -- a plain identity string, e.g. "Mehmet" or
    "agent"; the SAME string get_triples_with_types_and_activity_id()'s caller already uses for
    its capsule's "author" field, NOT a URI) as the implicit `agent` of `subject`'s activity,
    linked to that identity's "http://cltl.nl/leolani/friends/<name>" URI -- the same scheme
    _role_filler_object() uses to resolve a first-/second-person pronoun ("I"/"you") to an
    identity. This is that same kind of resolution, just for when the activity got no
    agent-like role filler at all, explicit pronoun or otherwise.

    speaker must be a bare identity string, not a URI: an earlier version of this function
    assumed the opposite (tried to slice a name out of it via speaker.rfind('/'), and used it
    directly as the object's "uri") -- since speaker never actually looks like a URI here, that
    built things like {"uri": "Mehmet"}, which crashes deep in cltl.brain's RDF layer
    (iribaker.to_iri: "no scheme or no net location part") the moment such a triple is pushed.
    """
    agent_uri = "http://cltl.nl/leolani/friends/" + speaker
    triple = {"subject": {"label": subject, "type": ["activity"], "uri": subject_uri},
              "predicate": {"label": "agent", "uri": "http://cltl.nl/leolani/n2mu/agent"},
              "object": {"label": speaker, "type": ["agent"], "uri": agent_uri}}
    return triple

def get_triples_with_types_and_activity_id(event, utterance_time: date, speaker=None, other=None):
    """Like get_triples_with_types, but for the current SRL schema, where 'activity' is a dict
    that carries its own activity_id (e.g. "chat0.1") -- assigned once per real-world activity
    and reused on every later mention, including a bare reference entry with no phrase of its
    own -- instead of the activity phrase text being combined with a random/context-tracked
    event_id to build the subject URI. Using activity_id directly means the same real-world
    activity always gets the same subject URI, with no need to track phrases across turns.

    Also covers the full current role set (agent, patient, agent_patient, experiencer,
    participant, qualification, instrument, location, result, time) instead of the old
    agent/patient/manner/instrument/location/time set -- "manner" no longer exists in this
    schema.

    `speaker` and `other` (the turn's own speaker and the other party in the two-party
    conversation, both plain identity strings -- a patient's name, or "agent") are used to
    resolve any role filler that's a first-/second-person pronoun to that identity instead of
    leaving the bare pronoun in the graph -- see _role_filler_object(). Pass None for both to
    disable this (fillers are then always the literal phrase, as before).
    """
    triples = []
    activity = event.get('activity') or {}
    activity_id = activity.get('activity_id')
    if not activity_id:
        return triples

    # A bare reference entry (a later mention with no phrase of its own, just attaching more
    # role info to an activity introduced earlier) has no "value" here; fall back to the
    # activity_id itself as the subject label since this function has no cross-turn state to
    # look up the phrase that originally introduced it.
    subject = activity.get('value') or activity_id
    subject_uri = "http://cltl.nl/leolani/n2mu/" + activity_id
    activity_type = ["activity"]
    if activity.get('type'):
        activity_type.append(activity['type'])
    for time in (event.get('time_resolved') or []):
        if time.get('temporal_type') in ("recurring", "vague"):
            activity_type.append("<https://cltl.nl/eckg/EventSeries")
            break

    for role in ROLE_FIELDS_WITH_TYPE:
        for filler in (event.get(role) or []):
            triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                      "predicate": {"label": role, "uri": "http://cltl.nl/leolani/n2mu/" + role},
                      "object": _role_filler_object(filler, speaker, other)}
            triples.append(triple)

    # Exactly one synthetic "speaker is the agent" triple per activity, only when NONE of the
    # agent-like roles (AGENT_LIKE_ROLES) got a filler -- not one check per role in
    # ROLE_FIELDS_WITH_TYPE (which would fire once per role with no filler, e.g. once each for
    # "location", "instrument", ... -- clearly not the intent). `speaker` can be None/"" when
    # the caller passed neither speaker nor other (see this function's own docstring); skip
    # rather than push a broken identity in that case.
    if speaker and not any(event.get(role) for role in AGENT_LIKE_ROLES):
        triples.append(add_speaker_as_agent(subject, subject_uri, speaker))
    for result in (event.get('result') or []):
        triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "result", "uri": "http://cltl.nl/leolani/n2mu/result"},
                  "object": _role_filler_object(result, speaker, other)}
        triples.append(triple)

    for time in (event.get('time_resolved') or []):
        a_label = time.get('time_expression')
        a_type = time.get('temporal_type')
        uri = ""
        time_type = "dateTime"
        if time.get('date_range_start'):
            time_type = "rangeTime"
            uri = "http://cltl.nl/leolani/n2mu/time/" + parser.parse(time['date_range_start']).date().isoformat()
        elif time.get('absolute_date'):
            time_type = "dateTime"
            uri = "http://cltl.nl/leolani/n2mu/time/" + parser.parse(time['absolute_date']).date().isoformat()
        elif a_type == "recurring":
            ## We define a date 2 months ago as a baseline proxy for a series of recurring events
            time_type = "recurringTime"
            uri = "http://cltl.nl/leolani/n2mu/time/" + (utterance_time - relativedelta(months=2)).isoformat()
        elif a_type == "vague":
            ## We define a date 1 month ago as a proxy for a vaguely defined series of events
            time_type = "vagueTime"
            uri = "http://cltl.nl/leolani/n2mu/time/" + (utterance_time - relativedelta(months=1)).isoformat()
        triple = {"subject": {"label": subject, "type": activity_type, "uri": subject_uri},
                  "predicate": {"label": "time", "uri": "http://cltl.nl/leolani/n2mu/time/" + time_type},
                  "object": {"label": a_label, "type": [a_type], "uri": uri}}
        triples.append(triple)

    return triples

def get_capsule_with_event_details_from_turn (turn_data, emotion_detector):
    turn = turn_data['Input']
    event_data = turn_data['Output']
    chat_id = turn_data['chat']
    chat_date = parser.parse(turn_data['date'])
    turn_id = turn['turn']
    if event_data:
        ### We use a random digit to make the event reference unique
        ### This random digit is combined with the activity expression to identify the event (activity or condition)
        event_id = random.random()
        triples = get_triples(event_data, event_id)
        offset = "0-"+str(len(turn["utterance"]))
        perspective_value =get_utterance_perspective(turn["utterance"], emotion_detector)
        capsule = { "chat": chat_id,
            "turn": turn_id,
            "author": {"label":turn['speaker'], "type": ["agent"], "uri":"http://cltl.nl/leolani/friends/"+turn['speaker']},
            "utterance": turn["utterance"],
            "utterance_type": UtteranceType.STATEMENT,
            "position": offset,
            "perspective":  perspective_value,
             "timestamp": datetime.combine(chat_date, datetime.now().time()),
             "context_id": event_id
        }
        event_details_list = []
        for triple in triples:
            event_details_list.append({
                "subject" : triple["subject"],
                "predicate" : triple["predicate"],
                "object" : triple["object"]})
        capsule["event_details"] = event_details_list
    return capsule

## One event identified by phrase per conversation approach
# ## This variant taks as paramter a dict with phrases and event identifiers. If the activity phrase is in the dict, the identifier is re-used
def get_capsule_with_event_details_from_turn_with_conversationa_context (conversational_context: {}, turn_data, emotion_detector):
    turn = turn_data['Input']
    event_data_list = turn_data['Output']
    chat_id = turn_data['chat']
    chat_date = parser.parse(turn_data['date'])
    turn_id = turn['turn']
    for event_data in event_data_list:
        ### We use a random digit to make the event reference unique
        ### This random digit is combined with the activity expression to identify the event (activity or condition)
        ### We first check the conversational context if such a phrase was already mentioned.
        ### If so, we re-use the ID.
        ### @TODO add variants to the conversational context and a similarity function.
        event_id = random.random()
        print('event_data', event_data, type(event_data))
        subject_phrase = event_data['activity']
        if subject_phrase in conversational_context:
            event_id = conversational_context[subject_phrase]
        else:
            conversational_context[subject_phrase] = event_id
        triples = get_triples(event_data, event_id)
        offset = "0-"+str(len(turn["utterance"]))
        perspective_value =get_utterance_perspective(turn["utterance"], emotion_detector)
        capsule = { "chat": chat_id,
            "turn": turn_id,
            "author": {"label":turn['speaker'], "type": ["agent"], "uri":"http://cltl.nl/leolani/friends/"+turn['speaker']},
            "utterance": turn["utterance"],
            "utterance_type": UtteranceType.STATEMENT,
            "position": offset,
            "perspective":  perspective_value,
             "timestamp": datetime.combine(chat_date, datetime.now().time()),
             "context_id": event_id
        }
        event_details_list = []
        for triple in triples:
            event_details_list.append({
                "subject" : triple["subject"],
                "predicate" : triple["predicate"],
                "object" : triple["object"]})
        capsule["event_details"] = event_details_list
    return capsule


## One event identified by phrase and time per conversation approach
# ## This variant taks as paramter a dict with phrases and event identifiers.
# If the activity phrase is similar to an event in dict in the dict,
# and there is not time clash
# the identifier is re-used
def get_capsule_with_event_details_from_turn_with_conversational_context_similarity_match_and_time (conversational_context: {}, turn_data, emotion_detector):
    turn = turn_data['Input']
    event_data_list = turn_data['Output']
    chat_id = turn_data['chat']
    chat_date = parser.parse(turn_data['date'])
    turn_id = turn['turn']
    for event_data in event_data_list:
        ### We use a random digit to make the event reference unique
        ### This random digit is combined with the activity expression to identify the event (activity or condition)
        ### We first check the conversational context if such a phrase was already mentioned.
        ### If so, we re-use the ID.
        ### @TODO add variants to the conversational context and a similarity function.
        event_id = random.random()
        print('event_data', event_data, type(event_data))
        utterance_timestamp = datetime.combine(chat_date, datetime.now().time())
        subject_phrase = event_data['activity']
        if subject_phrase in conversational_context:
            event_id = conversational_context[subject_phrase]
        else:
            conversational_context[subject_phrase] = event_id
        triples = get_triples_with_types(event_data, event_id, chat_date)
        offset = "0-"+str(len(turn["utterance"]))
        perspective_value =get_utterance_perspective(turn["utterance"], emotion_detector)
        capsule = { "chat": chat_id,
            "turn": turn_id,
            "author": {"label":turn['speaker'], "type": ["agent"], "uri":"http://cltl.nl/leolani/friends/"+turn['speaker']},
            "utterance": turn["utterance"],
            "utterance_type": UtteranceType.STATEMENT,
            "position": offset,
            "perspective":  perspective_value,
             "timestamp": utterance_timestamp,
             "context_id": event_id
        }
        event_details_list = []
        for triple in triples:
            event_details_list.append({
                "subject" : triple["subject"],
                "predicate" : triple["predicate"],
                "object" : triple["object"]})
        capsule["event_details"] = event_details_list
    return capsule


## Activity-id-based approach for the current SRL output (data/event_srl.json.zip): each
## Output entry's activity already carries its own activity_id, assigned once per real-world
## activity and reused on every later mention -- including a bare reference entry with no
## phrase of its own. That makes activity_id itself a stable, ready-made coreference key, so
## there is no conversational_context dict to thread across turns: the subject URI is built
## directly from activity_id (see get_triples_with_types_and_activity_id), and two mentions of
## the same activity_id -- in this turn or any other -- automatically resolve to the same URI.
##
# cltl.brain's own Certainty enum only has CERTAIN/PROBABLE/POSSIBLE/UNDERSPECIFIED (see
# cltl.commons.discrete.Certainty), so our three-way certainty/uncertain/neutral scale is
# remapped onto its closest equivalent: certain stays certain, neutral (no stated opinion
# either way) becomes probable, and uncertain becomes possible.
CERTAINTY_TO_BRAIN = {"certain": "certain", "neutral": "probable", "uncertain": "possible"}

# cltl.brain's Perspective has no "factuality" field at all -- the closest existing field is
# "polarity" (Polarity.POSITIVE/NEGATIVE/EXPECT/UNDERSPECIFIED), whose docstring describes it as
# "the main flag to signal negation", i.e. the same confirm/deny axis factuality captures.
# Polarity.EXPECT covers our third factuality value directly, so all three map one-to-one.
FACTUALITY_TO_POLARITY = {"confirm": "positive", "deny": "negative", "expect": "expect"}


def _prepare_perspective_for_brain(perspective):
    """Adapt our {emotion, factuality, certainty} perspective dict so it survives
    cltl.brain.infrastructure.rdf_builder.fill_perspective() unharmed and maps onto its
    vocabulary as closely as our schema allows.

    - emotion: fill_perspective reads "emotion" for BOTH Ekman's 6 basic emotions and the 28
      GoEmotion labels from the exact same dict key. If "emotion" is a bare string that also
      happens to be a valid Ekman emotion name -- this includes "neutral" (our default) plus
      "anger"/"disgust"/"fear"/"joy"/"sadness"/"surprise" -- fill_perspective's own fallback
      logic picks that single Ekman Emotion value instead of the GoEmotion list, and
      cltl.brain.LTM_shared._create_attribution then crashes trying to iterate over it
      ("TypeError: 'Emotion' object is not iterable"). Wrapping "emotion" in a list sidesteps
      this: Emotion.as_enum() only ever matches a bare string, so a list always falls through
      to GoEmotion.as_enum(), which returns a (correctly iterable) GoEmotion list.
    - certainty: remapped via CERTAINTY_TO_BRAIN so "neutral"/"uncertain" land on a real
      Certainty value (probable/possible) instead of silently collapsing to UNDERSPECIFIED.
    - factuality: has no direct home in cltl.brain's Perspective, so it's translated into
      "polarity" via FACTUALITY_TO_POLARITY (confirm/deny/expect -> positive/negative/expect,
      one-to-one since Polarity gained an EXPECT member) and dropped from the dict afterwards
      -- fill_perspective never reads "factuality" itself, so leaving it in place would just
      mean it's silently ignored.
    """
    if not perspective:
        return perspective
    fixed = dict(perspective)

    emotion = fixed.get('emotion')
    if emotion is not None and not isinstance(emotion, list):
        fixed['emotion'] = [emotion]

    certainty = fixed.get('certainty')
    if certainty in CERTAINTY_TO_BRAIN:
        fixed['certainty'] = CERTAINTY_TO_BRAIN[certainty]

    factuality = fixed.pop('factuality', None)
    polarity = FACTUALITY_TO_POLARITY.get(factuality)
    if polarity is not None:
        fixed['polarity'] = polarity

    return fixed


## Perspective is taken directly from the entry's own "perspective" field (emotion, factuality,
## certainty -- already annotated by the SRL extraction) instead of being recomputed from the
## utterance text via get_utterance_perspective/emotion_detector: it's the same annotation the
## rest of the entry's triples come from, and there is no emotion_detector parameter here
## because nothing in this function calls into it anymore.
def get_capsule_with_event_details_from_turn_with_activity_id (turn_data):
    turn = turn_data['Input']
    event_data_list = turn_data['Output']
    chat_id = turn_data['chat']
    chat_date = parser.parse(turn_data['date'])
    turn_id = turn['turn']
    # The two-party conversation's other identity, for resolving "you"/"yours"/"yourself" role
    # fillers in this turn's utterance (see _pronoun_identity()): if the human is speaking, the
    # other party is "agent"; if "agent" is speaking, the other party is the human.
    speaker = turn['speaker']
    other = turn_data['human'] if speaker == "agent" else "agent"
    capsules = []
    for event_data in event_data_list:
        activity = event_data.get('activity') or {}
        activity_id = activity.get('activity_id')
        if not activity_id:
            continue
        triples = get_triples_with_types_and_activity_id(event_data, chat_date, speaker=speaker, other=other)
        offset = "0-"+str(len(turn["utterance"]))
        perspective_value = _prepare_perspective_for_brain(event_data.get('perspective'))
        capsule = { "chat": chat_id,
            "turn": turn_id,
            "author": {"label":turn['speaker'], "type": ["agent"], "uri":"http://cltl.nl/leolani/friends/"+turn['speaker']},
            "utterance": turn["utterance"],
            "utterance_type": UtteranceType.STATEMENT,
            "position": offset,
            "perspective":  perspective_value,
             "timestamp": datetime.combine(chat_date, datetime.now().time()),
             "context_id": activity_id
        }
        event_details_list = []
        for triple in triples:
            event_details_list.append({
                "subject" : triple["subject"],
                "predicate" : triple["predicate"],
                "object" : triple["object"]})
        capsule["event_details"] = event_details_list
        capsules.append(capsule)
    return capsules

def get_triples_from_turn(turn_data):
    event_data = turn_data['Output']
    triples = None
    if event_data:
        event_id = random.random()
        triples = get_triples(event_data, event_id)
    return triples

# TRIPLE EXAMPLE WITH EVENT DETAILS
# {"chat": 12,
#  "turn": 1,
#  "author": {"label": "piek", "type": ["person"], 'uri': "http://cltl.nl/leolani/friends/piek-1"},
#  "utterance": "John drinks beer",
#  "utterance_type": UtteranceType.STATEMENT,
#  "position": "0-15",
#  "event_details": [{
#      "subject": {"label": "drink", "type": ["event"],
#                  "uri": "http://cltl.nl/leolani/world/drink-2"
#                  },
#      "predicate": {"label": "hasActor", "uri": "sem:hasActor"},
#      "object": {"label": "john", "type": ["person"],
#                 "uri": "http://cltl.nl/leolani/world/john-1"
#                 }
#
#  },
#      {
#          "subject": {"label": "drink", "type": ["event"],
#                      "uri": "http://cltl.nl/leolani/world/drink-2"},
#          "predicate": {"label": "hasActor", "uri": "sem:hasActor"},
#          "object": {"label": "beer", "type": ["drink"],
#                     "uri": "http://cltl.nl/leolani/world/beer"}
#      },
#      {
#          "subject": {"label": "drink", "type": ["event"],
#                      "uri": "http://cltl.nl/leolani/world/drink-2"},
#          "predicate": {"label": "hasPlace", "uri": "sem:hasPlace"},
#          "object": {"label": "pub", "type": ["place"],
#                     "uri": "http://cltl.nl/leolani/world/pub"}
#      },
#      {
#          "subject": {"label": "drink", "type": ["event"],
#                      "uri": "http://cltl.nl/leolani/world/drink-2"},
#          "predicate": {"label": "hasTime", "uri": "sem:hasTime"},
#          "object": {"label": "8-3-2026", "type": ["date"],
#                     "uri": date_iri}
#      }
#  ],
#  "perspective": {
#      "certainty": 1,
#      "polarity": 1,
#      "sentiment": 1
#  },
#  "timestamp": datetime.combine(start_date, datetime.now().time()),
#  "context_id": context_id
#  }

# EXAMPLE OF THE SRL OUTPUT WHICH IS THE INPUT FOR CREATING CAPSULES
# conversation turn:

        # {
        #     "chat": 255,
        #     "date": "2014,Feb,04",
        #     "human": "Mehmet",
        #     "Input": {
        #         "turn": 6,
        #         "speaker": "Mehmet",
        #         "utterance": "I\u2019ll give that a try. I've also noticed that my afternoon snacks are sometimes not very healthy."
        #     },
        #     "Output": {
        #         "activity": "Provides suggestions on managing blood sugar, portion control, and importance of Mediterranean diet.",
        #         "agent": [
        #             "doctor",
        #             "health advisor"
        #         ],
        #         "patient": [
        #             "Mehmet"
        #         ],
        #         "instrument": [
        #             "medication management",
        #             "dietary suggestions"
        #         ],
        #         "manner": [
        #             "supportive",
        #             "encouraging"
        #         ],
        #         "location": [],
        #         "time": "during a medical consultation"
        #     }
        # }

# CLASSICAL EXAMPLES OF TRIPLES

# capsule:
# {  # CARL SAYS CANNOT SEE HIS PILLS
#         "chat": 1,
#         "turn": 1,
#         "author": {"label": "carl", "type": ["person"], 'uri': "http://cltl.nl/leolani/friends/carl-1"},
#         "utterance": "I need to take my pills, but I cannot find them.",
#         "utterance_type": UtteranceType.STATEMENT,
#         "position": "0-25",
#         "subject": {"label": "carl", "type": ["person"], 'uri': "http://cltl.nl/leolani/world/carl-1"},
#         "predicate": {"label": "see", "uri": "http://cltl.nl/leolani/n2mu/see"},
#         "object": {"label": "pills", "type": ["object", "medicine"], 'uri': "http://cltl.nl/leolani/world/pills-1"},
#         "perspective": {"certainty": 1, "polarity": -1, "sentiment": -1},
#         "timestamp": datetime.combine(start_date, datetime.now().time()),
#         "context_id": context_id
#     }


# event_capsule:
# {  # CARL SAYS CANNOT SEE HIS PILLS
#         "chat": 1,
#         "turn": 1,
#         "author": {"label": "carl", "type": ["person"], 'uri': "http://cltl.nl/leolani/friends/carl-1"},
#         "utterance": "I need to take my pills, but I cannot find them.",
#         "utterance_type": UtteranceType.STATEMENT,
#         "position": "0-25",
#         "event_details: {
#         "subject": {"label": "see", "type": ["person"], 'uri': "http://cltl.nl/leolani/world/see-1"},
#         "predicate": {"label": "hasActor", "uri": "sem:hasActor"},
#         "object": {"label": "pills", "type": ["object", "medicine"], 'uri': "http://cltl.nl/leolani/world/pills-1"}
#          },
#         "perspective": {"certainty": 1, "polarity": -1, "sentiment": -1},
#         "timestamp": datetime.combine(start_date, datetime.now().time()),
#         "context_id": context_id
#     }
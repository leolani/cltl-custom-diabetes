import random

import cltl.gaps_from_kg.thought_util as util
from datetime import datetime
from cltl.brain.long_term_memory import LongTermMemory
from cltl.commons.discrete import Certainty, Polarity, Sentiment, Emotion, GoEmotion


def _parse_event_time(value: str):
    """One event's own time value (a resource URI whose local name is a timestamp, or a plain
    literal) turned into a datetime, or None if it isn't a parseable date/time at all. Handles
    both the real KG's own ISO 8601 "YYYY-MM-DDTHH:MM:SS" local name (the standard "T" date/time
    separator -- what n2mu:time/dateTime etc. actually resolve to once n2mu_sem_roles.py's
    rdfs:subPropertyOf mapping lets get_sem_relation_query() see them at all) and the older,
    underscore-separated "YYYY-MM-DD_HH:MM:SS" convention some of this project's own synthetic
    test-data generators (see thought_util.get_event_gaps_for_period()) still produce.

    Returns None (rather than raising) when neither format matches -- events_from_chat's own SRL
    extractor stores whatever "time" a turn actually mentioned under this same n2mu:time/*
    predicate group whether or not it was ever resolved to a real calendar date, so a value like
    "for an hour", "yesterday" or "in the morning" (turned into "for_an_hour" etc. by the same
    space-to-underscore handling every other label-derived URI gets -- see
    notebooks/chat_sessions.DEFAULT_GAP_ACTIVITY_TYPES's own comment on that) is an entirely
    normal, expected value here, not a malformed date to raise over. The caller
    (get_temporal_containers()/get_temporal_container_for_agent()) simply has no date to go on
    from THIS particular time value when this returns None -- get_temporal_containers() then
    falls back to the CONVERSATION's own date (see its own comment, right after the sem-relation
    loop) before giving up and landing the activity in the "unknown" bucket.
    """
    local_name = value.rsplit("/", 1)[-1]
    try:
        return datetime.fromisoformat(local_name)
    except ValueError:
        pass
    try:
        return datetime.strptime(local_name, "%Y-%m-%d_%H:%M:%S")
    except ValueError:
        return None


def get_last_conversation_date (target:str, brain:LongTermMemory, current_date:datetime, fixed_previous_date:datetime):
    #### We get all utterances to get the date of the previous encounter
    query = util.get_all_utterances(target)
    brain_response = brain._submit_query(query)
    previous_conversation_date = None
    for result in brain_response:
        # print(result)
        year = ""
        month = ""
        day = ""
        if 'year' in result:
            year = result['year']['value']
        if 'month' in result:
            month = result['month']['value']
        if 'day' in result:
            day = result['day']['value']
        new_date = datetime(int(year) ,int(month), int(day))
        if previous_conversation_date:
            if new_date >previous_conversation_date:
                previous_conversation_date = new_date
        else:
            previous_conversation_date = new_date
    if not previous_conversation_date:
        previous_conversation_date = fixed_previous_date
    print('Our previous conservation was on', previous_conversation_date.strftime('%A'), previous_conversation_date)
    print("Today is", current_date.strftime('%A'), current_date)
    duration = current_date -previous_conversation_date
    print('What happened in the last', duration.days, 'days?')
    return previous_conversation_date

## This functions gets all activities from the eKG (brain) and creates a dict for important properties
## It returns 4 different lists as the temporal containers: history, gap, future and unknown time.
## the temporal containers are defined using the current_date and the date of the most recent conversation.
def get_temporal_containers (brain:LongTermMemory, current_date:datetime, recent_date:datetime, activity_type:str ="n2mu:icf", label:str=None):
    ##### We get all activities  and divide them into H, G, and F given now and the previous encounter
    history = []
    gap = []
    unknown = []
    future = []
    query = None
    if activity_type and label is None:
        # activity_type need to be full uri's prefixed with a namespace.
        # the standard namespace for activities is "n2mu", e.g. "n2my:icf"
        query = util.get_all_instances_query(activity_type)
    else:
        query = util.get_all_label_matches_query(label)
    print(query)
    brain_response = brain._submit_query(query)
    print('I found', len(brain_response), 'activities')
    #print(brain_response)
    for activity in brain_response:
        event_date = None
        event_location = None
        event_actors = []
        activity_id = activity["id"]["value"]
        if "label" in activity:
            activity_label= activity["label"]["value"]
        else:
            activity_label = label
        #### Get SEM relations
        query = util.get_sem_relation_query(activity_id)
        sem_response = brain._submit_query(query)
        for sem in sem_response:
            if 'actor_id' in sem:
                actor = sem['actor_id']['value']
                actor = actor[actor.rindex("/")+1:]
                if not actor in event_actors:
                    event_actors.append(actor)
            if 'place_id' in sem:
                event_location = sem['place_id']['value']
                event_location = event_location[event_location.rindex("/")+1:]
            if 'time_id' in sem:
                parsed_time = _parse_event_time(sem['time_id']['value'])
                if parsed_time is not None:
                    event_date = parsed_time

        #### Fall back to the CONVERSATION's own date (when this activity was actually said,
        #### via gaf:denotedIn -> sem:hasBeginTimeStamp) whenever none of the activity's own time
        #### values resolved to a real date above -- most of the time this means the human used a
        #### vague or relative phrase ("for an hour", "recently", "yesterday") that
        #### _parse_event_time() correctly can't turn into a calendar date on its own. Without
        #### this, an activity like that would land in the "unknown" bucket forever, even though
        #### its own utterance date is perfectly well known -- "said about it during the
        #### conversation" is the best date to attribute it to short of actually understanding the
        #### phrase itself.
        if not event_date:
            time_query = util.get_utterance_time_query(activity_id)
            time_response = brain._submit_query(time_query)
            if time_response:
                event_date = _parse_event_time(time_response[0]['time_id']['value'])

        #### Get perspectives
        emotion = GoEmotion.NEUTRAL
        certainty = Certainty.UNDERSPECIFIED
        sentiment = Sentiment.UNDERSPECIFIED
        polarity = Polarity.UNDERSPECIFIED
        query = util.get_perspective_query(activity_id)
        perspective_response = brain._submit_query(query)
        if perspective_response:
            for p in perspective_response:
                perspective = p['perspective_value']['value']
                attribute = perspective[perspective.rindex("/")+1:]
                value = perspective[perspective.rindex("#")+1:]
                if attribute.startswith("factuality"):
                    polarity = value
                elif attribute.startswith("sentiment"):
                    sentiment = Sentiment.from_str(value).value
                elif attribute.startswith("certainty"):
                    certainty = value
                # elif attribute.startswith("emotion"):
                #         emotion = GoEmotion.as_enum(value)

        ## Creating the data structure for each activity
        activity_result = {'id':activity_id, 'label':activity_label, "actors":event_actors, "location":event_location, "time": event_date,
                           #"perspective": len(event_perspectives)
                           'emotion': emotion,
                           'certainty': certainty,
                           'sentiment': sentiment,
                           'polarity': polarity
                           }

        ## Saving the activity in different temporal containers
        if not event_date:
            unknown.append(activity_result)
        elif event_date<recent_date:
            history.append(activity_result)
        elif event_date>current_date:
            future.append(activity_result)
        else:
            gap.append(activity_result)

    return history, gap, future, unknown


def get_label_dictionary(brain:LongTermMemory, brain_response):
    label_dictionary = {}
    for activity in brain_response:
        label = ""
        event_patients = []
        activity_id = activity["id"]["value"]
        query = util.get_role_relation_query(activity_id)
        # print(query)
        srl_response = brain._submit_query(query)
        for srl in srl_response:
            if 'patient_id' in srl:
                actor = srl['patient_id']['value']
                actor = actor[actor.rindex("/") + 1:].lower()
                if not actor in event_patients:
                    event_patients.append(actor)
        if len(event_patients) > 0:
            for patient in event_patients:
                sublabels = label.split("_")
                for sublabel in sublabels:
                    if sublabel in label_dictionary:
                        label_dictionary[sublabel] +=1
                    else:
                        label_dictionary[sublabel] = 1
    return label_dictionary


## This functions gets all activities involving a specific agent from the eKG (brain) and creates a dict for important properties
## It returns a single temporal containers.
def get_temporal_container_for_agent (brain:LongTermMemory, agent:"carl", activity_type:str ="n2mu:activity", label:str=None):
    ##### We get all activities  and divide them into H, G, and F given now and the previous encounter
    history = []
    stopwords = ["levels", "jan's", "issues"]
    query = None
    if activity_type and label is None:
        # activity_type need to be full uri's prefixed with a namespace.
        # the standard namespace for activities is "n2mu", e.g. "n2mu:icf"
        query = util.get_all_instances_query(activity_type)
    else:
        query = util.get_all_label_matches_query(label)
    print(query)
    brain_response = brain._submit_query(query)
    print('I found', len(brain_response), 'activities')
    #print(brain_response)
    for activity in brain_response:
        event_date = None
        event_location = None
        event_agents = []
        event_patients = []
        activity_id = activity["id"]["value"]
        if "label" in activity:
            activity_label= activity["label"]["value"]
        else:
            activity_label = label
        #### Get SRL relations
        query = util.get_role_relation_query(activity_id)
        #print(query)
        srl_response = brain._submit_query(query)
        for srl in srl_response:
            if 'agent_id' in srl:
                actor = srl['agent_id']['value']
                actor = actor[actor.rindex("/")+1:]
                if not actor in event_agents:
                    event_agents.append(actor)
            if 'patient_id' in srl:
                actor = srl['patient_id']['value']
                actor = actor[actor.rindex("/") + 1:].lower()
                if not actor in event_patients:
                    event_patients.append(actor)
            # if 'instrument_id' in srl:
            #     actor = srl['instrument_id']['value']
            #     actor = actor[actor.rindex("/") + 1:]
            #     if not actor in event_actors:
            #         event_actors.append(actor)
            # if 'manner_id' in srl:
            #     actor = srl['manner_id']['value']
            #     actor = actor[actor.rindex("/") + 1:]
            #     if not actor in event_actors:
            #         event_actors.append(actor)
            if 'place_id' in srl:
                event_location = srl['place_id']['value']
                event_location = event_location[event_location.rindex("/")+1:]
            if 'time_id' in srl:
                parsed_time = _parse_event_time(srl['time_id']['value'])
                if parsed_time is not None:
                    event_date = parsed_time

        #### Get perspectives
        emotion = GoEmotion.NEUTRAL
        certainty = Certainty.UNDERSPECIFIED
        sentiment = Sentiment.UNDERSPECIFIED
        polarity = Polarity.UNDERSPECIFIED
        query = util.get_perspective_query(activity_id)
        perspective_response = brain._submit_query(query)
        if perspective_response:
            for p in perspective_response:
                perspective = p['perspective_value']['value']
                attribute = perspective[perspective.rindex("/")+1:]
                value = perspective[perspective.rindex("#")+1:]
                if attribute.startswith("factuality"):
                    polarity = value
                elif attribute.startswith("sentiment"):
                    sentiment = Sentiment.from_str(value).value
                elif attribute.startswith("certainty"):
                    certainty = value
                # elif attribute.startswith("emotion"):
                #         emotion = GoEmotion.as_enum(value)
        sentiment = random.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        certainty = random.choice([5, 10, 15, 20, 25, 30, 35, 40])

        ## Creating the data structure for each activity
        label = ""
        if len(event_patients)>0:
            for patient in event_patients:
                label += patient+";"
        else:
            label = activity_label
        for stop in stopwords:
            label = label.replace(stop, "")
        label = label.strip(';')
        label = label.strip('_')
        activity_result = {'id':activity_id, 'activity':activity_label, "label": label, "agents":event_agents, "patients": event_patients, "location":event_location, "time": event_date,
                           #"perspective": len(event_perspectives)
                           'emotion': emotion,
                           'certainty': certainty,
                           'sentiment': sentiment,
                           'polarity': polarity
                           }

        ## Saving the activity in different temporal containers
        if  event_date and agent in event_agents:
            history.append(activity_result)


    return history

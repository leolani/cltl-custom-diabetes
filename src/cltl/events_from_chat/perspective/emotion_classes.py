import dataclasses
from enum import Enum, auto
from typing import Optional, Any, List, Union

@dataclasses.dataclass
class EmotionType(Enum):
    GO = auto()
    EKMAN = auto()
    EMOTIC = auto()
    SENTIMENT = auto()


class Sentiment(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()
    NEUTRAL = auto()


class EkmanEmotion(Enum):
    ANGER = auto()
    DISGUST = auto()
    FEAR = auto()
    JOY = auto()
    SADNESS = auto()
    SURPRISE = auto()
    NEUTRAL = auto()


class GoEmotion(Enum):
    AMUSEMENT = auto()
    EXCITEMENT = auto()
    JOY = auto()
    LOVE = auto()
    DESIRE = auto()
    OPTIMISM = auto()
    CARING = auto()
    PRIDE = auto()
    ADMIRATION = auto()
    GRATITUDE = auto()
    RELIEF = auto()
    APPROVAL = auto()
    FEAR = auto()
    NERVOUSNESS = auto()
    REMORSE = auto()
    EMBARRASSMENT = auto()
    DISAPPOINTMENT = auto()
    SADNESS = auto()
    GRIEF = auto()
    DISGUST = auto()
    ANGER = auto()
    ANNOYANCE = auto()
    DISAPPROVAL = auto()
    REALIZATION = auto()
    SURPRISE = auto()
    CURIOSITY = auto()
    CONFUSION = auto()
    NEUTRAL = auto()


class EmoticEmotion(Enum):
    """ 26 Emotions use in
    Kosti R., J.M. Alvarex, A. Recasens, and A. Paedriza, (2019), "Context based emotion recognition using emotic dataset",
    IEEE Transactions on patterns analysis and machine intelligence.

    Ordered as used in the Emotic model.
    """
    # KEEP ORDER!
    AFFECTION = 1
    ANGER = 2
    ANNOYANCE = 3
    ANTICIPATION = 4
    AVERSION = 5
    CONFIDENCE = 6
    DISAPPROVAL = 7
    DISCONNECTION = 8
    DISQUIETMENT = 9
    DOUBT_CONFUSION = 10
    EMBARRASSMENT = 11
    ENGAGEMENT = 12
    ESTEEM = 13
    EXCITEMENT = 14
    FATIGUE = 15
    FEAR = 16
    HAPPINESS = 17
    PAIN = 18
    PEACE = 19
    PLEASURE = 20
    SADNESS = 21
    SENSITIVITY = 22
    SUFFERING = 23
    SURPRISE = 24
    SYMPATHY = 25
    YEARNING = 26


class VADEmotion(Enum):
    """ VAD Emotions use in
        Kosti R., J.M. Alvarex, A. Recasens, and A. Paedriza, (2019), "Context based emotion recognition using emotic dataset",
        IEEE Transactions on patterns analysis and machine intelligence.

    Ordered as used in the Emotic model.
    """
    # KEEP ORDER!
    VALENCE = 1
    AROUSAL = 2
    DOMINANCE = 3


# Use a mapping to get a dictionary of the mapped GO_emotion scores
def get_mapped_scores(emotion_map, go_emotion_scores):
    mapped_scores = {}

    for prediction in go_emotion_scores:
        go_emotion = prediction['label']
        for key in emotion_map:
            if go_emotion in emotion_map[key]:
                if not key in mapped_scores:
                    mapped_scores[key] = [prediction['score']]
                else:
                    mapped_scores[key].append(prediction['score'])
    return mapped_scores


# Get the averaged score for an emotion or sentiment from the GO_emotion scores mapped according to the emotion_map
def get_total_mapped_scores(emotion_map, go_emotion_scores):
    total_mapped_scores = []
    mapped_scores = get_mapped_scores(emotion_map, go_emotion_scores)
    for emotion in mapped_scores:
        lst = mapped_scores[emotion]
        total_score = sum(lst)
        total_mapped_scores.append({'label':emotion, 'score':total_score})
    return sort_predictions(total_mapped_scores)


# TODO use enums here to avoid duplication
# MAP EKMAN TO SENTIMENT
ekman_sentiment_map={
    "positive": ["joy", "positive"],
    "negative": ["anger", "disgust", "fear", "sadness", "negative"],
    "neutral": ["neutral", "surprise", "ambiguous"]
}


# Mapping GO_Emotions to sentiment values
go_sentiment_map={
    "positive": ["curiosity", "amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
    "negative": ["fear",  "confusion", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
    "neutral": ["realization", "surprise", "neutral"]
}


# Mapping GO_Emotions to Ekman values
go_ekman_map={
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness", "confusion"],
    "joy": ["joy", "curiosity", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
    "surprise": ["surprise", "realization"],
    "neutral": ["neutral"]
}

face_ekman_map={
    "joy" : ['affection', 'confidence', 'esteem', 'excitement', 'happiness',  'peace', 'pleasure',  'sympathy'],
    "anger": ['anger', 'annoyance', 'embarrassment'],
    "fear": ['doubt_confusion', 'fear', 'pain',  'suffering', 'yearning'],
    "disgust": ['aversion', 'disapproval', 'disconnection', 'embarrassment'],
    "sadness": ['disconnection', 'disquietment', 'fatigue', 'sadness'],
    "surprise": ['engagement', 'excitement', 'sensitivity', 'surprise'],
    "neutral": ['anticipation', 'peace']
}

face_sentiment_map={
    "positive" : ['affection', 'confidence', 'esteem', 'excitement', 'happiness',  'peace', 'pleasure',  'sympathy', 'engagement', 'excitement', 'sensitivity', 'surprise'],
    "negative": ['anger', 'annoyance', 'embarrassment', 'doubt_confusion', 'fear', 'pain',  'suffering', 'yearning', 'aversion', 'disapproval', 'disconnection', 'embarrassment', 'disconnection', 'disquietment', 'fatigue', 'sadness'],
    "neutral": ['anticipation', 'peace']
}

# Sort a list of results in JSON format by the value of the score element
def sort_predictions(predictions):
    return sorted(predictions, key=lambda x: x['score'], reverse=True)


class Emotion:
    """
    Information about an Emotion.
    "Emotion type"
    "Emotion value"
    "confidence"
    "source"
    """
    # TODO Collection[EmotionType]?
    type: EmotionType
    value: str
    confidence: Optional[float]

    def __init__(self, type: EmotionType, value: str, confidence: Optional[float] = None):
        self.type = type
        self.value = value
        self.confidence = confidence

    def to_enum(self) -> Union[GoEmotion, EkmanEmotion, EmoticEmotion, Sentiment]:
        if self.type == EmotionType.GO:
            return GoEmotion[self.value.upper()]
        if self.type == EmotionType.EKMAN:
            return EkmanEmotion[self.value.upper()]
        if self.type == EmotionType.EMOTIC:
            return EmoticEmotion[self.value.upper()]
        if self.type == EmotionType.SENTIMENT:
            return Sentiment[self.value.upper()]

        raise ValueError("Unknown type: " + self.type)

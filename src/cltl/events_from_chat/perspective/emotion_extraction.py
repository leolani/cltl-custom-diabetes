import time
import perspective.emotion_classes as emo
from typing import Optional, Any, List, Union

from transformers import pipeline

_MODEL_NAME = "bhadresh-savani/bert-base-go-emotion"
_THRESHOLD = 0.6

# GO Emotions is a finetuned BERT transformer with the GO Emotion data
#https://github.com/google-research/google-research/tree/master/goemotions
#https://github.com/google-research/google-research/blob/master/goemotions/goemotions_model_card.pdf


class GoEmotionDetector():
    def __init__(self, model: str):
        self.emotion_pipeline = pipeline('sentiment-analysis',  model=model, top_k=None)

    def extract_audio_emotions(self, audio_signal: Any) -> List[emo.Emotion]:
        raise NotImplementedError()

    def extract_text_emotions(self, utterance: str, threshold:0.6) -> List[emo.Emotion]:
        if not utterance:
            return []
        _THRESHOLD = threshold
        response = self.emotion_pipeline(utterance)
        emotion_labels = emo.sort_predictions(response[0])
        go_labels = self._filter_by_threshold(emo.EmotionType.GO, emotion_labels)

        ekman_labels = emo.get_total_mapped_scores(emo.go_ekman_map, emotion_labels)
        ekman_labels = self._filter_by_threshold(emo.EmotionType.EKMAN, ekman_labels)

        sentiment_labels = emo.get_total_mapped_scores(emo.go_sentiment_map, emotion_labels)
        sentiment_labels = self._filter_by_threshold(emo.EmotionType.SENTIMENT, sentiment_labels)

        return go_labels, ekman_labels, sentiment_labels

    def _filter_by_threshold(self, emotion_type, results):
        return [emo.Emotion(type=emotion_type, value=result['label'], confidence=result['score'])
                for result in results
                if result['score'] > 0 and result['score'] / results[0]['score'] > _THRESHOLD]




if __name__ == "__main__":
    utterance = "I love cats."
    model_path = "/Users/piek/Desktop/d-Leolani/leolani-models/bert-base-go-emotion"

    model_path = "AnasAlokla/multilingual_go_emotions"
    #  Languages: Arabic, English, French, Spanish, Dutch, Turkish
    analyzer = GoEmotionDetector(model=model_path)
    emotions = analyzer.extract_text_emotions(utterance)
    emotion_json ={}
    for emotion in emotions:
        emotion_json.update({'type': emotion.type, 'value':emotion.value, 'confident': emotion.confidence})
        print(emotion_json)

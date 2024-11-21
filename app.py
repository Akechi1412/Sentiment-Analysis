import traceback
import tensorflow as tf
from fastapi import FastAPI, Response
from pydantic import BaseModel
from utils.data_preprocessing import clean_text, preprocess_texts
from tensorflow.keras import models # type: ignore
from utils.layers import BERTEncoder
from transformers import AutoTokenizer

app = FastAPI()

model_name = 'vinai/phobert-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = models.load_model(
    'models/best_model.h5',
    custom_objects={'BERTEncoder': BERTEncoder})

class PredictionInput(BaseModel):
    comment: str

@app.get('/')
async def welcome():
    return {'message': 'Welcome!'}

@app.post('/predict')
async def predict(input_data: PredictionInput, response: Response):
    try:      
        comment = clean_text(input_data.comment)
        max_length = 50
        input_ids, attention_mask = preprocess_texts(tokenizer, [comment], max_length=max_length)
        pred_prob = model.predict([input_ids, attention_mask])
        pred = tf.argmax(pred_prob, axis=1).numpy()
        pred = pred[0]
        if pred == 0:
            return {
                'sentiment': 'Negative',
                'message': 'Sucessfully!'
            }
        elif pred == 1:
            return {
                'sentiment': 'Neutral',
                'message': 'Sucessfully!'
            }
        elif pred == 2:
            return {
                'sentiment': 'Positive',
                'message': 'Sucessfully!'
            }

    except Exception as e:
        response.status_code = 500
        return {
            'sentiment': 'Unknown',
            'message': f'An unexpected error occurred: {traceback.format_exc()}'
        }
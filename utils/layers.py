from tensorflow.keras import Layer # type: ignore
from transformers import TFAutoModel 

class BERTEncoder(Layer):
    def __init__(self, bert_model_name, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.bert_model_name = bert_model_name
        self.bert_model = TFAutoModel.from_pretrained(bert_model_name, from_pt=True)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def get_config(self):
        config = super().get_config()
        config.update({
            'bert_model_name': self.bert_model_name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        bert_model_name = config.pop('bert_model_name')
        return cls(bert_model_name=bert_model_name)
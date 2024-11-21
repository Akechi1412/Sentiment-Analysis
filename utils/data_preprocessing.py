import re
import string
from underthesea import text_normalize, word_tokenize
import emoji
from transformers import AutoTokenizer

def clean_text(text):
    """
    Preprocesses a single text string by applying a series of normalization steps.
    
    Steps include:
    1. Converting text to lowercase.
    2. Removing emojis, replacing them with spaces.
    3. Reducing repeated characters (e.g., 'aaaa' -> 'aa').
    4. Normalizing punctuation spacing to ensure proper separation.
    5. Stripping leading and trailing spaces.
    6. Removing all punctuation.
    7. Normalizing Vietnamese text for consistent diacritics and formatting.
    8. Tokenizing the text into words for further processing.

    Args:
        text (str): The input text string to preprocess.

    Returns:
        str: The cleaned and tokenized text string.
    """
    # Lowercase text
    text = text.lower()

    # Remove emoji
    text = emoji.replace_emoji(text, replace=' ')

    # Reduce repeated characters (e.g., 'aaaa' -> 'aa')
    text = re.sub(r'(\w)\1+', r'\1\1', text)

    # Normalize punctuation spacing
    text = re.sub(r'(\w)([' + string.punctuation + '])(\w)', r'\1 \2 \3', text)
    text = re.sub(r'([' + string.punctuation + '])([' + string.punctuation + '])+', r'\1', text)
    
    # Remove leading/trailing spaces
    text = text.strip()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Normalize Vietnamese text
    text = text_normalize(text)

    # Tokenize
    text = word_tokenize(text, format='text')

    return text

def preprocess_texts(model_name, texts, max_length=128):
    """
    Tokenizes and formats a list of text strings for input into a Transformer model.

    This function uses a pretrained tokenizer to:
    1. Tokenize the input texts into numerical representations (input IDs).
    2. Pad or truncate the sequences to a specified maximum length.
    3. Generate attention masks to indicate the positions of valid tokens.

    Args:
        model_name (str): The name or path of the pretrained Transformer model.
        texts (list[str]): A list of input text strings to process.
        max_length (int, optional): The maximum sequence length. Defaults to 128.

    Returns:
        tuple: A tuple containing two elements:
            - input_ids (tf.Tensor): The tokenized and padded input IDs for the texts.
            - attention_mask (tf.Tensor): The attention masks corresponding to the input IDs.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_texts = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return encoded_texts['input_ids'], encoded_texts['attention_mask']

if __name__ == '__main__':
    text = 'Chất lượng sản phẩm tuyệt vời,sẽ ủng hộ shop nhiều.';
    print(clean_text(text))
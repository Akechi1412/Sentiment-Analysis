import re
import string
from underthesea import text_normalize, word_tokenize
import emoji

def clean_text(text):
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

if __name__ == '__main__':
    text = 'Chất lượng sản phẩm tuyệt vời,sẽ ủng hộ shop nhiều.';
    print(clean_text(text))
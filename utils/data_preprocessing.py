import unicodedata
import re
from spacy.lang.vi import Vietnamese

def replace_spam_link(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, 'spam_link', text)

def remove_non_sentiment_characters(text):
    pattern = r'[^\w\s\*\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U00002600-\U000026FF]'  # Keep alphanumeric characters, whitespace, *, and emojis
    return re.sub(pattern, '', text)

def remove_extra_whitespace(text):
    return ' '.join(text.split())

def remove_repeated_characters(text):
    return re.sub(r'(\w)(\1+)(\s|$)', r'\1\3', text)

def normalize_abbreviations(word_list):
    abbreviations = {
        'k': 'không',
        'ko': 'không',
        'k0': 'không',
        'kh': 'không',
        'kg': 'không',
        'hk': 'không',
        'bt': 'bình thường',
        'bth': 'bình thường',
        'bthg': 'bình thường',
        'sp': 'sản phẩm',
        'z': 'vậy',
        'df': 'lỗi',
        'ok': 'oke',
        'oki': 'oke',
        'nc': 'nước',
        'nchung': 'nói chung',
        'dc': 'được',
        'đc': 'được',
        'ctrinh': 'chương trình',
        'kmai': 'khuyến mãi',
        'sd': 'sử dụng',
        'nch': 'nói chuyện',
        'dthw': 'dễ thương',
        'nt': 'nhắn tin'
    }
    normalized_word_list = [abbreviations.get(word, word) for word in word_list]

    return normalized_word_list

def word_tokenize(text):
    nlp = Vietnamese()
    doc = nlp(text)
    word_list = []
    for token in doc:
        word_list.append(token.text)
    return word_list

def nomalize_and_slit_text(text):
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = replace_spam_link(text)
    text = remove_non_sentiment_characters(text)
    text = remove_extra_whitespace(text)
    text = remove_repeated_characters(text)
    word_list = word_tokenize(text)
    word_list = normalize_abbreviations(word_list)
    return word_list
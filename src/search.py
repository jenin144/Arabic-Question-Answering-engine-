#search.py
import re
from collections import Counter
from nltk.stem import ISRIStemmer
from src.preprocess import clean_arabic_text, tokenize_arabic_text, remove_stopwords, stem_tokens


def preprocess_query(query):
    """معالجة موحدة لاستعلامات البحث"""
    if not isinstance(query, str) or not query.strip():
        return []

    cleaned_query = clean_arabic_text(query)
    tokens = tokenize_arabic_text(cleaned_query)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)

    # إذا كانت النتيجة قصيرة جداً، استخدم الكلمات الأصلية
    if len(tokens) < 2:
        tokens = tokenize_arabic_text(cleaned_query)
        # إزالة كلمات الاستفهام فقط
        question_words = {'ماذا', 'من', 'أين', 'متى', 'كيف', 'لماذا', 'كم', 'أي', 'هل', 'ما'}
        tokens = [token for token in tokens if token.lower() not in question_words]

    return tokens

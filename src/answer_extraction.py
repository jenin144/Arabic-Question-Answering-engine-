# answer_extraction.py
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.search import preprocess_query, stem_tokens, tokenize_arabic_text


def load_ner_model(model_name="asafaya/bert-base-arabic"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer)



def calculate_ner_tfidf_score(document, query_tokens, ner_pipeline):
    search_pattern = r'\\b(?:' + '|'.join(re.escape(token) for token in query_tokens) + r')\\b'
    sentences = re.split(r'[.؟!\n،]+', document)
    relevant = [s.strip() for s in sentences if re.search(search_pattern, s) and len(s.split()) > 3]
    if not relevant:
        return 0.0
    total_ner, total_sim = 0.0, 0.0
    for sent in relevant:
        entities = ner_pipeline(sent)
        ner_score = sum(1 for e in entities if e['entity'].startswith('B-'))
        vect = TfidfVectorizer()
        vecs = vect.fit_transform([' '.join(query_tokens), sent])
        sim = cosine_similarity(vecs[0:1], vecs[1:]).flatten()[0]
        total_ner += ner_score
        total_sim += sim
    avg_ner = total_ner / len(relevant)
    avg_sim = total_sim / len(relevant)
    return avg_ner + avg_sim



def extract_answers_with_ner_arabic(document, query_tokens, ner_pipeline):


    search_pattern = r'\b(?:' + '|'.join(re.escape(token) for token in query_tokens) + r')\b'
    sentences = re.split(r'[.؟!\n،]+', document)

    relevant_sentences = [s.strip() for s in sentences if re.search(search_pattern, s) and len(s.strip().split()) > 3]

    if not relevant_sentences:
        return document[:300]

    scores = []
    for sentence in relevant_sentences:
        ner_entities = ner_pipeline(sentence)
        ner_score = sum(1 for e in ner_entities if e['entity'].startswith('B-'))

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([' '.join(query_tokens), sentence])
        similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()[0]

        final_score = ner_score + similarity
        scores.append((sentence, final_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s for s, _ in scores[:3]]

    return ' '.join(top_sentences)




def extract_best_answer_traditional_inverted_index(doc_id, query_tokens, inverted_index, df):
    """طريقة تقليدية لاستخراج الإجابة باستخدام inverted index"""
    # استرجاع نص الوثيقة
    doc_text = df[df['QID'] == doc_id]['doc'].values[0]
    sentences = [s.strip() for s in doc_text.split('.') if s.strip()]

    # حساب عدد المطابقات لكل جملة بناءً على الكلمات الموجودة في inverted index
    sentence_scores = {}
    for token in query_tokens:
        if token in inverted_index:
            if doc_id in inverted_index[token]:
                for idx, sentence in enumerate(sentences):
                    stemmed = stem_tokens(tokenize_arabic_text(sentence))
                    if token in stemmed:
                        sentence_scores[idx] = sentence_scores.get(idx, 0) + 1

    if sentence_scores:
        # اختيار الجملة بأكبر عدد من المطابقات
        best_sentence_num = max(sentence_scores, key=sentence_scores.get)
        return sentences[best_sentence_num]

    # fallback
    return doc_text[:300] + "..."

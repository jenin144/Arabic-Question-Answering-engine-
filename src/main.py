import re
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from src.preprocess import load_data, preprocess_documents, create_inverted_index
from src.search import preprocess_query
from src.answer_extraction import load_ner_model, calculate_ner_tfidf_score, extract_answers_with_ner_arabic, extract_best_answer_traditional_inverted_index

# المتغيرات العالمية
bm25_model = None
ner_pipeline = None
df = None
qa_pipeline = None
inverted_index = None

# معاملات الهجينة للـ Hybrid Score
ALPHA = 0.6  # وزن BM25
BETA = 0.4   # وزن NER+TFIDF

# التهيئة مرة واحدة
def initialize_system():
    global bm25_model, df, qa_pipeline, inverted_index, ner_pipeline
    if df is not None:
        return
    
    print("🔄 تحميل البيانات...")
    df = load_data("data/QAfull.csv")
    
    print("🔄 معالجة الوثائق...")
    df = preprocess_documents(df)
    inverted_index = create_inverted_index(df)
    
    print("🔄 إعداد BM25...")
    tokenized = df['Tokenized_Document'].tolist()
    bm25_model = BM25Okapi(tokenized)
    
    # تحميل نموذج QA
    try:
        qa_pipeline = pipeline(
            "question-answering",
            model="deepset/xlm-roberta-large-squad2",
            tokenizer="deepset/xlm-roberta-large-squad2"
        )
        print("✅ نموذج QA جاهز")
    except Exception as e:
        print(f"❌ خطأ في تحميل نموذج QA: {e}")
        qa_pipeline = None
    
    # تحميل نموذج NER
    try:
        ner_pipeline = load_ner_model()
        print("✅ نموذج NER جاهز")
    except Exception as e:
        print(f"❌ خطأ في تحميل نموذج NER: {e}")
        ner_pipeline = None


def extract_with_qa_model(doc_text, query, debug=False):
    """استخراج الإجابة باستخدام نموذج QA"""
    if not qa_pipeline:
        return None, 0.0
    
    try:
        # تقليم النص إذا كان طويلاً جداً
        max_length = 512
        if len(doc_text) > max_length:
            doc_text = doc_text[:max_length]
            
        res = qa_pipeline({"context": doc_text, "question": query})
        return res.get('answer'), res.get('score', 0.0)
    except Exception as e:
        if debug:
            print(f"❌ خطأ في نموذج QA: {e}")
        return None, 0.0


def calculate_hybrid_scores(query_tokens, debug=False):
    """حساب الـ Hybrid Score لجميع المستندات"""
    # حساب درجات BM25 لجميع المستندات
    bm25_scores = bm25_model.get_scores(query_tokens)
    
    # تطبيع درجات BM25 (0-1)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    normalized_bm25 = [score / max_bm25 for score in bm25_scores]
    
    hybrid_scores = []
    
    for idx, bm25_score in enumerate(normalized_bm25):
        doc_text = df.iloc[idx]['doc']
        
        # حساب NER+TFIDF score
        if ner_pipeline:
            ner_tfidf_score = calculate_ner_tfidf_score(doc_text, query_tokens, ner_pipeline)
        else:
            ner_tfidf_score = 0.0
        
        # حساب الـ Hybrid Score
        hybrid_score = ALPHA * bm25_score + BETA * ner_tfidf_score
        
        hybrid_scores.append({
            'index': idx,
            'hybrid_score': hybrid_score,
            'bm25_score': bm25_score,
            'ner_tfidf_score': ner_tfidf_score
        })
        
        if debug and idx < 5:  # طباعة أول 5 للتصحيح
            print(f"Doc {idx}: BM25={bm25_score:.3f}, NER+TFIDF={ner_tfidf_score:.3f}, Hybrid={hybrid_score:.3f}")
    
    # ترتيب حسب الـ Hybrid Score
    hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    return hybrid_scores


def process_query_enhanced(query, debug=False):
    """المعالجة النهائية للاستعلام باستخدام Hybrid Ranking"""
    initialize_system()
    
    if debug:
        print(f"\n🔍 الاستعلام: {query}")
        print(f"⚖️ معاملات الهجينة: α={ALPHA}, β={BETA}")
    
    # معالجة الاستعلام
    tokens = preprocess_query(query)
    if not tokens:
        return {'query': query, 'top_answers': [], 'error': 'لا أفهم السؤال'}
    
    if debug:
        print(f"🔤 الكلمات المفتاحية: {tokens}")
    
    # حساب الـ Hybrid Scores وترتيب المستندات
    ranked_docs = calculate_hybrid_scores(tokens, debug)
    
    answers = []
    
    # معالجة أفضل 5 مستندات حسب الـ Hybrid Score
    for rank, doc_info in enumerate(ranked_docs[:5], start=1):
        idx = doc_info['index']
        doc = df.iloc[idx]
        doc_id, doc_text, url = doc['QID'], doc['doc'], doc.get('url', '')
        
        # توزيع طرق الاستخراج حسب الترتيب
        if rank == 1:
            # المرتبة الأولى: QA Model
            ans, conf = extract_with_qa_model(doc_text, query, debug)
            method = 'qa_model'
            if debug:
                print(f"🥇 المرتبة {rank}: QA Model - الثقة: {conf:.3f}")
                
        elif rank == 2:
            # المرتبة الثانية: NER+TFIDF
            ans = extract_answers_with_ner_arabic(doc_text, tokens, ner_pipeline)
            method = 'ner_tfidf'

            if debug:
                print(f"🥈 المرتبة {rank}: NER+TFIDF")
                
        else:
            # المراتب الأخرى: Traditional
            ans = extract_best_answer_traditional_inverted_index(doc_id, tokens, inverted_index, df)
            conf = 0.0
            method = 'traditional'
            if debug:
                print(f"🥉 المرتبة {rank}: Traditional")
        
        # التحقق من جودة الإجابة
        if ans and len(ans.strip()) >= 20:
            answers.append({
                'rank': rank,
                'answer': ans.strip(),
                'document_id': doc_id,
                'hybrid_score': doc_info['hybrid_score'],
                'bm25_score': doc_info['bm25_score'],
                'ner_tfidf_score': doc_info['ner_tfidf_score'],
                'qa_confidence': conf,
                'extraction_method': method,
                'source_link': url
            })
            
            if debug:
                print(f"   ✅ إجابة مقبولة: {ans[:100]}...")
        else:
            if debug:
                print(f"   ❌ إجابة مرفوضة (قصيرة جداً)")
        
        # إيقاف عند العثور على 3 إجابات جيدة
        if len(answers) >= 3:
            break
    
    result = {
        'query': query,
        'top_answers': answers,
        'total_documents': len(df),
        'hybrid_params': {'alpha': ALPHA, 'beta': BETA},
        'error': None if answers else 'لم يتم العثور على إجابة مناسبة'
    }
    
    if debug:
        print(f"\n📊 النتائج: {len(answers)} إجابة من أصل {len(ranked_docs)} مستند")
    
    return result


def main_process_query(query, debug=False):
    """دالة رئيسية لمعالجة الاستعلام"""
    return process_query_enhanced(query, debug=debug)


def set_hybrid_weights(alpha, beta):
    """تعديل أوزان الهجينة"""
    global ALPHA, BETA
    if alpha + beta != 1.0:
        # تطبيع الأوزان
        total = alpha + beta
        alpha = alpha / total
        beta = beta / total
    
    ALPHA = alpha
    BETA = beta
    print(f"⚖️ تم تحديث أوزان الهجينة: α={ALPHA:.2f}, β={BETA:.2f}")


def main_process_query(query, debug=False):
    return process_query_enhanced(query, debug=debug)


# اختبار سريع
if __name__ == '__main__':
    print("🚀 بدء الاختبار...")
    
    # اختبار مع درجة تصحيح الأخطاء
    result = process_query_enhanced("ما أعراض السكري؟", debug=True)
    
    print(f"\n📋 النتائج النهائية:")
    print(f"الاستعلام: {result['query']}")
    print(f"عدد الإجابات: {len(result['top_answers'])}")
    
    for a in result['top_answers']:
        print(f"\n🔹 إجابة {a['rank']}:")
        print(f"   النص: {a['answer'][:150]}...")
        print(f"   الطريقة: {a['extraction_method']}")
        print(f"   الدرجات: Hybrid={a['hybrid_score']:.3f}, BM25={a['bm25_score']:.3f}, NER+TFIDF={a['ner_tfidf_score']:.3f}")
        if a['qa_confidence'] > 0:
            print(f"   ثقة QA: {a['qa_confidence']:.3f}")
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import ISRIStemmer
from collections import defaultdict

# Ensure the necessary NLTK data is downloaded
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

def load_data(csv_file_path):
    """تحميل ملف CSV مع معالجة ترميز النص بشكل صحيح"""
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file_path, encoding='cp1256')  # ترميز العربية الشائع
    
    print(f"تم تحميل {len(df)} وثيقة من {csv_file_path}")
    return df

def clean_arabic_text(text):
    """تنظيف النص العربي مع الحفاظ على المعنى"""
    if not isinstance(text, str):
        return ""
    
    # إزالة HTML tags والروابط
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # إزالة الأرقام الإنجليزية والعربية إذا كانت منفردة
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[٠-٩]+', '', text)
    
    # تطبيع الأحرف العربية - محافظ
    text = re.sub(r'[إأآ]', 'ا', text)  # توحيد الألف
    text = re.sub(r'ى', 'ي', text)      # توحيد الياء
    text = re.sub(r'ة', 'ه', text)      # توحيد التاء المربوطة (اختياري)
    
    # إزالة علامات الترقيم الزائدة (ولكن الاحتفاظ بالأساسية)
    text = re.sub(r'[""''`]', '', text)  # إزالة علامات التنصيص المختلفة
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s.،؟!]', ' ', text)
    
    # تنظيف المسافات
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def tokenize_arabic_text(text):
    """تقسيم النص العربي إلى كلمات مع معالجة محسّنة"""
    if not isinstance(text, str) or not text.strip():
        return []
    
    # تنظيف النص أولاً
    text = clean_arabic_text(text)
    
    # استخدام regex للحصول على الكلمات العربية
    pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+'
    tokens = re.findall(pattern, text)
    
    # فلترة الكلمات القصيرة جداً والطويلة جداً
    tokens = [token for token in tokens if 2 <= len(token) <= 20]
    
    # إزالة الكلمات المكررة المتتالية
    if tokens:
        filtered_tokens = [tokens[0]]
        for token in tokens[1:]:
            if token != filtered_tokens[-1]:
                filtered_tokens.append(token)
        tokens = filtered_tokens
    
    return tokens

def remove_stopwords(tokens):
    """إزالة الكلمات الوقفية مع الحفاظ على الكلمات المهمة"""
    try:
        arabic_stopwords = set(stopwords.words('arabic'))
    except:
        arabic_stopwords = set()
    
    # كلمات الاستفهام العربية
    question_words = {
        'ماذا', 'من', 'أين', 'متى', 'كيف', 'لماذا', 'كم', 'أي', 'هل', 
      'ما سبب',  ' ما', 'أين', 'كيف', 'أي', 'هل', 'كم'
    }
    
    # كلمات وقفية إضافية شائعة
    additional_stopwords = {
        'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل',
        'عند', 'حول', 'تحت', 'فوق', 'أمام', 'خلف', 'بين',
        'كان', 'كانت', 'يكون', 'تكون', 'سوف', 'قد', 'لقد',
        'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'الذين', 'اللواتي'
    }
    
    # دمج جميع الكلمات الوقفية
    all_stopwords = arabic_stopwords.union(question_words).union(additional_stopwords)
    
    # إزالة الكلمات الوقفية مع الحفاظ على 70% من الكلمات على الأقل
    original_length = len(tokens)
    filtered_tokens = [token for token in tokens if token.lower() not in all_stopwords]
    
    # إذا تم حذف أكثر من 70% من الكلمات، احتفظ بالنص الأصلي
    if len(filtered_tokens) < original_length * 0.3:
        return tokens
    
    return filtered_tokens if filtered_tokens else tokens

def stem_tokens(tokens):
    """تطبيق الجذور العربية مع معالجة الأخطاء"""
    try:
        stemmer = ISRIStemmer()
    except:
        # إذا فشل تحميل الـ stemmer، أرجع الكلمات كما هي
        return tokens
    
    stemmed_tokens = []
    
    for token in tokens:
        try:
            # محاولة استخراج الجذر
            stemmed = stemmer.stem(token)
            
            # فحص صحة الجذر المستخرج
            if (stemmed and 
                len(stemmed) >= 2 and 
                len(stemmed) >= len(token) * 0.5 and  # الجذر لا يجب أن يكون قصيراً جداً
                len(stemmed) <= len(token)):          # الجذر لا يجب أن يكون أطول من الكلمة الأصلية
                stemmed_tokens.append(stemmed)
            else:
                stemmed_tokens.append(token)  # الاحتفاظ بالكلمة الأصلية
                
        except Exception:
            stemmed_tokens.append(token)  # الاحتفاظ بالكلمة الأصلية في حالة الخطأ
    
    return stemmed_tokens

def preprocess_documents(df):
    """معالجة الوثائق مع تحسينات للبحث السياقي"""
    print("بدء معالجة الوثائق...")
    
    if 'doc' not in df.columns:
        raise ValueError("العمود 'doc' غير موجود في الداتافريم")
    
    # إضافة إحصائيات أولية
    initial_count = len(df)
    print(f"عدد الوثائق الأولي: {initial_count}")
    
    # فلترة الوثائق الفارغة أو القصيرة جداً
    df = df.dropna(subset=['doc'])
    df = df[df['doc'].str.len() > 20]  # إزالة الوثائق القصيرة جداً
    
    print(f"بعد فلترة الوثائق الفارغة: {len(df)}")
    
    # معالجة متدرجة
    print("1. تقسيم الوثائق إلى كلمات...")
    df['Tokenized_Document'] = df['doc'].apply(tokenize_arabic_text)
    
    print("2. إزالة الكلمات الوقفية...")
    df['Tokenized_Document'] = df['Tokenized_Document'].apply(remove_stopwords)
    
  # print("3. استخراج جذور الكلمات...")
    df['Tokenized_Document'] = df['Tokenized_Document'].apply(stem_tokens)
    
    # فلترة الوثائق التي لا تحتوي على كلمات كافية بعد المعالجة
    df = df[df['Tokenized_Document'].apply(len) >= 3]
    
    final_count = len(df)
    print(f"عدد الوثائق النهائي: {final_count}")
    print(f"تم حذف {initial_count - final_count} وثيقة")
    
    # إضافة معلومات إضافية للبحث
    df['Original_Length'] = df['doc'].str.len()
    df['Token_Count'] = df['Tokenized_Document'].apply(len)
    
    return df.reset_index(drop=True)

def create_inverted_index(df):
    """إنشاء فهرس مقلوب محسّن للبحث السريع"""
    print("إنشاء الفهرس المقلوب...")
    
    inverted_index = defaultdict(list)
    token_frequencies = defaultdict(int)  # لحساب تكرار الكلمات
    
    for index, row in df.iterrows():
        doc_id = row['QID']
        tokens = row['Tokenized_Document']
        
        # إضافة الكلمات الفريدة فقط لكل وثيقة
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)
                token_frequencies[token] += 1
    
    # تحويل إلى dict عادي وإضافة إحصائيات
    final_index = dict(inverted_index)
    
    print(f"تم إنشاء فهرس لـ {len(final_index)} كلمة فريدة")
    print(f"متوسط عدد الوثائق لكل كلمة: {sum(len(docs) for docs in final_index.values()) / len(final_index):.1f}")
    
    return final_index


def validate_preprocessing(df):
    """التحقق من صحة المعالجة"""
    issues = []
    
    # فحص الوثائق الفارغة
    empty_docs = df[df['Tokenized_Document'].apply(len) == 0]
    if not empty_docs.empty:
        issues.append(f"وجدت {len(empty_docs)} وثيقة فارغة بعد المعالجة")
    
    # فحص الوثائق القصيرة جداً
    short_docs = df[df['Tokenized_Document'].apply(len) < 3]
    if not short_docs.empty:
        issues.append(f"وجدت {len(short_docs)} وثيقة قصيرة جداً (أقل من 3 كلمات)")
    
    # فحص الكلمات الطويلة جداً
    long_tokens = []
    for tokens in df['Tokenized_Document']:
        long_tokens.extend([token for token in tokens if len(token) > 20])
    
    if long_tokens:
        issues.append(f"وجدت {len(long_tokens)} كلمة طويلة جداً (أكثر من 20 حرف)")
    
    if issues:
        print("تحذيرات المعالجة:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ تم التحقق من المعالجة بنجاح")
    
    return issues

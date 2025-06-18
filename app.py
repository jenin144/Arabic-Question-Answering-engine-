from flask import Flask, render_template, request
from src.main import main_process_query as process_query, initialize_system
import logging
import time

app = Flask(__name__)

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# متغير لحفظ حالة التهيئة
system_initialized = False

def ensure_template_compatibility(answers):
    """ضمان توافق الإجابات مع الـ template"""
    if not answers:
        return []
    
    compatible_answers = []
    
    for i, answer in enumerate(answers):
        # التأكد من وجود جميع المفاتيح المطلوبة
        compatible_answer = {
            'rank': answer.get('rank', i + 1),
            'answer': answer.get('answer', ''),
            'document_id': answer.get('document_id', ''),
            'similarity_score': float(answer.get('similarity_score', 0.0)),
            'relevance_score': float(answer.get('relevance_score', answer.get('similarity_score', 0.0))),
            'source_link': answer.get('source_link'),
            'source_title': answer.get('source_title', f'مصدر طبي رقم {i + 1}')
        }
        
        # التأكد من أن النص ليس فارغاً
        if not compatible_answer['answer'].strip():
            compatible_answer['answer'] = 'لم يتم العثور على إجابة مناسبة'
        
        # التأكد من أن النقاط في المدى المناسب
        compatible_answer['similarity_score'] = max(0.0, min(1.0, compatible_answer['similarity_score']))
        compatible_answer['relevance_score'] = max(0.0, min(1.0, compatible_answer['relevance_score']))
        
        compatible_answers.append(compatible_answer)
    
    return compatible_answers

def initialize_system_once():
    """تهيئة النظام مرة واحدة"""
    global system_initialized
    if not system_initialized:
        try:
            logger.info("تهيئة النظام...")
            initialize_system()
            system_initialized = True
            logger.info("تم تهيئة النظام بنجاح")
            return True
        except Exception as e:
            logger.error(f"خطأ في تهيئة النظام: {e}")
            return False
    return True

@app.route('/', methods=['GET', 'POST'])
def index():
    # تهيئة النظام إذا لم يكن قد تم تهيئته
    if not initialize_system_once():
        error_message = "خطأ في تهيئة النظام. يرجى المحاولة مرة أخرى."
        return render_template('index.html', 
                             query="", 
                             result={'top_answers': [], 'error': error_message, 'total_found': 0},
                             processing_time=0)
    
    if request.method == 'POST':
        start_time = time.time()
        
        try:
            query = request.form.get('query', '').strip()
            
            if not query:
                return render_template('index.html', 
                                     query="", 
                                     result={'top_answers': [], 'error': 'يرجى إدخال سؤال للبحث', 'total_found': 0},
                                     processing_time=0)
            
            logger.info(f"معالجة الاستعلام: {query}")
            
            # معالجة الاستعلام
            result = process_query(query)
            
            # حساب وقت المعالجة
            processing_time = round((time.time() - start_time) * 1000, 2)
            
            # التأكد من توافق البيانات مع الـ template
            if 'top_answers' in result:
                result['top_answers'] = ensure_template_compatibility(result['top_answers'])
            
            # إضافة وقت المعالجة للنتيجة
            result['processing_time'] = processing_time
            
            logger.info(f"تم العثور على {len(result.get('top_answers', []))} إجابات في {processing_time}ms")
            
            return render_template('index.html', 
                                 query=query, 
                                 result=result,
                                 processing_time=processing_time)
        
        except Exception as e:
            processing_time = round((time.time() - start_time) * 1000, 2)
            logger.error(f"خطأ في معالجة البحث: {e}")
            
            error_result = {
                'top_answers': [],
                'error': 'حدث خطأ في البحث. يرجى المحاولة مرة أخرى.',
                'total_found': 0,
                'processing_time': processing_time
            }
            
            return render_template('index.html', 
                                 query=request.form.get('query', ''), 
                                 result=error_result,
                                 processing_time=processing_time)
    
    # GET request - الصفحة الأولى
    return render_template('index.html', 
                         query="", 
                         result={'top_answers': [], 'error': None, 'total_found': 0},
                         processing_time=0)

@app.route('/test')
def test():
    """طريق اختبار للتحقق من عمل النظام"""
    try:
        # تهيئة النظام أولاً
        if not initialize_system_once():
            return "<h1>خطأ في تهيئة النظام</h1>"
        
        from src.main import test_query
        
        test_queries = [
            "ما هي اعراض التهاب طبلة الاذن؟",
            "علاج الصداع",
            "أعراض السكري"
        ]
        
        results = []
        for query in test_queries:
            try:
                result = test_query(query)
                results.append(f"<h3>الاستعلام: {query}</h3>")
                results.append(f"<p>النتائج: {len(result.get('top_answers', []))} إجابات</p>")
                
                if result.get('top_answers'):
                    results.append("<ul>")
                    for answer in result['top_answers'][:2]:  # أول إجابتين فقط
                        results.append(f"<li>{answer.get('answer', '')[:200]}...</li>")
                    results.append("</ul>")
                else:
                    results.append(f"<p style='color: red;'>خطأ: {result.get('error', 'لا توجد إجابات')}</p>")
                
                results.append("<hr>")
                
            except Exception as e:
                results.append(f"<h3>الاستعلام: {query}</h3>")
                results.append(f"<p style='color: red;'>خطأ: {str(e)}</p>")
                results.append("<hr>")
        
        html_content = f"""
        <html dir="rtl">
        <head>
            <meta charset="UTF-8">
            <title>اختبار النظام</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; direction: rtl; }}
                h1 {{ color: #2c3e50; }}
                h3 {{ color: #27ae60; }}
                p {{ margin: 10px 0; }}
                ul {{ margin: 10px 0; }}
                li {{ margin: 5px 0; }}
                hr {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🧪 نتائج اختبار النظام</h1>
            <p><strong>حالة النظام:</strong> {'✅ يعمل' if system_initialized else '❌ غير مُهيأ'}</p>
            <p><strong>عدد الاختبارات:</strong> {len(test_queries)}</p>
            <hr>
            {''.join(results)}
            <p><a href="/">العودة للصفحة الرئيسية</a></p>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        return f"""
        <html dir="rtl">
        <head><meta charset="UTF-8"><title>خطأ في الاختبار</title></head>
        <body style="font-family: Arial; margin: 20px; direction: rtl;">
            <h1 style="color: red;">❌ خطأ في الاختبار</h1>
            <p><strong>الخطأ:</strong> {str(e)}</p>
            <p><a href="/">العودة للصفحة الرئيسية</a></p>
        </body>
        </html>
        """

@app.route('/api/search', methods=['POST'])
def api_search():
    """API للبحث - إرجاع JSON"""
    try:
        # تهيئة النظام
        if not initialize_system_once():
            return {
                'success': False,
                'error': 'خطأ في تهيئة النظام',
                'answers': [],
                'total_found': 0
            }, 500
        
        start_time = time.time()
        
        # الحصول على البيانات
        if request.is_json:
            data = request.get_json()
            query = data.get('query', '').strip()
        else:
            query = request.form.get('query', '').strip()
        
        if not query:
            return {
                'success': False,
                'error': 'يرجى إدخال سؤال للبحث',
                'answers': [],
                'total_found': 0
            }, 400
        
        # معالجة الاستعلام
        result = process_query(query)
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        # التأكد من التوافق
        answers = ensure_template_compatibility(result.get('top_answers', []))
        
        return {
            'success': True,
            'query': query,
            'answers': answers,
            'total_found': result.get('total_found', 0),
            'processing_time': processing_time,
            'error': result.get('error')
        }
        
    except Exception as e:
        logger.error(f"خطأ في API البحث: {e}")
        return {
            'success': False,
            'error': 'حدث خطأ في البحث',
            'answers': [],
            'total_found': 0
        }, 500

@app.route('/health')
def health_check():
    """فحص صحة النظام"""
    try:
        system_ok = initialize_system_once()
        
        return {
            'status': 'healthy' if system_ok else 'unhealthy',
            'system_initialized': system_initialized,
            'message': 'النظام يعمل بشكل طبيعي' if system_ok else 'يوجد مشكلة في تهيئة النظام'
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'system_initialized': False,
            'error': str(e),
            'message': 'يوجد مشكلة في النظام'
        }, 500

@app.errorhandler(404)
def page_not_found(e):
    """معالجة خطأ 404"""
    return """
    <html dir="rtl">
    <head><meta charset="UTF-8"><title>صفحة غير موجودة</title></head>
    <body style="font-family: Arial; margin: 20px; direction: rtl; text-align: center;">
        <h1 style="color: #e74c3c;">❌ الصفحة غير موجودة</h1>
        <p>الصفحة التي تبحث عنها غير موجودة.</p>
        <p><a href="/" style="color: #3498db;">العودة للصفحة الرئيسية</a></p>
    </body>
    </html>
    """, 404

@app.errorhandler(500)
def internal_server_error(e):
    """معالجة خطأ 500"""
    logger.error(f"خطأ داخلي في الخادم: {e}")
    return """
    <html dir="rtl">
    <head><meta charset="UTF-8"><title>خطأ في الخادم</title></head>
    <body style="font-family: Arial; margin: 20px; direction: rtl; text-align: center;">
        <h1 style="color: #e74c3c;">⚠️ خطأ في الخادم</h1>
        <p>حدث خطأ داخلي في الخادم. يرجى المحاولة مرة أخرى.</p>
        <p><a href="/" style="color: #3498db;">العودة للصفحة الرئيسية</a></p>
    </body>
    </html>
    """, 500

if __name__ == '__main__':
    print("🚀 بدء تشغيل نظام البحث الطبي المُحسّن...")
    print("📊 سيتم تهيئة النظام عند أول طلب")
    print("🌐 الصفحة الرئيسية: http://localhost:5000")
    print("🧪 صفحة الاختبار: http://localhost:5000/test")
    print("📋 API البحث: http://localhost:5000/api/search")
    print("💚 فحص الصحة: http://localhost:5000/health")
    
    app.run(debug=True)
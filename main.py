import random
import os
import uuid
import nltk
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from nltk.metrics import edit_distance

# --- Инициализация NLP ---
nltk.download('punkt')
segmenter = Segmenter()
emb = NewsEmbedding()
morph_vocab = MorphVocab()
morph_tagger = NewsMorphTagger(emb)

# --- Конфиг бота ---
BOT_CONFIG = {
    'intents': {
        'greeting': {
            'examples': ['привет', 'здравствуй', 'старт', 'добрый день'],
            'responses': ['Здравствуйте! Чем могу помочь?']
        },
        'goodbye': {
            'examples': ['пока', 'до свидания', 'увидимся'],
            'responses': ['До встречи! Заходите ещё.']
        }
    },
    'failure_phrases': [
        'Не совсем понял, можете уточнить?',
        'Попробуйте переформулировать вопрос.',
        'Я ещё учусь, но стараюсь понять вас.'
    ],
    'products': [
        {
            'name': 'iPhone 15 Pro',
            'keywords': ['яблоко', 'айфон', 'iphone'],
            'price': '129 990 ₽',
            'desc': 'Титан, камера 48 Мп'
        },
        {
            'name': 'Samsung Galaxy S24',
            'keywords': ['самсунг', 'галакси', 'android'],
            'price': '89 990 ₽',
            'desc': 'Экран 144 Гц'
        },
        {
            'name': 'Xiaomi Redmi Note 13',
            'keywords': ['ксиаоми', 'редми', 'бюджетный'],
            'price': '24 990 ₽',
            'desc': 'AMOLED экран'
        }
    ]
}

# Словарь эмоций
emo_dict = {
    'люблю': 1, 'обожаю': 1, 'супер': 1, 'отлично': 1, 'счастлив': 1,
    'ненавижу': -1, 'ужас': -1, 'плохо': -1, 'грусть': -1, 'злюсь': -1,
    'нравится': 1, 'печально': -1, 'нормально': 0, 'такое': 0,
}

# --- Загрузка диалогов ---
def load_dialogues():
    out = []
    try:
        with open('dialogues.txt', 'r', encoding='utf-8') as f:
            for block in f.read().split('\n\n'):
                parts = block.strip().split('\n')
                if len(parts) >= 2:
                    out.append((parts[0].strip(), parts[1].strip()))
    except FileNotFoundError:
        pass
    return out


dialogues = load_dialogues()

# --- Обучение классификатора намерений ---
X_text, y = [], []
for intent, data in BOT_CONFIG['intents'].items():
    X_text += data['examples']
    y += [intent] * len(data['examples'])

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC().fit(X, y)

# --- NLP‑функции ---
def clean_text(text: str) -> str:
    return ''.join(c for c in text.lower() if c.isalpha() or c == ' ').strip()


def lemmatize(text: str) -> str:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for tok in doc.tokens:
        tok.lemmatize(morph_vocab)
    return ' '.join(tok.lemma for tok in doc.tokens)


def analyze_sentiment(text: str) -> str:
    s = sum(emo_dict.get(w, 0) for w in text.split())
    return 'положительный' if s > 0 else 'отрицательный' if s < 0 else 'нейтральный'


def classify_intent(text: str) -> dict:
    lem = lemmatize(text)
    for p in BOT_CONFIG['products']:
        if any(k in lem for k in p['keywords']):
            return {'type': 'product', 'data': p}
    intent = clf.predict(vectorizer.transform([clean_text(text)]))[0]
    return {'type': 'intent', 'data': intent}


def generate_answer(text: str) -> str:
    t = clean_text(text)
    t_words = set(t.split())
    for q, a in dialogues:
        qc = clean_text(q)
        qc_words = set(qc.split())
        if qc_words <= t_words:
            return a
        if edit_distance(t, qc) / max(len(qc), 1) < 0.4:
            return a
    return None

# --- Отправка ответа с поддержкой голосового режима ---
async def send_response(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    mode = context.user_data.get('response_mode', 'text')
    if mode == 'voice':
        # Генерируем голосовое сообщение через gTTS
        path = f"tts_{uuid.uuid4()}.mp3"
        try:
            gTTS(text, lang='ru').save(path)
            await update.message.reply_voice(voice=open(path, 'rb'))
        finally:
            if os.path.exists(path):
                os.remove(path)
    else:
        await update.message.reply_text(text)

# --- Обработка текстовых сообщений ---
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, *, recognized_text: str = None):
    text_raw = recognized_text or update.message.text or ""
    text = clean_text(text_raw)

    # Управление режимом ответа
    if text in ['голосом', 'голос']:
        context.user_data['response_mode'] = 'voice'
        await send_response(update, context, "Переключился на голосовые ответы.")
        return
    elif text in ['текстом', 'текст']:
        context.user_data['response_mode'] = 'text'
        await send_response(update, context, "Переключился на текстовые ответы.")
        return

    # 0. Спец-ответ на «как дела»
    if 'дела' in text and ('как' in text or 'у тебя' in text):
        await send_response(update, context, "У меня всё отлично! А у вас как?")
        return

    # 1. Анализ тональности
    tone = analyze_sentiment(text)
    await send_response(update, context, f"🔍 Тональность: {tone}")

    # 2. Ответ по диалогам из файла
    answer = generate_answer(text)
    if answer:
        await send_response(update, context, answer)
        return

    # 3. Прямой запрос товаров по ключевым словам
    if any(word in text for word in ['телефон', 'смартфон', 'модел', 'покажи', 'новинк']):
        await show_products(update, context)
        return

    # 4. Обработка «да/нет» после предложения
    if context.user_data.get('last_intent') == 'offer_products':
        if text in ['да', 'хочу']:
            await show_products(update, context)
            return
        elif text in ['нет', 'не хочу']:
            context.user_data['last_intent'] = None
            await send_response(update, context, "Понятно. О чём ещё поговорим?")
            return

    # 5. Классификация: товары или intent
    intent = classify_intent(text)
    if intent['type'] == 'product':
        product = intent['data']
        context.user_data['selected'] = product
        context.user_data['last_intent'] = 'offer_products'
        await send_response(update, context,
                            f"🔹 {product['name']} — {product['desc']}\nЦена: {product['price']}\nНапишите «оформить», чтобы заказать."
                            )
        return
    elif intent['type'] == 'intent':
        response = random.choice(
            BOT_CONFIG['intents'][intent['data']]['responses'])
        context.user_data['last_intent'] = intent['data']
        await send_response(update, context, response)
        return

    # 6. Оформление заказа
    if text == 'оформить' and context.user_data.get('selected'):
        product = context.user_data['selected']
        context.user_data.clear()
        await send_response(update, context, f"✅ Заказ на {product['name']} оформлен. Спасибо!")
        return

    # 7. Фоллбек-ответ
    await send_response(update, context, random.choice(BOT_CONFIG['failure_phrases']))


# --- Обработка голосовых сообщений ---
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ogg_path = wav_path = None
    try:
        voice_file = await update.message.voice.get_file()
        ogg_path = f"voice_{uuid.uuid4()}.ogg"
        await voice_file.download_to_drive(ogg_path)

        audio = AudioSegment.from_ogg(ogg_path)
        wav_path = ogg_path.replace('.ogg', '.wav')
        audio.export(wav_path, format='wav')

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as src:
            data = recognizer.record(src)
            text = recognizer.recognize_google(data, language='ru-RU')

        await send_response(update, context, f"🗣 Распознано: {text}")
        await handle_text(update, context, recognized_text=text)

    except Exception as e:
        await send_response(update, context, f"⚠️ Ошибка распознавания: {e}")

    finally:
        for path in (ogg_path, wav_path):
            if path and os.path.exists(path):
                os.remove(path)

# --- Команды ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    context.user_data['last_intent'] = None
    context.user_data['response_mode'] = 'text'
    await update.message.reply_text(
        "Привет! Я бот. Напишите сообщение или отправьте голосовое сообщение.\n"
        "Команды для переключения режима ответа:\n"
        "- голосом\n"
        "- текстом"
    )

# --- Отобразить товары ---
async def show_products(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["Вот наши модели телефонов:"]
    for p in BOT_CONFIG['products']:
        lines.append(f"• {p['name']}: {p['desc']} — {p['price']}")
    lines.append(
        "Напишите название модели, чтобы узнать подробности, или «оформить» для заказа.")
    await send_response(update, context, '\n'.join(lines))
    context.user_data['last_intent'] = 'offer_products'

# --- Запуск ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(
        '7995703622:AAHcpbmNiv70ZoM3te69qiUZdEZhlHv8Drk').build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(
        filters.TEXT & (~filters.COMMAND), handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    print("Bot started")
    app.run_polling()

import telebot
from telebot import types
import re
import nltk
import pymorphy3
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from scipy.stats import chi2_contingency
import io
import math
import random
import colorsys

morph = pymorphy3.MorphAnalyzer()
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
user_sessions = {}

bot = telebot.TeleBot('8570990084:AAFeiYBY4tcwymPVx_PH8_BzQGBUxF13H3I')


class UserData:
    """
    Класс для хранения данных пользователя во время сессии.
    Он используется для отслеживания состояния пользователя,
    загруженных данных и параметров текущего анализа.

    :ivar state: Текущее состояние пользователя в диалоге.
    :vartype state: str
    :ivar analysis_type: Тип выбранного анализа.
    :vartype analysis_type: str or None
    :ivar books: Словарь с загруженными книгами (ключ - автор, значение - текст).
    :vartype books: dict
    :ivar anonymous_book: Текст анонимной книги для анализа авторства.
    :vartype anonymous_book: str or None
    :ivar characters: Список персонажей для анализа.
    :vartype characters: list
    :ivar expected_books_count: Ожидаемое количество книг для текущего анализа.
    :vartype expected_books_count: int
    :ivar books_received: Количество уже полученных книг.
    :vartype books_received: int
    :ivar authorship_method: Выбранный метод анализа авторства.
    :vartype authorship_method: str or None
    :ivar words_by_author: Словарь с предобработанными словами по авторам.
    :vartype words_by_author: dict
    """
    def __init__(self):
        """
        Инициализирует новый объект UserData с начальными значениями.
        Создает все необходимые атрибуты для отслеживания состояния пользователя
        в процессе взаимодействия с ботом.
        """
        self.state = 'main_menu'
        self.analysis_type = None
        self.books = {}
        self.anonymous_book = None
        self.characters = []
        self.expected_books_count = 0
        self.books_received = 0
        self.authorship_method = None
        self.words_by_author = {}


def preprocess_text(text):
    """
    Предобработка текста: токенизация, лемматизация и удаление стоп-слов.

    :param text: Исходный текст для обработки.
    :type text: str
    :returns: Список лемматизированных слов.
    :rtype: list[str]
    """
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[^\w\s-]', '', text)
    tokens = text.split()
    russian_stopwords = set(nltk.corpus.stopwords.words('russian'))
    english_stopwords = set(nltk.corpus.stopwords.words('english'))
    lemmas = []
    for token in tokens:
        if token not in russian_stopwords and token not in english_stopwords and len(token) > 2:
            parsed = morph.parse(token)[0]
            lemmas.append(parsed.normal_form)
    return lemmas


def get_word_length_frequencies(text):
    """
    Вычисление частотности длин слов в тексте.

    :param text: Текст для анализа.
    :type text: str
    :returns: Словарь, где ключ - длина слова, значение - количество слов такой длины.
    :rtype: dict[int, int]
    """
    words = re.findall(r'\b\w+\b', text.lower())
    length_freq = Counter()
    for word in words:
        length_freq[len(word)] += 1
    return dict(sorted(length_freq.items()))


def get_sentence_length_frequencies(text):
    """
    Вычисление частотности длин предложений.

    :param text: Текст для анализа.
    :type text: str
    :returns: Словарь, где ключ - длина предложения, значение - количество предложений такой длины.
    :rtype: dict[int, int]
    """
    sentences = re.split(r'[.!?]+', text)
    length_freq = Counter()
    for sent in sentences:
        words = re.findall(r'\b\w+\b', sent)
        if words:
            length_freq[len(words)] += 1
    return dict(sorted(length_freq.items()))


def prepare_words_by_author(user_data):
    """
    Подготавливает словарь words_by_author для анализа.

    :param user_data: Данные пользователя с загруженными книгами.
    :type user_data: UserData
    :returns: Словарь с предобработанными словами по авторам.
    :rtype: dict[str, list[str]]
    """
    words_by_author = {}
    for author, text in user_data.books.items():
        words = preprocess_text(text)[:20000]
        words_by_author[author] = words

    if user_data.anonymous_book:
        words = preprocess_text(user_data.anonymous_book)[:20000]
        words_by_author['unknown'] = words

    return words_by_author


def chi_square_test(words_by_author):
    """
    Сравнение лексикона авторов с использованием критерия хи-квадрат.

    :param words_by_author: Словарь с предобработанными словами по авторам.
    :type words_by_author: dict[str, list[str]]
    :returns: Кортеж (результаты в текстовом виде, словарь со значениями хи-квадрат).
    :rtype: tuple[list[str], dict[str, float]]
    """
    results = []
    chisquared_by_author = {}

    for author in words_by_author:
        if author != 'unknown':
            combined_corpus = (words_by_author[author] +
                               words_by_author['unknown'])

            author_proportion = (len(words_by_author[author]) / len(combined_corpus))

            combined_freq_dist = nltk.FreqDist(combined_corpus)
            most_common_words = list(combined_freq_dist.most_common(1000))
            chisquared = 0
            for word, combined_count in most_common_words:
                observed_count_author = words_by_author[author].count(word)
                expected_count_author = combined_count * author_proportion

                if expected_count_author > 0:
                    chisquared += ((observed_count_author - expected_count_author) ** 2 /
                                   expected_count_author)
            chisquared_by_author[author] = chisquared

    for author, chi2 in chisquared_by_author.items():
        results.append(f"Хи-квадрат для {author.capitalize()} = {chi2:.1f}")

    if chisquared_by_author:
        most_likely_author = min(chisquared_by_author,
                                 key=chisquared_by_author.get)
        results.append(
            f"\n*Судя по распределению частот слов, наиболее вероятный автор: {most_likely_author.capitalize()}*")

    return results, chisquared_by_author


def jaccard_test(words_by_author):
    """
    Вычисляет коэффициент Жаккара для каждого текста в сравнении с анонимным текстом.

    :param words_by_author: Словарь с предобработанными словами по авторам.
    :type words_by_author: dict[str, list[str]]
    :returns: Кортеж (результаты в текстовом виде, словарь с коэффициентами Жаккара).
    :rtype: tuple[list[str], dict[str, float]]
    """
    results = []
    jaccard_by_author = {}

    unique_words_unknown = set(words_by_author['unknown'][:20000])
    authors = []
    for author in words_by_author:
        if author != 'unknown':
            authors.append(author)

    for author in authors:
        unique_words_author = set(words_by_author[author][:20000])
        shared_words = unique_words_author.intersection(unique_words_unknown)

        jaccard_sim = (float(len(shared_words)) /
                       (len(unique_words_author) +
                        len(unique_words_unknown) -
                        len(shared_words)))

        jaccard_by_author[author] = jaccard_sim

        similarity_percent = jaccard_sim * 100
        results.append(f"Коэффициент Жаккара для {author.capitalize()} = {jaccard_sim:.3f} ({similarity_percent:.1f}%)")

    if jaccard_by_author:
        most_likely_author = max(jaccard_by_author,
                                 key=jaccard_by_author.get)
        results.append(f"\n*Судя по схожести лексики, наиболее вероятный автор: {most_likely_author.capitalize()}*")

    return results, jaccard_by_author


def get_random_color():
    """
    Генерирует случайный цвет в формате hex.

    :returns: Случайный цвет в формате HEX (#RRGGBB).
    :rtype: str
    """
    h = random.random()
    s = random.uniform(0.5, 0.9)
    v = random.uniform(0.7, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def create_word_cloud(text):
    """
    Создание облака слов со случайным цветом.

    :param text: Текст для создания облака слов.
    :type text: str
    :returns: Байтовый буфер с изображением облака слов.
    :rtype: io.BytesIO
    """
    cleaned_text = ' '.join(preprocess_text(text))
    random_color = get_random_color()

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        h, s, v = colorsys.rgb_to_hsv(
            int(random_color[1:3], 16) / 255,
            int(random_color[3:5], 16) / 255,
            int(random_color[5:7], 16) / 255
        )
        h_variation = h + random.uniform(-0.1, 0.1)
        s_variation = max(0.3, min(0.9, s + random.uniform(-0.1, 0.1)))
        v_variation = max(0.5, min(1.0, v + random.uniform(-0.1, 0.1)))
        r, g, b = colorsys.hsv_to_rgb(h_variation, s_variation, v_variation)

        return f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})'

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        contour_width=1,
        contour_color='steelblue',
        color_func=color_func
    ).generate(cleaned_text)

    img_buffer = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Облако слов', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(img_buffer, format='png', dpi=150)
    plt.close()
    img_buffer.seek(0)
    return img_buffer


def plot_character_dispersion(text, characters):
    """
    Построение дисперсионного графика появления персонажей по тексту.

    :param text: Текст для анализа.
    :type text: str
    :param characters: Список имен персонажей для поиска.
    :type characters: list[str]
    :returns: Байтовый буфер с изображением графика.
    :rtype: io.BytesIO
    """
    text_lower = text.lower()
    tokens = text_lower.split()
    target_words = [char.lower() for char in characters]
    word_positions = {}

    for word in target_words:
        positions = []
        for i, token in enumerate(tokens):
            if word in token:
                positions.append(i)
        word_positions[word] = positions

    plt.figure(figsize=(12, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    y_positions = list(range(len(target_words)))

    for i, (word, positions) in enumerate(word_positions.items()):
        if positions:
            plt.scatter(positions, [i] * len(positions),
                        s=10, color=colors[i % len(colors)],
                        marker='|', label=word.capitalize())

    plt.yticks(y_positions, [word.capitalize() for word in target_words])
    plt.xlabel('Позиция слова в тексте', fontsize=12)
    plt.ylabel('Персонажи', fontsize=12)
    plt.title('График появления персонажей в тексте', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150)
    plt.close()
    img_buffer.seek(0)
    return img_buffer


def plot_word_length_comparison(books_dict):
    """
    График частотности длин слов для разных авторов.

    :param books_dict: Словарь с текстами книг по авторам.
    :type books_dict: dict[str, str]
    :returns: Байтовый буфер с изображением графика.
    :rtype: io.BytesIO
    """
    plt.figure(figsize=(12, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for idx, (author, text) in enumerate(books_dict.items()):
        freqs = get_word_length_frequencies(text)
        if freqs:
            lengths = list(freqs.keys())
            counts = list(freqs.values())
            total = sum(counts)
            normalized = [c / total for c in counts]
            color = colors[idx % len(colors)]
            plt.plot(lengths, normalized, marker='o', label=author, linewidth=2, color=color)

    plt.xlabel('Длина слова', fontsize=12)
    plt.ylabel('Количество вхождений', fontsize=12)
    plt.title('Распределение длин слов по авторам', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150)
    plt.close()
    img_buffer.seek(0)
    return img_buffer


def plot_sentence_length_comparison(books_dict):
    """
    График частотности длин предложений для разных авторов.

    :param books_dict: Словарь с текстами книг по авторам.
    :type books_dict: dict[str, str]
    :returns: Байтовый буфер с изображением графика.
    :rtype: io.BytesIO
    """
    plt.figure(figsize=(12, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for idx, (author, text) in enumerate(books_dict.items()):
        freqs = get_sentence_length_frequencies(text)
        if freqs:
            lengths = list(freqs.keys())[:30]
            counts = [freqs.get(l, 0) for l in lengths]
            total = sum(counts)
            normalized = [c / total if total > 0 else 0 for c in counts]
            color = colors[idx % len(colors)]
            plt.plot(lengths, normalized, marker='s', label=author, linewidth=2, color=color)

    plt.xlabel('Длина предложения', fontsize=12)
    plt.ylabel('Количество вхождений', fontsize=12)
    plt.title('Распределение длин предложений по авторам', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150)
    plt.close()
    img_buffer.seek(0)
    return img_buffer


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """
    Обработчик команд /start и /help. Отправляет приветственное сообщение.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        user_sessions[user_id] = UserData()

    welcome_text = """
📚 *Добро пожаловать в Book Analyzer Bot!*

*Выберите тип анализа:*

1. ☁️ *Облако слов* - создание визуализации частых слов из книги
2. 👥 *График персонажей* - анализ появления персонажей в тексте
3. 🔍 *Анализ авторства* - определение автора анонимной книги

*Как работает анализ авторства:*
1. Вы загружаете 2 книги с известными авторами
2. Затем загружаете анонимную книгу
3. Бот сравнивает её с двумя известными

Выберите действие с помощью кнопок ниже 👇
"""

    markup = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    btn1 = types.KeyboardButton('☁️ Облако слов')
    btn2 = types.KeyboardButton('👥 График персонажей')
    btn3 = types.KeyboardButton('🔍 Анализ авторства')
    markup.add(btn1, btn2, btn3)

    user_sessions[user_id].state = 'main_menu'
    bot.reply_to(message, welcome_text, parse_mode='Markdown', reply_markup=markup)


@bot.message_handler(func=lambda message: message.text == '☁️ Облако слов')
def handle_word_cloud_choice(message):
    """
    Обработчик выбора анализа "Облако слов".

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        user_sessions[user_id] = UserData()

    user_data = user_sessions[user_id]
    user_data.state = 'waiting_book_wordcloud'
    user_data.analysis_type = 'word_cloud'
    user_data.expected_books_count = 1
    user_data.books_received = 0
    user_data.books.clear()

    bot.reply_to(message,
                 "☁️ *Выбран анализ: Облако слов*\n\n"
                 "Отправьте мне *одну книгу* в формате .txt\n"
                 "Я создам визуализацию самых частых слов.",
                 parse_mode='Markdown')


@bot.message_handler(func=lambda message: message.text == '👥 График персонажей')
def handle_characters_choice(message):
    """
    Обработчик выбора анализа "График персонажей".

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        user_sessions[user_id] = UserData()

    user_data = user_sessions[user_id]
    user_data.state = 'waiting_book_characters'
    user_data.analysis_type = 'characters'
    user_data.expected_books_count = 1
    user_data.books_received = 0
    user_data.books.clear()

    bot.reply_to(message,
                 "👥 *Выбран анализ: График персонажей*\n\n"
                 "Отправьте мне *одну книгу* в формате .txt\n"
                 "После загрузки я попрошу ввести имена персонажей для анализа.\n"
                 "*Этот график показывает, где в тексте появляются персонажи*",
                 parse_mode='Markdown')


@bot.message_handler(func=lambda message: message.text == '🔍 Анализ авторства')
def handle_authorship_choice(message):
    """
    Обработчик выбора анализа "Анализ авторства".

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        user_sessions[user_id] = UserData()

    user_data = user_sessions[user_id]
    user_data.state = 'waiting_author1'
    user_data.analysis_type = 'authorship'
    user_data.expected_books_count = 3
    user_data.books_received = 0
    user_data.books.clear()
    user_data.anonymous_book = None
    user_data.authorship_method = None

    markup = types.ForceReply(selective=False)

    bot.reply_to(message,
                 "🔍 *Выбран анализ: Определение авторства*\n\n"
                 "📚 *Шаг 1 из 3*\n"
                 "Отправьте *первую книгу* с известным автором.\n"
                 "После отправки файла, введите имя автора этой книги.",
                 parse_mode='Markdown',
                 reply_markup=markup)


@bot.message_handler(content_types=['document'])
def handle_document(message):
    """
    Обработка загружаемых текстовых файлов.

    :param message: Сообщение с документом от пользователя.
    :type message: telebot.types.Message
    :raises Exception: При ошибке чтения файла.
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        bot.reply_to(message, "Сначала выберите тип анализа через /start")
        return

    user_data = user_sessions[user_id]

    if message.document.mime_type != 'text/plain' and not message.document.file_name.endswith('.txt'):
        bot.reply_to(message, "❌ Пожалуйста, загрузите текстовый файл в формате .txt")
        return

    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        try:
            text = downloaded_file.decode('utf-8')
        except:
            try:
                text = downloaded_file.decode('windows-1251')
            except:
                bot.reply_to(message, "❌ Не удалось прочитать файл. Используйте кодировку UTF-8.")
                return

        if user_data.state == 'waiting_book_wordcloud':
            handle_word_cloud_book(message, text, user_data)

        elif user_data.state == 'waiting_book_characters':
            handle_characters_book(message, text, user_data)

        elif user_data.state == 'waiting_author1':
            handle_author_book(message, text, user_data, 1)

        elif user_data.state == 'waiting_author2':
            handle_author_book(message, text, user_data, 2)

        elif user_data.state == 'waiting_anonymous_book':
            handle_anonymous_book(message, text, user_data)

        else:
            bot.reply_to(message, "❌ Неправильная последовательность действий. Начните с /start")

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка при обработке файла: {str(e)}")


def handle_word_cloud_book(message, text, user_data):
    """
    Обработка книги для облака слов.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    :param text: Текст загруженной книги.
    :type text: str
    :param user_data: Данные пользователя.
    :type user_data: UserData
    """
    user_data.books['book1'] = text
    user_data.books_received += 1

    bot.reply_to(message, "✅ Книга загружена! Создаю облако слов...")

    try:
        img_buffer = create_word_cloud(text)
        bot.send_photo(message.chat.id, img_buffer,
                       caption="☁️ Облако слов из загруженной книги")

        return_to_main_menu(message.chat.id, user_data)

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка при создании облака слов: {str(e)}")


def handle_characters_book(message, text, user_data):
    """
    Обработка книги для графика персонажей.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    :param text: Текст загруженной книги.
    :type text: str
    :param user_data: Данные пользователя.
    :type user_data: UserData
    """
    user_data.books['book1'] = text
    user_data.books_received += 1
    user_data.state = 'waiting_characters_list'

    markup = types.ForceReply(selective=False)
    bot.reply_to(message,
                 "✅ Книга загружена!\n\n"
                 "👥 Теперь введите имена персонажей для анализа (через запятую):\n"
                 "Пример: *Евгений, Татьяна, Владимир*\n\n"
                 "Бот будет искать эти имена в тексте и построит график.",
                 parse_mode='Markdown',
                 reply_markup=markup)


def handle_author_book(message, text, user_data, author_num):
    """
    Обработка книги с известным автором.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    :param text: Текст загруженной книги.
    :type text: str
    :param user_data: Данные пользователя.
    :type user_data: UserData
    :param author_num: Номер автора (1 или 2).
    :type author_num: int
    """
    user_data.books[f'temp_author{author_num}'] = text
    user_data.state = f'waiting_author{author_num}_name'

    markup = types.ForceReply(selective=False)
    bot.reply_to(message,
                 f"📚 *Шаг {author_num}.{author_num if author_num == 1 else 2} из 3*\n"
                 f"Книга {author_num} загружена!\n\n"
                 f"Теперь введите имя автора этой книги:\n"
                 f"Пример: *Лев Толстой* или *Федор Достоевский*",
                 parse_mode='Markdown',
                 reply_markup=markup)


def handle_anonymous_book(message, text, user_data):
    """
    Обработка анонимной книги.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    :param text: Текст анонимной книги.
    :type text: str
    :param user_data: Данные пользователя.
    :type user_data: UserData
    """
    user_data.anonymous_book = text
    user_data.books_received += 1

    show_authorship_methods_menu(message, user_data)


def show_authorship_methods_menu(message, user_data):
    """
    Показывает меню выбора метода анализа авторства.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    :param user_data: Данные пользователя.
    :type user_data: UserData
    """
    user_data.state = 'choose_authorship_method'

    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn1 = types.KeyboardButton('📈 График длин слов')
    btn2 = types.KeyboardButton('📉 График предложений')
    btn3 = types.KeyboardButton('📝 Сравнение хи-квадрат')
    btn4 = types.KeyboardButton('🧮 Коэффициент Жаккара')
    btn5 = types.KeyboardButton('🏠 Главное меню')
    markup.add(btn1, btn2, btn3, btn4, btn5)

    bot.send_message(message.chat.id,
                     "✅ Все книги загружены!\n\n"
                     "📚 *Загруженные авторы:*\n" +
                     "\n".join([f"• {author}" for author in user_data.books.keys()]) +
                     "\n\n*Выберите метод анализа:*\n\n"
                     "1. 📈 *График длин слов* - сравнение распределения длин слов\n"
                     "2. 📉 *График предложений* - сравнение длин предложений\n"
                     "3. 📝 *Сравнение хи-квадрат* - статистический анализ лексикона\n"
                     "4. 🧮 *Коэффициент Жаккара* - схожесть словарного запаса",
                     parse_mode='Markdown',
                     reply_markup=markup)


@bot.message_handler(func=lambda message: message.text == '📈 График длин слов')
def handle_word_length_analysis(message):
    """
    Обработчик для анализа длин слов.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        return

    user_data = user_sessions[user_id]

    if len(user_data.books) < 2 or user_data.anonymous_book is None:
        bot.reply_to(message, "❌ Не хватает данных для анализа")
        return

    bot.send_message(message.chat.id, "🔄 Создаю график сравнения длин слов...")

    try:
        all_books = user_data.books.copy()
        all_books['Анонимная книга'] = user_data.anonymous_book

        img_buffer = plot_word_length_comparison(all_books)
        bot.send_photo(message.chat.id, img_buffer,
                       caption="📈 Сравнение распределения длин слов\n"
                               "*Анонимная книга* сравнивается с известными авторами")

        offer_next_analysis(message.chat.id, user_data)

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка при создании графика: {str(e)}")


@bot.message_handler(func=lambda message: message.text == '📉 График предложений')
def handle_sentence_length_analysis(message):
    """
    Обработчик для анализа длин предложений.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        return

    user_data = user_sessions[user_id]

    if len(user_data.books) < 2 or user_data.anonymous_book is None:
        bot.reply_to(message, "❌ Не хватает данных для анализа")
        return

    bot.send_message(message.chat.id, "🔄 Создаю график сравнения длин предложений...")

    try:
        all_books = user_data.books.copy()
        all_books['Анонимная книга'] = user_data.anonymous_book

        img_buffer = plot_sentence_length_comparison(all_books)
        bot.send_photo(message.chat.id, img_buffer,
                       caption="📉 Сравнение распределения длин предложений\n"
                               "*Анонимная книга* сравнивается с известными авторами")

        offer_next_analysis(message.chat.id, user_data)

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка при создании графика: {str(e)}")


@bot.message_handler(func=lambda message: message.text == '📊 Сравнение хи-квадрат')
def handle_chi_square_analysis_new(message):
    """
    Обработчик для анализа хи-квадрат.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        return

    user_data = user_sessions[user_id]

    if len(user_data.books) < 2 or user_data.anonymous_book is None:
        bot.reply_to(message, "❌ Не хватает данных для анализа")
        return

    bot.send_message(message.chat.id, "🔄 Провожу хи-квадрат тест...")

    try:
        words_by_author = prepare_words_by_author(user_data)

        results, chisquared_by_author = chi_square_test(words_by_author)

        report = "*РЕЗУЛЬТАТЫ ХИ-КВАДРАТ ТЕСТА*\n\n"
        report += "\n".join(results)

        bot.send_message(message.chat.id, report, parse_mode='Markdown')

        offer_next_analysis(message.chat.id, user_data)

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка при анализе: {str(e)}")


@bot.message_handler(func=lambda message: message.text == '🔗 Коэффициент Жаккара')
def handle_jaccard_analysis_new(message):
    """
    Обработчик для анализа коэффициента Жаккара.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        return

    user_data = user_sessions[user_id]

    if len(user_data.books) < 2 or user_data.anonymous_book is None:
        bot.reply_to(message, "❌ Не хватает данных для анализа")
        return

    bot.send_message(message.chat.id, "🔄 Вычисляю коэффициенты Жаккара...")

    try:
        words_by_author = prepare_words_by_author(user_data)

        results, jaccard_by_author = jaccard_test(words_by_author)

        report = "*РЕЗУЛЬТАТЫ КОЭФФИЦИЕНТА ЖАККАРА*\n\n"
        report += "\n".join(results)

        bot.send_message(message.chat.id, report, parse_mode='Markdown')

        offer_next_analysis(message.chat.id, user_data)

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка при анализе: {str(e)}")


@bot.message_handler(func=lambda message: message.text == '🏠 Главное меню')
def handle_back_to_main(message):
    """
    Обработчик возврата в главное меню.

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id in user_sessions:
        user_data = user_sessions[user_id]
        return_to_main_menu(message.chat.id, user_data)


def offer_next_analysis(chat_id, user_data):
    """
    Предложение выбора следующего метода анализа.

    :param chat_id: ID чата.
    :type chat_id: int
    :param user_data: Данные пользователя.
    :type user_data: UserData
    """
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn1 = types.KeyboardButton('📈 График длин слов')
    btn2 = types.KeyboardButton('📉 График предложений')
    btn3 = types.KeyboardButton('📊 Сравнение хи-квадрат')
    btn4 = types.KeyboardButton('🔗 Коэффициент Жаккара')
    btn5 = types.KeyboardButton('🏠 Главное меню')
    markup.add(btn1, btn2, btn3, btn4, btn5)

    bot.send_message(chat_id,
                     "📚 *Выберите следующий метод анализа:*\n\n"
                     "Или вернитесь в главное меню",
                     parse_mode='Markdown',
                     reply_markup=markup)


@bot.message_handler(func=lambda message: True)
def handle_text_messages(message):
    """
    Обработка текстовых сообщений (имен авторов, персонажей и т.д.).

    :param message: Сообщение от пользователя.
    :type message: telebot.types.Message
    """
    user_id = message.from_user.id
    if user_id not in user_sessions:
        return

    user_data = user_sessions[user_id]
    text = message.text.strip()

    if user_data.state == 'waiting_author1_name':
        author1_name = text
        if 'temp_author1' in user_data.books:
            user_data.books[author1_name] = user_data.books.pop('temp_author1')
            user_data.books_received += 1

        user_data.state = 'waiting_author2'

        markup = types.ForceReply(selective=False)
        bot.reply_to(message,
                     f"✅ Автор сохранен: *{author1_name}*\n\n"
                     "📚 *Шаг 2 из 3*\n"
                     "Теперь отправьте *вторую книгу* с известным автором.\n"
                     "После отправки файла, введите имя автора этой книги.",
                     parse_mode='Markdown',
                     reply_markup=markup)

    # Обработка имени автора 2
    elif user_data.state == 'waiting_author2_name':
        author2_name = text
        if 'temp_author2' in user_data.books:
            user_data.books[author2_name] = user_data.books.pop('temp_author2')
            user_data.books_received += 1

        user_data.state = 'waiting_anonymous_book'

        bot.reply_to(message,
                     f"✅ Автор сохранен: *{author2_name}*\n\n"
                     "📚 *Шаг 3 из 3*\n"
                     "Теперь отправьте *анонимную книгу*, автора которой нужно определить.\n"
                     "Эта книга должна быть написана одним из двух указанных авторов.",
                     parse_mode='Markdown')

    elif user_data.state == 'waiting_characters_list':
        characters = [char.strip() for char in text.split(',')]
        characters = [char for char in characters if char]

        if len(characters) == 0:
            bot.reply_to(message, "❌ Не указаны имена персонажей. Попробуйте снова.")
            return

        if len(characters) > 10:
            characters = characters[:10]
            bot.send_message(message.chat.id, f"⚠️ Ограничено 10 персонажами")

        user_data.characters = characters

        bot.send_message(message.chat.id, "🔄 Строю график появления персонажей...")

        try:
            img_buffer = plot_character_dispersion(
                user_data.books['book1'],
                characters
            )

            caption = f"👥 График персонажей:\n" + ", ".join(characters[:5])
            if len(characters) > 5:
                caption += f" и ещё {len(characters) - 5}"
            caption += "\n\n📊 *Пояснение:*\n• Каждая вертикальная черта - появление персонажа в тексте\n• По горизонтали - позиция в тексте (номер слова)\n• По вертикали - имена персонажей"

            bot.send_photo(message.chat.id, img_buffer, caption=caption, parse_mode='Markdown')

            return_to_main_menu(message.chat.id, user_data)

        except Exception as e:
            bot.reply_to(message, f"❌ Ошибка при построении графика: {str(e)}")

    else:
        bot.reply_to(message, "Используйте кнопки меню или выберите действие через /start")


def return_to_main_menu(chat_id, user_data):
    """
    Возвращает пользователя в главное меню и сбрасывает состояние.

    :param chat_id: ID чата.
    :type chat_id: int
    :param user_data: Данные пользователя.
    :type user_data: UserData
    """
    user_data.state = 'main_menu'
    user_data.analysis_type = None
    user_data.books.clear()
    user_data.characters = []
    user_data.anonymous_book = None
    user_data.expected_books_count = 0
    user_data.books_received = 0
    user_data.authorship_method = None

    markup = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    btn1 = types.KeyboardButton('☁️ Облако слов')
    btn2 = types.KeyboardButton('👥 График персонажей')
    btn3 = types.KeyboardButton('🔍 Анализ авторства')
    markup.add(btn1, btn2, btn3)

    bot.send_message(chat_id,
                     "🏠 Возвращаю в главное меню\n\n"
                     "Выберите тип анализа:",
                     reply_markup=markup)


if __name__ == '__main__':
    print("Бот запущен...")
    bot.infinity_polling()
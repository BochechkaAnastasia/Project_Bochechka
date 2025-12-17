import unittest
import re
import nltk
import pymorphy3
from collections import Counter
import math
import random
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
morph = pymorphy3.MorphAnalyzer()
from book_analyzer import (
    UserData,
    preprocess_text,
    get_word_length_frequencies,
    get_sentence_length_frequencies,
    prepare_words_by_author,
    chi_square_test,
    jaccard_test,
    get_random_color
)

class TestPreprocessText(unittest.TestCase):

    def test_1(self):
        self.assertEqual(preprocess_text(""), [])

    def test_2(self):
        text = "и в на с по из"
        self.assertEqual(preprocess_text(text), [])

    def test_3(self):
        text = "Кошка сидит на столе. Стол деревянный."
        result = preprocess_text(text)
        expected = ['кошка', 'сидеть', 'стол', 'стол', 'деревянный']
        self.assertEqual(result, expected)

    def test_4(self):
        text = "Привет, мир! Как дела?"
        result = preprocess_text(text)
        self.assertNotIn(',', ' '.join(result))
        self.assertNotIn('!', ' '.join(result))
        self.assertNotIn('?', ' '.join(result))

    def test_5(self):
        text1 = "КОШКА СИДИТ"
        text2 = "кошка сидит"
        text3 = "Кошка Сидит"
        result1 = preprocess_text(text1)
        result2 = preprocess_text(text2)
        result3 = preprocess_text(text3)
        self.assertEqual(result1, result2)
        self.assertEqual(result1, result3)

    def test_6(self):
        text = "я в и но или а она он"
        result = preprocess_text(text)
        self.assertEqual(result, [])

    def test_7(self):
        text = "Hello мир! 123 числа и слова."
        result = preprocess_text(text)
        self.assertEqual(result, ['hello', 'мир', '123', 'число', 'слово'])


class TestGetWordLengthFrequencies(unittest.TestCase):

    def test_1(self):
        self.assertEqual(get_word_length_frequencies(""), {})

    def test_2(self):
        text = "кот пес слон"
        result = get_word_length_frequencies(text)
        expected = {3: 2, 4: 1}
        self.assertEqual(result, expected)

    def test_3(self):
        text = "Привет, мир! Как дела?"
        result = get_word_length_frequencies(text)
        expected = {3: 2, 4: 1, 6: 1}
        self.assertEqual(result, expected)

    def test_4(self):
        text1 = "КОТ ПЕС"
        text2 = "кот пес"
        result1 = get_word_length_frequencies(text1)
        result2 = get_word_length_frequencies(text2)
        self.assertEqual(result1, result2)

    def test_5(self):
        text = "123 4567 89"
        result = get_word_length_frequencies(text)
        expected = {2: 1, 3: 1, 4: 1}
        self.assertEqual(result, expected)

    def test_6(self):
        text = "а " * 100 + "бб" * 50 + " " + "ввв " * 30
        result = get_word_length_frequencies(text)
        expected = {1: 100, 3: 30, 100: 1}
        self.assertEqual(result, expected)

    def test_7(self):
        text = "word-word word'word word_word"
        result = get_word_length_frequencies(text)
        self.assertEqual(result, {4: 4, 9: 1})


class TestGetSentenceLengthFrequencies(unittest.TestCase):

    def test_1(self):
        self.assertEqual(get_sentence_length_frequencies(""), {})

    def test_2(self):
        text = "Кот бежит. Собака спит."
        result = get_sentence_length_frequencies(text)
        expected = {2: 2}
        self.assertEqual(result, expected)

    def test_3(self):
        text = "А. Б В. Г Д Е."
        result = get_sentence_length_frequencies(text)
        expected = {1: 1, 2: 1, 3: 1}
        self.assertEqual(result, expected)

    def test_4(self):
        text = "Привет!!! Как дела?? Все хорошо."
        result = get_sentence_length_frequencies(text)
        expected = {1: 1, 2: 2}
        self.assertEqual(result, expected)

    def test_5(self):
        text = "Это один длинный текст без точек"
        result = get_sentence_length_frequencies(text)
        expected = {6: 1}
        self.assertEqual(result, expected)

    def test_6(self):
        text = "First sentence! Second sentence? Third sentence."
        result = get_sentence_length_frequencies(text)
        self.assertEqual(result, {2: 3})

    def test_7(self):
        text = "... ! ? .."
        result = get_sentence_length_frequencies(text)
        self.assertEqual(result, {})

    def test_8(self):
        text = "Слово. " * 100 + "Два слова. " * 50
        result = get_sentence_length_frequencies(text)
        expected = {1: 100, 2: 50}
        self.assertEqual(result, expected)


class TestPrepareWordsByAuthor(unittest.TestCase):

    def test_1(self):
        user_data = UserData()
        result = prepare_words_by_author(user_data)
        self.assertEqual(result, {})

    def test_2(self):
        user_data = UserData()
        user_data.books = {
            'Толстой': 'Война и мир начало',
            'Достоевский': 'Преступление и наказание'
        }
        result = prepare_words_by_author(user_data)
        self.assertIn('Толстой', result)
        self.assertIn('Достоевский', result)
        self.assertNotIn('unknown', result)

        for author, words in result.items():
            self.assertIsInstance(words, list)
            self.assertTrue(all(isinstance(word, str) for word in words))

    def test_3(self):
        user_data = UserData()
        user_data.books = {'Автор1': 'текст один'}
        user_data.anonymous_book = 'текст два'
        result = prepare_words_by_author(user_data)
        self.assertIn('Автор1', result)
        self.assertIn('unknown', result)
        self.assertEqual(len(result), 2)

    def test_4(self):
        user_data = UserData()
        long_text = 'слово ' * 30000
        user_data.books = {'Автор': long_text}
        result = prepare_words_by_author(user_data)
        self.assertLessEqual(len(result['Автор']), 20000)

    def test_5(self):
        user_data = UserData()
        user_data.books = {
            'Автор1': 'первый текст',
            'Автор2': 'второй текст',
            'Автор3': 'третий текст'
        }
        result = prepare_words_by_author(user_data)
        self.assertEqual(len(result), 3)
        for author in ['Автор1', 'Автор2', 'Автор3']:
            self.assertIn(author, result)


class TestChiSquareTest(unittest.TestCase):

    def test_1(self):
        result, chi_dict = chi_square_test({})
        self.assertEqual(result, [])
        self.assertEqual(chi_dict, {})

    def test_2(self):
        words_by_author = {'unknown': ['слово1', 'слово2']}
        result, chi_dict = chi_square_test(words_by_author)
        self.assertEqual(result, [])
        self.assertEqual(chi_dict, {})

    def test_3(self):
        words_by_author = {
            'Толстой': ['война', 'мир', 'мир', 'война'],
            'unknown': ['мир', 'война', 'текст']
        }
        result, chi_dict = chi_square_test(words_by_author)
        self.assertIn('Толстой', chi_dict)
        self.assertIsInstance(chi_dict['Толстой'], float)
        self.assertTrue(any('Толстой' in line for line in result))

    def test_4(self):
        words_by_author = {
            'Толстой': ['война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер', 'армия',
                        'история', 'народ', 'жизнь', 'семья', 'природа', 'крестьянин', 'дворянин',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер', 'армия',
                        'история', 'народ', 'жизнь', 'семья', 'природа', 'крестьянин', 'дворянин',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер', 'армия',
                        'история', 'народ', 'жизнь', 'семья', 'природа', 'крестьянин', 'дворянин',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер', 'армия',
                        'история', 'народ', 'жизнь', 'семья', 'природа', 'крестьянин', 'дворянин',
                        'свет', 'бог', 'душа', 'сердце', 'мысль', 'время', 'год', 'век', 'государство', 'общество',
                        'разум'],

            'Достоевский': ['преступление', 'наказание', 'человек', 'бог', 'страдание', 'грех',
                            'раскаяние', 'совесть', 'вина', 'смерть', 'жизнь', 'любовь', 'ненависть',
                            'страх', 'ужас', 'безумие', 'разум', 'душа', 'сердце', 'вера', 'безверие',
                            'преступление', 'наказание', 'человек', 'бог', 'страдание', 'грех',
                            'раскаяние', 'совесть', 'вина', 'смерть', 'жизнь', 'любовь', 'ненависть',
                            'страх', 'ужас', 'безумие', 'разум', 'душа', 'сердце', 'вера', 'безверие',
                            'преступление', 'наказание', 'человек', 'бог', 'страдание', 'грех',
                            'раскаяние', 'совесть', 'вина', 'смерть', 'жизнь', 'любовь', 'ненависть',
                            'страх', 'ужас', 'безумие', 'разум', 'душа', 'сердце', 'вера'],

            'unknown': ['война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер',
                        'преступление', 'наказание', 'бог', 'страдание', 'грех', 'раскаяние',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер',
                        'преступление', 'наказание', 'бог', 'страдание', 'грех', 'раскаяние',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер',
                        'преступление', 'наказание', 'бог', 'страдание', 'грех', 'раскаяние',
                        'армия', 'история', 'народ', 'жизнь', 'семья', 'природа', 'совесть',
                        'вина', 'страх', 'ужас', 'безумие', 'разум', 'душа', 'сердце', 'вера',
                        'свет', 'бог', 'душа', 'сердце', 'мысль', 'время', 'год', 'век',
                        'свобода', 'судьба', 'рок', 'идея', 'теория', 'философия', 'религия',
                        'государство', 'общество', 'разум', 'чувство', 'счастье', 'страдание']
        }
        result, chi_dict = chi_square_test(words_by_author)
        self.assertEqual(chi_dict, {'Толстой': 27.498591549295792, 'Достоевский': 24.666705069124397})

    def test_5(self):
        words_by_author= {
            'Shakespeare': ['love', 'death', 'heart', 'soul', 'king', 'queen', 'crown', 'throne',
                            'sword', 'blood', 'honor', 'betrayal', 'treason', 'passion', 'desire',
                            'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown', 'throne',
                            'sword', 'blood', 'honor', 'betrayal', 'treason', 'passion', 'desire',
                            'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown', 'throne',
                            'sword', 'blood', 'honor', 'betrayal', 'treason', 'passion', 'desire',
                            'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown', 'throne',
                            'sword', 'blood', 'honor', 'betrayal', 'treason', 'passion', 'desire',
                            'fate', 'destiny', 'fortune', 'tragedy', 'comedy', 'drama', 'scene',
                            'stage', 'actor', 'play', 'theater', 'audience', 'curtain', 'performance'],

            'Dickens': ['poverty', 'wealth', 'child', 'orphan', 'factory', 'workhouse', 'London',
                        'street', 'misery', 'happiness', 'family', 'inheritance', 'money', 'debt',
                        'poverty', 'wealth', 'child', 'orphan', 'factory', 'workhouse', 'London',
                        'street', 'misery', 'happiness', 'family', 'inheritance', 'money', 'debt',
                        'poverty', 'wealth', 'child', 'orphan', 'factory', 'workhouse', 'London',
                        'street', 'misery', 'happiness', 'family', 'inheritance', 'money', 'debt',
                        'poverty', 'wealth', 'child', 'orphan', 'factory', 'workhouse', 'London',
                        'street', 'misery', 'happiness', 'family', 'inheritance', 'money', 'debt',
                        'Victorian', 'society', 'class', 'gentleman', 'lady', 'servant', 'master',
                        'Christmas', 'ghost', 'spirit', 'redemption', 'transformation'],

            'unknown': ['love', 'death', 'heart', 'soul', 'king', 'queen', 'crown',
                        'poverty', 'wealth', 'child', 'orphan', 'London', 'street', 'misery',
                        'war', 'soldier', 'bullet', 'wound', 'courage', 'fear', 'bravery',
                        'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown',
                        'poverty', 'wealth', 'child', 'orphan', 'London', 'street', 'misery',
                        'war', 'soldier', 'bullet', 'wound', 'courage', 'fear', 'bravery',
                        'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown',
                        'poverty', 'wealth', 'child', 'orphan', 'London', 'street', 'misery',
                        'sword', 'blood', 'honor', 'betrayal', 'family', 'inheritance', 'money',
                        'alcohol', 'whiskey', 'drink', 'bar', 'woman', 'passion', 'affair',
                        'fate', 'destiny', 'society', 'class', 'Christmas', 'spirit', 'redemption',
                        'hunt', 'fish', 'bull', 'matador', 'Spain', 'Paris', 'expatriate']
        }
        result, chi_dict = chi_square_test(words_by_author)
        self.assertEqual(chi_dict, {'Shakespeare': 47.70270270270274, 'Dickens': 46.22352941176465})

    def test_6(self):
        words_by_author = {
            'Автор1': ['а', 'б', 'в'] * 10,
            'Автор2': ['г', 'д', 'е'] * 10,
            'unknown': ['а', 'б', 'в'] * 20
        }
        result, chi_dict = chi_square_test(words_by_author)
        self.assertEqual(chi_dict, {'Автор1': 0.0, 'Автор2': 60.00000000000001})


class TestJaccardTest(unittest.TestCase):

    def test_1(self):
        words_by_author = {'unknown': ['слово1', 'слово2']}
        result, jaccard_dict = jaccard_test(words_by_author)
        self.assertEqual(result, [])
        self.assertEqual(jaccard_dict, {})

    def test_2(self):
        words_by_author = {
            'Толстой': ['война', 'мир', 'любовь'],
            'unknown': ['мир', 'война', 'текст']
        }
        result, jaccard_dict = jaccard_test(words_by_author)
        self.assertIn('Толстой', jaccard_dict)
        self.assertIsInstance(jaccard_dict['Толстой'], float)
        self.assertTrue(any('Толстой' in line for line in result))

    def test_3(self):
        words_by_author = {
            'Автор1': ['слово']*100,
            'Автор2': ['слово']*200,
            'unknown': ['слово']*100
        }
        result, jaccard_dict = jaccard_test(words_by_author)
        self.assertEqual(jaccard_dict, {'Автор1': 1.0, 'Автор2': 1.0})

    def test_4(self):
        words_by_author = {
            'Автор1': ['а', 'б', 'в'],
            'Автор2': ['г', 'д', 'е'],
            'unknown': ['ж', 'з', 'и']
        }
        result, jaccard_dict = jaccard_test(words_by_author)
        self.assertEqual(jaccard_dict, {'Автор1': 0.0, 'Автор2': 0.0})

    def test_5(self):
        words_by_author = {
            'Толстой': ['война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер', 'армия',
                        'история', 'народ', 'жизнь', 'семья', 'природа', 'крестьянин', 'дворянин',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер', 'армия',
                        'история', 'народ', 'жизнь', 'семья', 'природа', 'крестьянин', 'дворянин',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер', 'армия',
                        'история', 'народ', 'жизнь', 'семья', 'природа', 'крестьянин', 'дворянин',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер', 'армия',
                        'история', 'народ', 'жизнь', 'семья', 'природа', 'крестьянин', 'дворянин',
                        'свет', 'бог', 'душа', 'сердце', 'мысль', 'время', 'год', 'век', 'государство', 'общество',
                        'разум'],

            'Достоевский': ['преступление', 'наказание', 'человек', 'бог', 'страдание', 'грех',
                            'раскаяние', 'совесть', 'вина', 'смерть', 'жизнь', 'любовь', 'ненависть',
                            'страх', 'ужас', 'безумие', 'разум', 'душа', 'сердце', 'вера', 'безверие',
                            'преступление', 'наказание', 'человек', 'бог', 'страдание', 'грех',
                            'раскаяние', 'совесть', 'вина', 'смерть', 'жизнь', 'любовь', 'ненависть',
                            'страх', 'ужас', 'безумие', 'разум', 'душа', 'сердце', 'вера', 'безверие',
                            'преступление', 'наказание', 'человек', 'бог', 'страдание', 'грех',
                            'раскаяние', 'совесть', 'вина', 'смерть', 'жизнь', 'любовь', 'ненависть',
                            'страх', 'ужас', 'безумие', 'разум', 'душа', 'сердце', 'вера'],

            'unknown': ['война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер',
                        'преступление', 'наказание', 'бог', 'страдание', 'грех', 'раскаяние',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер',
                        'преступление', 'наказание', 'бог', 'страдание', 'грех', 'раскаяние',
                        'война', 'мир', 'человек', 'любовь', 'смерть', 'русский', 'офицер',
                        'преступление', 'наказание', 'бог', 'страдание', 'грех', 'раскаяние',
                        'армия', 'история', 'народ', 'жизнь', 'семья', 'природа', 'совесть',
                        'вина', 'страх', 'ужас', 'безумие', 'разум', 'душа', 'сердце', 'вера',
                        'свет', 'бог', 'душа', 'сердце', 'мысль', 'время', 'год', 'век',
                        'свобода', 'судьба', 'рок', 'идея', 'теория', 'философия', 'религия',
                        'государство', 'общество', 'разум', 'чувство', 'счастье', 'страдание']
        }
        result, jaccard_dict = jaccard_test(words_by_author)
        self.assertEqual(jaccard_dict, {'Толстой': 0.5217391304347826, 'Достоевский': 0.41304347826086957})

    def test_6(self):
        words_by_author = {
            'Shakespeare': ['love', 'death', 'heart', 'soul', 'king', 'queen', 'crown', 'throne',
                            'sword', 'blood', 'honor', 'betrayal', 'treason', 'passion', 'desire',
                            'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown', 'throne',
                            'sword', 'blood', 'honor', 'betrayal', 'treason', 'passion', 'desire',
                            'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown', 'throne',
                            'sword', 'blood', 'honor', 'betrayal', 'treason', 'passion', 'desire',
                            'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown', 'throne',
                            'sword', 'blood', 'honor', 'betrayal', 'treason', 'passion', 'desire',
                            'fate', 'destiny', 'fortune', 'tragedy', 'comedy', 'drama', 'scene',
                            'stage', 'actor', 'play', 'theater', 'audience', 'curtain', 'performance'],

            'Dickens': ['poverty', 'wealth', 'child', 'orphan', 'factory', 'workhouse', 'London',
                        'street', 'misery', 'happiness', 'family', 'inheritance', 'money', 'debt',
                        'poverty', 'wealth', 'child', 'orphan', 'factory', 'workhouse', 'London',
                        'street', 'misery', 'happiness', 'family', 'inheritance', 'money', 'debt',
                        'poverty', 'wealth', 'child', 'orphan', 'factory', 'workhouse', 'London',
                        'street', 'misery', 'happiness', 'family', 'inheritance', 'money', 'debt',
                        'poverty', 'wealth', 'child', 'orphan', 'factory', 'workhouse', 'London',
                        'street', 'misery', 'happiness', 'family', 'inheritance', 'money', 'debt',
                        'Victorian', 'society', 'class', 'gentleman', 'lady', 'servant', 'master',
                        'Christmas', 'ghost', 'spirit', 'redemption', 'transformation'],

            'unknown': ['love', 'death', 'heart', 'soul', 'king', 'queen', 'crown',
                        'poverty', 'wealth', 'child', 'orphan', 'London', 'street', 'misery',
                        'war', 'soldier', 'bullet', 'wound', 'courage', 'fear', 'bravery',
                        'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown',
                        'poverty', 'wealth', 'child', 'orphan', 'London', 'street', 'misery',
                        'war', 'soldier', 'bullet', 'wound', 'courage', 'fear', 'bravery',
                        'love', 'death', 'heart', 'soul', 'king', 'queen', 'crown',
                        'poverty', 'wealth', 'child', 'orphan', 'London', 'street', 'misery',
                        'sword', 'blood', 'honor', 'betrayal', 'family', 'inheritance', 'money',
                        'alcohol', 'whiskey', 'drink', 'bar', 'woman', 'passion', 'affair',
                        'fate', 'destiny', 'society', 'class', 'Christmas', 'spirit', 'redemption',
                        'hunt', 'fish', 'bull', 'matador', 'Spain', 'Paris', 'expatriate']
        }
        result, jaccard_dict = jaccard_test(words_by_author)
        self.assertEqual(jaccard_dict, {'Shakespeare': 0.21875, 'Dickens': 0.25})


if __name__ == "__main__":
    unittest.main()
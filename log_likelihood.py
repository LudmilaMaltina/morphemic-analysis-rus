import string
import nltk
import math
import pymorphy2

with open('Морфемно-орфографический словарь А. Н. Тихонова.txt', encoding='windows-1251') as f:
    morphs = set()
    morphs_positions = []
    for line in f:
        line = line.split(' | ')[1].strip().split()[0]
        if line[-1] in string.punctuation:
            line = line[:-1]
        if line[-1].isdigit():
            line = line[:-1]
        line = line.replace("'", '').split('/')
        for position, morph in enumerate(line):
            if morph.startswith('-'):
                morph = morph[1:]
            if morph != '':
                morphs.add(morph)
                morphs_positions.append(morph + '-' + str(position))

total = len(morphs_positions)
morphs_positions_freq = {}

for morph_position in morphs_positions:
    if morph_position not in morphs_positions_freq:
        morphs_positions_freq[morph_position] = 1
    else:
        morphs_positions_freq[morph_position] += 1
morphs_positions_freq = {x: y/total for x, y in morphs_positions_freq.items()}


def maxmatch(s):
    i = 0
    j = len(s)
    word_parts = []
    while True:
        if s[i:j] in morphs:
            word_parts.append(s[i:j])
            i = j
            j = len(s)
        else:
            j -= 1
        if i == j:
            break
    return '/'.join(word_parts)


def all_splits(word):
    len_reduced = len(word) - 1
    n = 2 ** len_reduced
    splits = []
    for i in range(n):
        mask = str(bin(i))[2:]  # убрать 0b в начале
        mask = '0' * (len_reduced - len(mask)) + mask
        s = ''
        for j, m in enumerate(mask):
            s += word[j]
            if m == '1':
                s += '/'
        s += word[-1]
        splits.append(s)
    return splits


def possible_splits(word):
    splits = []
    for s in all_splits(word):
        s = s.split('/')
        if all(part in morphs for part in s):
            splits.append('/'.join(s))
    return splits


def most_probable_variant(word):
    probability = 0
    variant_probability = {}
    possible_results = possible_splits(word)
    for variant in possible_results:
        variant_str = variant
        variant = variant.split('/')
        for ind in range(len(variant)):
            variant[ind] = variant[ind] + '-' + str(ind)
            if variant[ind] not in morphs_positions_freq.keys():
                morphs_positions_freq[variant[ind]] = 1/total
            probability += math.log(morphs_positions_freq[variant[ind]])
        variant_probability[variant_str] = probability
    variant_probable = sorted(variant_probability.items(), key=lambda x: x[1], reverse=True)[0][0]
    return variant_probable

word = ''
while True:
    word_or_text = input('Выберите режим работы: слово (с) или текст (т) (для выхода наберите "exit"): ')
    if word_or_text == 'exit':
        break
    if word_or_text == 'с':
        word = input('Введите слово: ')
        if len(word) < 19:
            print(most_probable_variant(word))
        else:
            print(maxmatch(word))
    elif word_or_text == 'т':
        file_name = input('Укажите название файла: ')
        morph = pymorphy2.MorphAnalyzer()
        with open(file_name, encoding='utf-8') as f:
            text = f.read().lower()
            text_tokenized = nltk.word_tokenize(text)
            new_text = ''
            for t in text_tokenized:
                p_t = morph.parse(t)[0]
                if t.isalpha() and p_t.tag.POS not in {'PREP', 'CONJ', 'PRCL', "INTJ"}:
                    if len(t) < 19:
                        word_parts = most_probable_variant(t)
                    else:
                        word_parts = maxmatch(t)
                    new_text += t + ' | ' + word_parts + '\n'
        annotation_file = input('Введите название файла, в который Вам нужно сохранить результаты аннотирования: ')
        with open(annotation_file, 'w', encoding='utf-8') as f:
            f.write(new_text[:-1])
            print('В файл {} были записаны результаты аннотирования.'.format(annotation_file))

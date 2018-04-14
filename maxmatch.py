import string
import nltk
import pymorphy2

with open('Морфемно-орфографический словарь А. Н. Тихонова.txt', encoding='windows-1251') as f:
    morphs = []
    for line in f:
        line = line.split(' | ')[1].strip().split()[0]
        if line[-1] in string.punctuation:
            line = line[:-1]
        if line[-1].isdigit():
            line = line[:-1]
        line = line.replace("'", '').split('/')
        for morph in line:
            if morph.startswith('-'):
                morph = morph[1:]
            if morph != '':
                morphs.append(morph)



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

word = ''
while True:
    word_or_text = input('Выберите режим работы: слово (с) или текст (т) (для выхода наберите "exit"): ')
    if word_or_text == 'exit':
        break
    if word_or_text == 'с':
        word = input('Введите слово: ')
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
                    word_parts = maxmatch(t)
                    new_text += t + ' | ' + word_parts + '\n'
        annotation_file = input('Введите название файла, в который Вам нужно сохранить результаты аннотирования: ')
        with open(annotation_file, 'w', encoding='utf-8') as f:
            f.write(new_text[:-1])
            print('В файл {} были записаны результаты аннотирования.'.format(annotation_file))

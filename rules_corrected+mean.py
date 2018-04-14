#coding=utf-8
import pymorphy2
import nltk
import string

postfixes_with_hyphen = ['-либо', '-нибудь', '-то']
postfixes_reflective = ['ся', 'сь']
suffixes_infinitive = ['ти', 'ть']
suffixes_gerund = ['a', 'я', 'вши', 'в', 'учи', 'ючи']
suffixes_comparative = ['ее', 'ей']
suffixes_adverb = ['а', 'о', 'у']
suffixes_superlative = ['ейш', 'айш']
suffixes_participle = ['ущ', 'ющ', 'ащ', 'ящ', 'вш', 'ш', 'енн', 'ённ', 'нн', 'т', 'им', 'ем', 'ом']
suffix_past_tense = ['л']
ending_plur_impr = ['те']
suffix_plur_impr1 = ['и']
suffix_plur_impr2 = ['й']
endings_adjs_and_prts = ['а', 'о', 'и', 'ы']
suffixes_prts = ['ен', 'н', 'т']
prefixes = ['экстра', 'экс', 'ультра', 'у', 'транс', 'супер', 'суб', 'среди', 'со', 'сверх', 'с', 'ре', 'разо', 'раз',
            'прото', 'противо', 'про', 'при', 'предо', 'преди', 'пред', 'пре', 'пра', 'пост', 'после', 'подо', 'под',
            'по', 'перед', 'пере', 'ото', 'от', 'около', 'обо', 'обер-', 'об', 'о', 'низо', 'низ', 'ни', 'не', 'наи',
            'надо', 'над', 'на', 'между', 'меж', 'контр', 'кой-', 'кое-', 'ис', 'ир', 'интер', 'им', 'изо', 'из', 'за',
            'до', 'дис', 'дез', 'де', 'гипер', 'вы', 'вс', 'возо', 'воз', 'во', 'внутри', 'вне', 'вице-', 'взо', 'вз',
            'в', 'бес', 'без', 'архи', 'анти', 'а']
repeated_parts = ['яхт', 'ярко-', 'энерго', 'электро', 'эвако', 'штаб-', 'шеф-', 'чудо-', 'четверть-', 'царь-', 'хроно',
                  'хромо', 'хладо', 'франко', 'фото', 'фоно', 'флаг', 'физио', 'удобо', 'тёмно-', 'турбо', 'троллей',
                  'траги', 'типо', 'техно', 'термо', 'теле', 'строй', 'страто', 'стоп-', 'стерео', 'социо', 'сельхоз',
                  'сейсмо', 'светло-', 'свеже', 'санти', 'само', 'сам-', 'русо', 'радио', 'психо', 'псевдо', 'пресс-',
                  'полу', 'полит', 'поли', 'пол', 'пневмо', 'плащ-', 'пиро', 'перво', 'пан', 'органо', 'орг', 'обще',
                  'обоюдо', 'ново', 'нитро', 'ниже', 'неудобо', 'нео', 'нейро', 'невро', 'нарко', 'мото', 'мос', 'моно',
                  'молодо', 'много', 'младо', 'мини', 'милли', 'микро', 'механо', 'метео', 'место', 'мега', 'марш',
                  'мало', 'макси', 'макро', 'лиро-', 'лже', 'лейт-', 'кримино', 'крекинг', 'космо', 'кино', 'кило',
                  'квази', 'кардио', 'кабель', 'итало', 'инако', 'иммуно', 'изо', 'зоо', 'зауряд', 'зав', 'еже',
                  'досто', 'дизель', 'диапо', 'джаз', 'деци', 'дендро', 'греко', 'графо', 'градо', 'гос', 'горе-',
                  'глубоко', 'глав', 'гипро', 'гидро', 'гигро', 'гермо', 'германо', 'гео', 'гелио', 'гальвано', 'выше',
                  'высоко', 'все', 'видео', 'вибро', 'взаимо', 'верто', 'вело', 'вакуум', 'борт', 'блок', 'блиц',
                  'бледно', 'био', 'библио', 'бензо', 'баро', 'аэро', 'афро', 'астро', 'арифмо', 'антропо', 'англо',
                  'анархо', 'аллерго', 'аква', 'агро', 'агло', 'агит', 'авто', 'авиа', 'НИИ']
alternations = {'е': {'ь', 'ё'}, 'ь': {'е', 'ё'}, 'ё': {'ь', 'е'}, 'х': 'ш', 'ш': ['х', 'c'], 'з': ['ж', 'г'],
                'ж': ['з', 'г', 'д'], 'с': 'ш', 'г': ['ж', 'з'], 'к': ['ц', 'ч'], 'ц': ['к', 'ч'], 'ч': ['к', 'ц', 'т'],
                'т': ['ч', 'щ'], 'щ': ['т', 'ч'], 'д': 'ж'}
alternations2 = {'б': 'бл', 'п': 'пл', 'в': 'вл', 'ф': 'фл', 'м': 'мл', 'ж': 'жд'}
with_ka = ['ай', 'а', 'ату', 'ау', 'баю', 'вы', 'вя', 'гав', 'гар', 'да', 'е', 'ё', 'жам', 'буль', 'кар', 'ква', 'кря',
           'курлы', 'мурлы', 'мяу', 'фыр', 'хихи', 'хны', 'хрю', 'чав', 'ши', 'шушу', 'и', 'усь', 'ой', 'о', 'хны',
           'фу', 'сюсю', 'тпру', 'трень', 'турлы', 'тюлюлю', 'тяв', 'улюлю', 'хмы', 'хрум', 'чиви', 'чили', 'чо', 'я',
           'ну']
suf_adv = ['а', 'ства', 'ка', 'яка', 'е', 'ое', 'ые', 'и', 'ки', 'ами', 'ками', 'ой', 'кой', 'ик', 'ок', 'як', 'ом',
           'ком', 'иком', 'ышком', 'няком', 'уном', 'ишом', 'ым', 'о', 'ко', 'енько', 'ошенько', 'онько', 'охонько',
           'но', 'овато', 'у', 'ку', 'еньку', 'оньку', 'ому', 'оту', 'ах', 'ках', 'их', 'ы', 'жды', 'ажды', 'ою', 'ую',
           'остью', 'мя']
suf_noun = ['б', 'об', 'ытьб', 'в', 'ев', 'ив', 'ив', 'ов', 'овь', 'тв', 'тв', 'ств', 'овств', 'еств', 'инств',
            'тельств', 'аг', 'инг', 'ург', 'уг', 'ыг', 'д', 'ад', 'ад', 'иад', 'арад', 'оид', 'ядь', 'аж', 'ёж', 'ёжь',
            'из', 'оз', 'атай', 'ей', 'алей', 'ачей', 'ий', 'арий', 'орий', 'тяй', 'к', 'к', 'к', 'к', 'ак', 'ак',
            'чак', 'авк', 'овк', 'ловк', 'анек', 'ышек', 'ежк', 'ик', 'ик', 'ик', 'евик', 'ник', 'овник', 'еник',
            'ейник', 'арник', 'атник', 'льник', 'истик', 'чик', 'щик', 'овщик', 'льщик', 'айк', 'ейк', 'лк', 'лк',
            'анк', 'инк', 'онк', 'унк', 'ок', 'онок', 'чонок', 'ушок', 'ерк', 'урк', 'етк', 'отк', 'ютк', 'ук', 'чук',
            'чк', 'ачк', 'ечк', 'ечк', 'ичка', 'очк', 'шк', 'шк', 'ашк', 'ёшк', 'ишк', 'ушк', 'ушк', 'ышк', 'ышк', 'ык',
            'ульк', 'оньк', 'юк', 'няк', 'ль', 'л', 'л', 'л', 'ал', 'аль', 'ёл', 'ель', 'ел', 'тель', 'итель', 'ил',
            'ол', 'оль', 'ол', 'ул', 'ул', 'ыль', 'изм', 'онизм', 'им', 'им', 'нь', 'н', 'н', 'н', 'н', 'н', 'ан', 'ан',
            'уган', 'иан', 'овиан', 'лан', 'ман', 'овн', 'ень', 'ень', 'ён', 'мен', 'смен', 'знь', 'изн', 'овизн', 'ин',
            'ин', 'ин', 'ин', 'бин', 'овин', 'лин', 'елин', 'анин', 'анин', 'жанин', 'ианин', 'чанин', 'овчанин',
            'ичанин', 'инчанин', 'тянин', 'итянин', 'атин', 'чин', 'щин', 'овщин', 'льщин', 'он', 'он', 'и снь',
            'снь', 'отн', 'ятн', 'ун', 'ун', 'иничн', 'ышн', 'льн', 'ынь', 'иян', 'ар', 'арь', 'ар', 'атарь', 'ер',
            'онер', 'мейстер', 'up', 'ор', 'вор', 'тор', 'атор', 'итор', 'ур', 'тур', 'amyp', 'итур', 'ырь', 'яр', 'с',
            'ис', 'анс', 'есс', 'ус', 'ариус', 'ть', 'am', 'ат', 'ат', 'иат', 'чат', 'евт', 'ет', 'ет', 'итет', 'ит',
            'ит', 'нит', 'инит', 'ант', 'ент', 'амент', 'емент', 'от', 'оть', 'от', 'иот', 'ист', 'ость', 'имость',
            'ность', 'нность', 'енность', 'тость', 'ут', 'х', 'ах', 'их', 'ох', 'ух', 'ух', 'ц', 'ц', 'ц', 'ец', 'ец',
            'ец', 'авец', 'овец', 'лец', 'омец', 'нец', 'енец', 'инец', 'иц', 'овиц', 'лиц', 'ниц', 'овниц', 'ениц',
            'атниц', 'униц', 'ичниц', 'очниц', 'ешниц', 'льниц', 'тельниц', 'льц', 'ч', 'ч', 'ач', 'ич', 'евич', 'ович',
            'ыч', 'аш', 'аш', 'аш', 'иш', 'ошь', 'ош', 'уш', 'оныш', 'ищ', 'ищ', 'бищ', 'овищ', 'лищ', 'еньк']
suf_num = ['ер', 'дцать', 'надцать']
suf_verb = ['а', 'ива', 'ова', 'ствова', 'ествова', 'изова', 'ирова', 'изирова', 'ка', 'ича', 'нича', 'е', 'и', 'ну',
            'ану', 'ыва', 'ева']
suf_adj = ['ав', 'ощав', 'ив', 'лив', 'овлив', 'елив', 'члив', 'чив', 'ов', 'ов', 'ляв', 'ий', 'овий', 'ачий', 'ичий',
           'енек', 'онек', 'ск', 'вск', 'евск', 'овск', 'еск', 'ческ', 'ическ', 'лезск', 'эзск', 'йск', 'ейск', 'ийск',
           'имск', 'нск', 'анск', 'ианск', 'енск', 'инск', 'унск', 'тельск', 'ацк', 'ецк', 'усеньк', 'ошеньк', 'оньк',
           'охоньк', 'як', 'л', 'овал', 'им', 'ом', 'н', 'н', 'ан', 'ебн', 'обн', 'евн', 'ивн', 'овн', 'ен', 'ён',
           'яжн', 'езн', 'озн', 'иозн', 'ин', 'ин', 'нин', 'йн', 'ейн', 'нн', 'анн', 'ованн', 'ированн', 'енн', 'ённ',
           'овенн', 'ственн', 'менн', 'онн', 'ионн', 'ационн', 'арн', 'орн', 'сн', 'ичн', 'очн', 'шн', 'шн', 'ашн',
           'ашн', 'ишн', 'льн', 'альн', 'идальн', 'иальн', 'ональн', 'уальн', 'ельн', 'абельн', 'ибельн', 'тельн',
           'ительн', 'ильн', 'т', 'ат', 'оват', 'чат', 'ит', 'овит', 'мент', 'аст', 'ист', 'ч', 'ач', 'нич', 'уч', 'ш',
           'ш', 'айш', 'ейш', 'ащ', 'ущ', 'еющ']
endings = ['а', 'о', 'я', 'е', 'ый', 'ий', 'ой']


def identify_postfix_ending_i_suffix(morphemes, label):
    for i in morphemes:
        if word_parts[0].endswith(i):
            morpheme = i
            ind = len(word_parts[0]) - len(i)
            rest = word_parts[0][:ind]
            labels.insert(1, label)
            if len(word_parts):
                word_parts[0] = rest
                word_parts.insert(1, morpheme)
    return word_parts, labels


def identify_ending(lst):
    go_further = False
    num_diff_letters = 0
    for sing_or_plur in {'sing', 'plur'}:
        for i in lst:
            if not p.inflect({sing_or_plur, i}):
                continue
            else:
                form = p.inflect({sing_or_plur, i})
                if form.word != word:
                    min_length = min((len(word), len(form.word)))
                    for i in range(min_length):
                        if word[i] == form.word[i]:
                            continue
                        elif go_further:
                            go_further = False
                            continue
                        elif word[i] in alternations.keys() and form.word[i] in alternations[word[i]]:
                            continue
                        elif word[i - 1] in alternations2.keys():
                            if word[(i - 1): (i + 1)] in alternations2[word[(i - 1)]]:
                                continue
                        if i < min_length - 1:
                            if word[i] == 'щ' and form.word[i: (i + 2)] in ['ст', 'ск']:
                                continue
                            elif word[i: (i + 2)] in ['ст', 'ск'] and form.word[i] == 'щ':
                                go_further = True
                                continue
                            elif word[i - 1] in 'бвгджзклмнпрстфхцчшщ' and word[i] in 'оеё' and word[
                                        i + 1] in 'бвгджзклмнпрстфхцчшщ':
                                if form.word[i] == word[i + 1]:
                                    go_further = True
                                    continue
                            elif word[i - 1] in 'бвгджзклмнпрстфхцчшщ' and word[i] in 'бвгджзклмнпрстфхцчшщ':
                                if form.word[i - 1] in 'бвгджзклмнпрстфхцчшщ' and form.word[i] in 'оеё' and form.word[
                                            i + 1] in 'бвгджзклмнпрстфхцчшщ':
                                    continue
                        num_new = len(word) - i
                        if num_new > num_diff_letters:
                            num_diff_letters = num_new
    if num_diff_letters == 0:
        before_ending = word
        ending = '☐'
    else:
        ending = word[(len(word) - num_diff_letters):]
        before_ending = word[:-num_diff_letters]
        if ending.startswith('ь'):
            ending = ending[1:]
            if len(ending) == 0:
                ending = '☐'
            before_ending = word[:-num_diff_letters] + 'ь'
    if len(word_parts):
        word_parts[0] = before_ending
        word_parts.insert(1, ending)
        labels.insert(1, 'окончание')
    else:
        word_parts.extend([before_ending, ending])
    return word_parts, labels


def insert_ending_or_i_suffix(morpheme, label):
    word_parts.insert(1, morpheme)
    labels.insert(1, label)
    return word_parts, labels


def identify_prefix(form_1, form_2, num):
    global forms
    global new_flag
    new_flag = False
    if len(word) > 3:
        for i in prefixes:
            if forms[num].startswith(i):
                for r in roots:
                    if forms[num].startswith(r) and len(r) > len(i) + 1:
                        break
                else:
                    forms[num + 1] = forms[num][len(i):]
                    if i != 'с' and ((i[-1] == 'с' and forms[num + 1][0] in 'аеёиоуыэюябвгджзйлмнр') or (
                                    i[-1] == 'з' and forms[num + 1][0] in 'кпстфхцчшщ')):
                        continue
                    if forms[num + 1] in words or (
                        forms[num + 1].startswith('ы') and ('и' + forms[num + 1][1:] in words)):
                        new_flag = True
                        rest = word_parts[num][len(i):]
                        if rest == '':
                            continue
                        prefix = i
                        if len(rest) > 1:
                            if rest[0] == 'ъ':
                                rest = rest[1:]
                                prefix = i + 'ъ'
                        word_parts[num] = rest
                        word_parts.insert(num, prefix)
                        labels.insert(num, 'префикс')
                    else:
                        new_flag = False
                        for w in words:
                            if w.endswith(forms[num + 1]) and w not in forms[:num + 1]:
                                for c in prefixes:
                                    if w.startswith(c):
                                        w_new = w[len(c):]
                                        if w_new == forms[num + 1]:
                                            rest = word_parts[num][len(i):]
                                            if rest == '':
                                                continue
                                            prefix = i
                                            if len(rest) > 1:
                                                if rest[0] == 'ъ':
                                                    rest = rest[1:]
                                                    prefix = i + 'ъ'
                                            word_parts[num] = rest
                                            word_parts.insert(num, prefix)
                                            labels.insert(num, 'префикс')
                                            new_flag = True
                                            break
                            if new_flag:
                                break
                return word_parts, labels



def identify_rep_part(form_1, form_2, num):
    global forms
    global new_flag
    new_flag = False
    for i in repeated_parts:
        if forms[num].startswith(i):
            forms[num + 1] = forms[num][len(i):]
            if forms[num + 1] in words:
                new_flag = True
                rest = word_parts[num][len(i):]
                if rest == '':
                    continue
                rep_part = i
                word_parts[num] = rest
                word_parts.insert(num, rep_part)
                labels.insert(num, 'повторяющийся компонент')
    return word_parts, labels


def prefix_and_suffix_simple(prefix, suffix):
    if word_without_ending.startswith(prefix) and word_without_ending.endswith(suffix):
        word_parts.insert(1, word_without_ending[len(prefix):(len(word_without_ending) - len(suffix))])
        word_parts[0] = prefix
        word_parts.insert(2, suffix)
        labels[0] = 'префикс'
        labels.insert(1, 'корень')
        labels.insert(2, 'суффикс')
        global mark
        mark = True
    return word_parts, labels


def prefix_and_suffix(prefix, suffix, new_ending):
    new_word, new_word2, new_word3 = '', '', ''
    if word_without_ending.startswith(prefix) and word_without_ending.endswith(suffix):
        for d in new_ending:
            global mark
            if not mark:
                part = word_without_ending[len(prefix):(len(word_without_ending) - len(suffix))]
                if 'ё' in part:
                    ind = part.index('ё')
                    part_new = part[:ind] + 'е' + part[ind + 1:]
                else:
                    part_new = part
                new_word3 = part_new + d
                if part_new.endswith('ч'):
                    new_word = part_new[:-1] + 'к' + d
                    new_word2 = part_new[:-1] + 'ц' + d
                elif part_new.endswith('ж'):
                    new_word = part_new[:-1] + 'г' + d
                elif part_new.endswith('ш'):
                    new_word = part_new[:-1] + 'х' + d
                if new_word3 in words or new_word in words or new_word2 in words:
                    word_parts.insert(1, part)
                    word_parts[0] = prefix
                    word_parts.insert(2, suffix)
                    labels[0] = 'префикс'
                    labels.insert(1, 'корень')
                    labels.insert(2, 'суффикс')
                    mark = True
    return word_parts, labels


def pref_or_rep_part(forms, function):
    global new_flag
    new_flag = False
    function(forms[0], forms[1], 0)
    for i in range(1, 3):
        if new_flag:
            function(forms[i], forms[i + 1], i)
    return word_parts, labels


def identify_prefix_simple(prefix):
    if word_without_ending.startswith(prefix) and 'префикс' not in labels:
        word_parts[0] = word_without_ending[len(prefix):]
        word_parts.insert(0, prefix)
        labels[0] = 'корень'
        labels.insert(0, 'префикс')
    return word_parts, labels


def identify_suffix_simple(suffix):
    global unknown
    if unknown.endswith(suffix):
        word_parts[i] = unknown[:-(len(suffix))]
        word_parts.insert(i + 1, suffix)
        labels[i] = 'корень?'
        labels.insert(i + 1, 'суффикс')
        unknown = word_parts[i]
    return word_parts, labels


def identify_suffix(suffix, add_part, list_exceptions, necessary_form, pos_needed):
    word_new, word_new2, word_new3 = '', '', ''
    global unknown
    if suffix not in word_parts and necessary_form.endswith(suffix):
        for m in add_part:
            if suffix not in word_parts:
                if necessary_form in list_exceptions:
                    identify_suffix_simple(suffix)
                else:
                    word_new3 = necessary_form[:-len(suffix)] + m
                    if necessary_form[:-len(suffix)].endswith('ч'):
                        word_new = necessary_form[:-(len(suffix) + 1)] + 'к' + m
                        word_new2 = necessary_form[:-(len(suffix) + 1)] + 'ц' + m
                    elif necessary_form[:-len(suffix)].endswith('ж'):
                        word_new = necessary_form[:-(len(suffix) + 1)] + 'г' + m
                    elif necessary_form[:-len(suffix)].endswith('ш'):
                        word_new = necessary_form[:-(len(suffix) + 1)] + 'х' + m
                    if word_new3 in words or word_new in words or word_new2 in words:
                        if word_new3 in words:
                            word_new = word_new3
                        elif word_new2 in words:
                            word_new = word_new2
                        elif word_new in words:
                            word_new = word_new
                    p_new = morph.parse(word_new)[0]
                    if p_new.tag.POS == pos_needed:
                        identify_suffix_simple(suffix)
    return word_parts, labels


def pref_rep_part_suff(prefix, rep_part, suffix, new_ending):
    new_word, new_word2, new_word3 = '', '', ''
    start_component = prefix + rep_part
    if word_without_ending.startswith(start_component) and word_without_ending.endswith(suffix):
        for d in new_ending:
            global mark
            if mark:
                part = word_without_ending[len(start_component):(len(word_without_ending) - len(suffix))]
                if 'ё' in part:
                    ind = part.index('ё')
                    part_new = part[:ind] + 'е' + part[ind + 1:]
                else:
                    part_new = part
                new_word3 = part_new + d
                if part_new.endswith('ч'):
                    new_word = part_new[:-1] + 'к' + d
                    new_word2 = part_new[:-1] + 'ц' + d
                elif part_new.endswith('ж'):
                    new_word = part_new[:-1] + 'г' + d
                elif part_new.endswith('ш'):
                    new_word = part_new[:-1] + 'х' + d
                if new_word3 in words or new_word in words or new_word2 in words:
                    word_parts[0] = part
                    labels[0] = 'корень'
                    word_parts.insert(1, suffix)
                    labels.insert(1, 'суффикс')
                    word_parts.insert(0, rep_part)
                    labels.insert(0, 'повторяющийся компонент')
                    word_parts.insert(0, prefix)
                    labels.insert(0, 'префикс')
                    mark = True
    return word_parts, labels


def identify_suffix_full(suffix, add_part, list_exeptions, pos_needed):
    identify_suffix(suffix, add_part, list_exeptions, word_without_ending, pos_needed)
    if suffix not in word_parts:
        identify_suffix(suffix, add_part, list_exeptions, unknown, pos_needed)
    return word_parts, labels


def func(pref):
    if word_without_ending.startswith(pref) and word_without_ending.endswith('и') and word_parts[-1][0] in 'еёиюя':
        prefix_and_suffix(pref, 'и', {'', 'ь', 'о', 'е', 'а', 'я'})
    elif word_without_ending.startswith(pref) and word_without_ending.endswith('ь') and word_parts[-1][0] in 'еёиюя':
        identify_prefix_simple(pref)
        global mark
        mark = True
    return word_parts, labels


def eni(necessary_form):
    for k in {'ени', 'ень'}:
        alts = {'ч': {'т', 'к'}, 'щ': {'т', 'ст'}, 'ж': {'д', 'з', 'г'}, 'жд': {'д', 'зд'}, 'ш': {'с'}, 'бл': {'б'},
                'вл': {'в'}, 'мл': {'м'}, 'пл': {'п'}, 'фл': {'ф'}}
        if k not in word_parts and necessary_form.endswith(k):
            word_new = necessary_form[:-3]
            if word_new.endswith('ов'):
                word_new = word_new[:-2] + 'уть'
                if word_new in words:
                    identify_suffix_simple(k)
                    if word_without_ending != 'вдохновени':
                        identify_suffix_simple('ов')
            for m in ['ить', 'ти', 'нуть', 'ать', 'еть', 'ь']:
                if k not in word_parts:
                    word_new = necessary_form[:-3] + m
                    if word_new in words:
                        identify_suffix_simple(k)
                    else:
                        for alt in alts:
                            if necessary_form[:-3].endswith(alt):
                                for var in alts[alt]:
                                    word_new = necessary_form[:-3][:-(len(alt))] + var + m
                                    if word_new in words:
                                        identify_suffix_simple(k)
    return word_parts, labels


def suff_nicha(necessary_form):
    if 'нича' not in word_parts and 'ича' not in word_parts and 'а' not in word_parts and word_without_ending.endswith(
            'нича'):
        word_new = necessary_form[:-4] + 'ник'
        if word_new in words:
            identify_suffix_simple('а')
            if unknown != 'плотнич':
                identify_suffix_simple('нич')
        else:
            word_new = necessary_form[:-4]
            if word_new in words:
                word_parts[i] = unknown[:-4]
                word_parts.insert(i + 1, 'нича')
            if 'нича' not in word_parts:
                for k in {'ый', 'ий', 'ой'}:
                    word_new = necessary_form[:-3] + k
                    if word_new in words:
                        word_parts[i] = unknown[:-3]
                        word_parts.insert(i + 1, 'ича')
    return word_parts, labels


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


def most_probable_variant_mean(word):
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
            probability += morphs_positions_freq[variant[ind]]
        probability = probability/len(variant)
        variant_probability[variant_str] = probability
    if variant_probability:
        variant_probable = sorted(variant_probability.items(), key=lambda x: x[1], reverse=True)[0][0]
        return variant_probable


with open('Морфемно-орфографический словарь А. Н. Тихонова.txt') as f:
    words = set()
    morphs = set()
    morphs_positions = []
    for line in f:
        for_morphs = line.split(' | ')[1].strip().split()[0]
        if for_morphs[-1] in string.punctuation:
            for_morphs = for_morphs[:-1]
        if for_morphs[-1].isdigit():
            for_morphs = for_morphs[:-1]
        for_morphs = for_morphs.replace("'", '').split('/')
        for position, morph in enumerate(for_morphs):
            if morph.startswith('-'):
                morph = morph[1:]
            if morph != '':
                morphs.add(morph)
                morphs_positions.append(morph + '-' + str(position))
        line = line.split(' | ')[0]
        if line[-1].isdigit():
            line = line[:-1]
        if line not in words:
            words.add(line)

    roots = []
    for k in morphs:
        if ((k not in prefixes) and (k not in suf_adv) and (k not in suf_noun) and (k not in suf_num)
            and (k not in suf_verb) and (k not in suf_adj) and (k not in postfixes_with_hyphen)
            and (k not in postfixes_reflective) and (k not in suffixes_infinitive) and (k not in suffixes_gerund)
            and (k not in suffixes_comparative) and (k not in suffixes_adverb) and (k not in suffixes_superlative)
            and (k not in suffixes_participle) and (k not in suffix_past_tense) and (k not in ending_plur_impr)
            and (k not in suffix_plur_impr1) and (k not in suffix_plur_impr2) and (k not in endings_adjs_and_prts)
            and (k not in suffixes_prts) and (k not in endings)):
            roots.append(k)

total = len(morphs_positions)
morphs_positions_freq = {}

for morph_position in morphs_positions:
    if morph_position not in morphs_positions_freq:
        morphs_positions_freq[morph_position] = 1
    else:
        morphs_positions_freq[morph_position] += 1
morphs_positions_freq = {x: y / total for x, y in morphs_positions_freq.items()}


def analysis():
    global word_parts
    global labels
    global unknown
    global word_without_ending
    global i
    global morph
    global p
    global word
    word_parts = [word]
    morph = pymorphy2.MorphAnalyzer()
    p = morph.parse(word)[0]
    labels = ['?']
    if p.tag.POS in {'NPRO', 'ADVB', 'ADJF'}:
        word_parts, labels = identify_postfix_ending_i_suffix(postfixes_with_hyphen, 'постфикс')
        word = word_parts[0]
    elif p.tag.POS == 'VERB' or p.tag.POS == 'INFN' or p.tag.POS == 'PRTF' or p.tag.POS == 'GRND':
        word_parts, labels = identify_postfix_ending_i_suffix(postfixes_reflective, 'постфикс')
        word = word_parts[0]

    p = morph.parse(word)[0]
    normal = p.normal_form

    if p.tag.POS == 'INFN':
        word_parts, labels = identify_postfix_ending_i_suffix(suffixes_infinitive, 'суффикс')
    elif p.tag.POS == 'GRND':
        word_parts, labels = identify_postfix_ending_i_suffix(suffixes_gerund, 'суффикс')
    elif p.tag.POS == 'COMP':
        word_parts, labels = identify_postfix_ending_i_suffix(suffixes_comparative, 'суффикс')
    elif p.tag.POS in {'ADVB', 'PRED'}:
        word_parts, labels = identify_postfix_ending_i_suffix(suffixes_adverb, 'суффикс')

    if p.tag.POS in {'NPRO', 'ADJF', 'NUMR', 'PRTF', 'NOUN'}:
        if 'Fixd' not in p.tag:  # Выделим окончание
            word_parts, labels = identify_ending(['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'voct', 'gen2', 'acc2', 'loc2'])
            if p.tag.POS in {'ADJF', 'ADVB'}:
                word_parts, labels = identify_postfix_ending_i_suffix(suffixes_superlative, 'суффикс')
            elif p.tag.POS == 'PRTF':
                word_parts, labels = identify_postfix_ending_i_suffix(suffixes_participle, 'суффикс')
    elif p.tag.POS == 'VERB':
        if 'impr' not in p.tag and 'past' not in p.tag:  # Выделим окончание
            word_parts, labels = identify_ending(['1per', '2per', '3per'])
        elif 'past' in p.tag:
            word_parts, labels = identify_ending(['masc', 'femn', 'neut'])
            word_parts, labels = identify_postfix_ending_i_suffix(suffix_past_tense, 'суффикс')
            if 'masc' in p.tag and word_parts[1] != 'л':
                word_parts, labels = insert_ending_or_i_suffix('∅', 'суффикс')
        elif 'impr' in p.tag:
            word_parts, labels = identify_postfix_ending_i_suffix(ending_plur_impr, 'окончание')
            if len(word_parts) == 1 or word_parts[1] != 'те':
                word_parts, labels = insert_ending_or_i_suffix('☐', 'окончание')
            identify_postfix_ending_i_suffix(suffix_plur_impr1, 'суффикс')
            if word_parts[1] != 'и':
                if word[-1] == 'й':
                    infinitive = p.normal_form
                    if infinitive.endswith('ся'):
                        infinitive = infinitive[:-2]
                    if infinitive.endswith('ть'):
                        infinitive = infinitive[:-2]
                    if infinitive[-1] in 'яюеё' and infinitive[-2] in 'аеёиоуыэюя':
                        word_parts, labels = insert_ending_or_i_suffix('∅', 'суффикс')
                    else:
                        word_parts, labels = identify_postfix_ending_i_suffix(suffix_plur_impr2, 'суффикс')

    elif p.tag.POS in {'ADJS', 'PRTS'}:
        word_parts, labels = identify_postfix_ending_i_suffix(endings_adjs_and_prts, 'окончание')
        if (len(word_parts) == 1) or (word_parts[1] not in endings_adjs_and_prts):
            word_parts, labels = insert_ending_or_i_suffix('☐', 'окончание')
        if p.tag.POS == 'PRTS':
            word_parts, labels = identify_postfix_ending_i_suffix(suffixes_prts, 'суффикс')

    if p.tag.POS == 'COMP':
        if word.endswith('ше'):
            check_endings = ['ий', 'ой']
            for i in check_endings:
                check = word[:-1] + 'ий'
                if check not in words:
                    continue
                else:
                    word_parts[0] = word[:-1]
                    word_parts, labels = insert_ending_or_i_suffix('е', 'суффикс')
                    break
            if len(word_parts) == 1:
                word_parts[0] = word[:-2]
                word_parts, labels = insert_ending_or_i_suffix('ше', 'суффикс')
        elif word.endswith('е'):
            word_parts[0] = word[:-1]
            word_parts, labels = insert_ending_or_i_suffix('е', 'суффикс')

    word_without_prefix, word_without_prefix1, word_without_prefix2 = '', '', ''
    word_without_rep_part, word_without_rep_part1, word_without_rep_part2 = '', '', ''

    word_without_ending = ''
    for num_part in range(len(word_parts)):
        if labels[num_part] != 'окончание' and not (p.tag.POS == 'VERB' and word_parts[num_part] == 'л') and word_parts[num_part] not in suffixes_infinitive and word_parts[num_part] not in suffixes_gerund and word_parts[num_part] not in suffixes_participle and word_parts[num_part] not in postfixes_reflective:
            word_without_ending += word_parts[num_part]

    mark = False
    if p.tag.POS == 'NOUN':
        if word_without_ending in {'безделушк', 'безделушек'}:
            word_parts, labels = prefix_and_suffix('без', 'ушк', {'о'})
            word_parts, labels = prefix_and_suffix('без', 'ушек', {'о'})
        elif word_without_ending in {'взморь', 'всполь', 'взгорь', 'всхолмь', 'взгорбь'}:
            for pref in {'вз', 'вс'}:  # 505
                word_parts, labels = func(pref)
        elif word_without_ending in {'взгорок', 'взгорк', 'взгорбок', 'взгорбк', 'взлобок', 'взлобк'}:  # 505
            word_parts, labels = prefix_and_suffix('вз', 'ок', {'', 'а'})
            word_parts, labels = prefix_and_suffix('вз', 'к', {'', 'а'})
        elif word_without_ending in {'зашеин', 'закраин'}:
            word_parts, labels = prefix_and_suffix('за', 'ин', {'я', 'й'})
        elif word_without_ending in {'закопёрщик'}:
            word_parts, labels = prefix_and_suffix('за', 'щик', {''})
        elif word_without_ending in {'закопёрщиц'}:
            word_parts, labels = prefix_and_suffix('за', 'щиц', {''})
        elif word_without_ending in {'нагорь', 'надворь'}:  # 508
            word_parts, labels = func('на')
        elif word_without_ending in {'поголовь', 'полюдь'}:  # 516
            word_parts, labels = func('по')
        elif word_without_ending in {'предсерди'}:
            word_parts, labels = func('пред')
        elif word_without_ending in {'обочин', 'окраин'}:  # 511
            word_parts, labels = prefix_and_suffix('о', 'ин', {'', 'й'})
        elif word_without_ending in {'побратим'}:  # 516
            word_parts, labels = prefix_and_suffix('по', 'им', {''})
        elif word_without_ending in {'поручень', 'поручн'}:
            word_parts, labels = prefix_and_suffix('по', 'ень', {'а'})
            word_parts, labels = prefix_and_suffix('по', 'н', {'а'})
        elif word_without_ending in {'пощёчин'}:  # 516
            word_parts, labels = prefix_and_suffix('по', 'ин', {''})
        elif word_without_ending in {'надколенник'}:
            word_parts, labels = prefix_and_suffix('над', 'ник', {'о'})
        elif word_without_ending in {'отродь'}:
            word_parts, labels = func('от')
        elif word_without_ending in {'околоцветник', 'околоплодник'}:  # 512
            word_parts, labels = prefix_and_suffix('около', 'ник', {''})
        elif word_without_ending in {'околоплодь'}:  # 512
            word_parts, labels = func('около')
        elif word_without_ending in {'подмастерь'}:  # 518
            word_parts, labels = func('под')
        elif word_without_ending in {'попутчик'}:
            word_parts, labels = prefix_and_suffix('по', 'чик', {'ь'})
        elif word_without_ending in {'подлещик'}:
            word_parts, labels = prefix_and_suffix('под', 'ик', {''})
        elif word_without_ending in {'подкулачник'}:
            word_parts, labels = prefix_and_suffix('под', 'ник', {''})
        elif word_without_ending in {'подорожник', 'поморник', 'побережник', 'поручейник'}:
            word_parts, labels = prefix_and_suffix('по', 'ник', {'', 'а', 'е'})  # 516
        elif word_without_ending in {'ошейник'}:  # 511
            word_parts, labels = prefix_and_suffix_simple('о', 'ник')
        elif word_without_ending in {'падчериц'}:  # 514
            word_parts, labels = prefix_and_suffix_simple('па', 'иц')
        elif word_without_ending in {'позёмк'}:
            word_parts, labels = prefix_and_suffix_simple('по', 'к')
        elif word_without_ending in {'позёмок'}:
            word_parts, labels = prefix_and_suffix_simple('по', 'ок')
        elif word_without_ending in {'подорлик'}:
            word_parts, labels = prefix_and_suffix_simple('под', 'ик')
        elif word_without_ending in {'проулок'}:
            word_parts, labels = prefix_and_suffix_simple('про', 'ок')
        elif word_without_ending in {'проулк'}:
            word_parts, labels = prefix_and_suffix_simple('про', 'к')
        elif word_without_ending in {'подрозетник', 'подзатыльник'}:  # 517
            word_parts, labels = prefix_and_suffix_simple('под', 'ник')
        elif word_without_ending in {'распутиц'}:  # 523
            word_parts, labels = prefix_and_suffix_simple('рас', 'иц')
        elif word_without_ending in {'сукровиц'}:  # 525
            word_parts, labels = prefix_and_suffix_simple('су', 'иц')
        elif word_without_ending in {'небылиц'}:  # 525
            word_parts, labels = prefix_and_suffix_simple('не', 'иц')
        elif word_without_ending in {'аритми', 'атони'}:  # 525
            word_parts, labels = prefix_and_suffix_simple('а', 'и')
        elif word_without_ending in {'предислови'}:  # 525
            word_parts, labels = func('преди')
        elif word_without_ending in {'послеслови'}:  # 525
            word_parts, labels = func('после')
        elif word_without_ending in {'невери'}:  # 528
            word_parts, labels = func('не')
        elif word_without_ending in {'окось'}:  # 529
            word_parts, labels = func('о')
        for suff in {'ок', 'к'}:
            if word_without_ending in {'пасынок', 'пасынк', 'патрубок', 'патрубк', 'падымок', 'падымк', 'паводок',
                                       'паводк', 'паголенок', 'паголенк'}:
                word_parts, labels = prefix_and_suffix('па', suff, {'', 'ь', 'а'})
            elif word_without_ending in {'отголосок', 'отголоск', 'отзимок', 'отзимк'}:
                word_parts, labels = prefix_and_suffix('от', suff, {'', 'а'})
            elif word_without_ending in {'простенок', 'простенка', 'пролесок', 'пролеска', 'прожилок', 'прожилк',
                                         'посёлок', 'просёлк'}:
                word_parts, labels = prefix_and_suffix('от', suff, {'', 'а', 'о'})
            elif word_without_ending in {'погодок', 'погодк'}:  # 516
                word_parts, labels = prefix_and_suffix('по', suff, {''})
            elif word_without_ending in {'подгруздок', 'подгруздк', 'подлисок', 'подлис', 'подсвинок', 'подсвинк',
                                         'подтёлок', 'подтёлк', 'подсумок', 'подсумк', 'подпилок', 'подпилк',
                                         'подголосок', 'подголоcк', 'подпасок', 'подпаск', 'подсудок', 'подсудк',
                                         'подчасок', 'подчаск'}:
                word_parts, labels = prefix_and_suffix('под', suff, {'а', 'ь', 'ья', 'ка', 'тух', 'ов'})
            elif word_without_ending in {'суглинок', 'суглинк'}:
                word_parts, labels = prefix_and_suffix('су', suff, {'а'})
        for suff in {'ек', 'к'}:
            if word_without_ending in {'отроек', 'отройк'}:
                word_parts, labels = prefix_and_suffix('от', suff, {'', 'й'})
        else:
            word_parts, labels = prefix_and_suffix('анти', 'ин', {'', 'ь', 'о', 'е', 'а', 'я'})
            word_parts, labels = prefix_and_suffix('на', 'ник', {'', 'ь', 'о', 'е', 'а', 'я'})
            word_parts, labels = prefix_and_suffix('со', 'ник', {'', 'ь', 'о', 'е', 'а', 'я'})
            word_parts, labels = prefix_and_suffix('под', 'ник', {'', 'ь', 'о', 'е', 'а', 'я'})
            for pref in {'без', 'бес', 'пере'}:
                for suff in {'иц', 'ок', 'к'}:  # без/бес + иц - 504, пере + иц - 515
                    word_parts, labels = prefix_and_suffix(pref, suff, {'', 'ь', 'о', 'е', 'а', 'я'})
            for pref in {'за', 'по', 'между', 'меж', 'над', 'пере', 'пред', 'при', 'со'}:  # за - 506 при - 520
                word_parts, labels = func(pref)
            for suff in {'ок', 'к'}:
                word_parts, labels = prefix_and_suffix('недо', suff, {'', 'ь', 'о', 'е', 'а', 'я'})
                word_parts, labels = prefix_and_suffix('пере', suff, {'', 'ь', 'о', 'е', 'а', 'я'})
                if 'недо' in word_parts:
                    word_parts[0] = 'не'
                    word_parts.insert(1, 'до')
                    labels.insert(1, 'префикс')

    elif p.tag.POS == 'ADVB':
        for k in {'а', 'я', 'ы', 'и'}:
            word_parts, labels = pref_rep_part_suff('в', 'пол', k, {'', 'ь', 'й', 'о', 'е', 'й', 'а', 'я'})
            word_parts, labels = pref_rep_part_suff('в', 'пол', 'а', {'ый', 'ий', 'ой'})
        for pref in {'до', 'из', 'ис', 'с', 'со'}:
            word_parts, labels = prefix_and_suffix(pref, 'а', {'ый', 'ий', 'ой'})
        if word in {'книзу', 'кверху'}:
            word_parts, labels = prefix_and_suffix('к', 'у', {''})
        if mark and word_parts[-1] in {'а', 'о', 'у'}:
            word_parts = word_parts[:-1]
            labels = labels[:-1]

    if mark:
        return word_parts, labels

    elif not mark:
        global forms
        forms = [normal, word_without_prefix, word_without_prefix1, word_without_prefix2]
        word_parts, labels = pref_or_rep_part(forms, identify_prefix)
        forms = [normal, word_without_rep_part, word_without_rep_part1, word_without_rep_part2]
        word_parts, labels = pref_or_rep_part(forms, identify_rep_part)

        for i, el in enumerate(labels):
            if el == '?':
                unknown = word_parts[i]
                if len(unknown) == 1:
                    labels[i] = 'корень'
                break

        if not mark:
            if p.tag.POS == 'NOUN' and 'Fixd' not in p.tag:
                if word in {'судей', 'судий', 'гостий', 'сватий', 'игумений'}:  # гостий, сватий, игумений - 388
                    word_parts[0] = word[:-2]
                    word_parts[1] = word[-2:]
                    word_parts.insert(2, '☐')
                    labels = ['корень', 'суффикс', 'окончание']
                elif word_without_ending in {'паклен', 'паужин', 'паветер', 'паветр', 'пагруздь', 'пагрузд',
                                             'пащенок',
                                             'пасок'}:
                    word_parts, labels = identify_prefix_simple('па')
                elif word_without_ending in {'сумрак', 'супесок', 'супеск'}:
                    word_parts, labels = identify_prefix_simple('су')
                else:
                    if word_without_ending.endswith('стви'):
                        word_parts, labels = identify_suffix_full('стви', {'ный', 'ний', 'ной'}, set(), 'ADJF')
                        if 'стви' in word_parts:
                            word_parts[-2] = 'и'
                            word_parts.insert(-2, 'ств')
                            labels.insert(-2, 'суффикс')
                    if word_parts[-2].endswith('унь') and word_parts[-1][0] in 'еёиюя':  # 388
                        if word_without_ending[:-1] in words:
                            word_parts, labels = identify_suffix_simple('унь')
                    if word_without_ending.endswith('ун') and word_parts[-1] == 'ий':
                        word_parts[0] = word_parts[0][:-2]
                        word_parts.insert(1, 'ун')
                        labels.insert(1, 'суффикс')
                    if word_without_ending[-3:] in {'ени', 'ень', 'ани', 'ань'}:
                        word_parts, labels = eni(word_without_ending)
                        if 'ени' not in word_parts and 'ень' not in word_parts:
                            eni(unknown)
                    for k in {'ость', 'ность', 'ост', 'ност'}:  # 271
                        word_parts, labels = identify_suffix_full(k, {'еть', 'ить', 'ать', 'иться'}, {'ревность'}, 'INFN')
                        if word_parts[-2] in {'ность', 'ност'}:
                            word_parts[-2] = word_parts[-2][1:]
                            word_parts.insert(-2, 'н')
                    word_parts, labels = identify_suffix_full('иц', {'', 'ь', 'лец', 'ец'}, {'императриц'}, 'NOUN')  # 382
                    for k in {'иц', 'ц'}:
                        word_parts, labels = identify_suffix_full(k, {'а', 'я', 'ь', 'ка'}, set(), 'NOUN')  # 416
                    word_parts, labels = identify_suffix_full('ниц', {'ник'}, set(), 'NOUN')
                    word_parts, labels = identify_suffix_full('щиц', {'щик'}, set(), 'NOUN')
                    word_parts, labels = identify_suffix_full('чиц', {'чик'}, set(), 'NOUN')
                    word_parts, labels = identify_suffix_full('ниц', {''}, set(), 'NOUN')
                    for suff in {'ин', 'ын', 'инь', 'ынь'}:
                        word_parts, labels = identify_suffix_full(suff, {'', 'ь', 'й'},
                                             {'княгин', 'барын', 'боярын', 'княгинь', 'барынь', 'боярынь'}, 'NOUN')
                    word_parts, labels = identify_suffix_full('есс', {''}, set(), 'NOUN')
                    word_parts, labels = identify_suffix_full('ис', {''}, set(), 'NOUN')
                    if labels[i] == '?' and word_without_ending.endswith('трис'):
                        for k in {'тёр', 'тор'}:
                            word_new = normal[:-5] + k
                            if word_new in words:
                                word_parts[i] = unknown[:-2]
                                word_parts.insert(i + 1, 'ис')
                                labels[i] = 'корень?'
                                labels.insert(i + 1, 'суффикс')
                    word_parts, labels = identify_suffix_full('ш', {'', 'ь'}, set(), 'NOUN')
                    if 'ш' not in word_parts and word_without_ending.endswith('ьш'):
                        word_new = unknown[:-2]
                        if word_new in words:
                            word_parts, labels = identify_suffix_simple('ш')
                    for suff in {'оныш', 'ёныш'}:
                        identify_suffix(suff, {'ь', 'а', 'я', 'и'}, set(), word_without_ending, 'NOUN')
                    for k in {'онок', 'онк', 'ёнок', 'ёнк', 'ат', 'ят'}:
                        if word_parts[-1].startswith(k):
                            new_ending = word_parts[-1][len(k):]
                            if new_ending == '':
                                new_ending = '☐'
                            word_parts.append(new_ending)
                            labels[-1] = 'суффикс'
                            labels.append('окончание')
                            if word_parts[-3] not in {'арапч', 'татарч', 'барч', 'курч'}:
                                word_parts[-2] = k
                            else:
                                for m in {'арапч', 'татарч', 'барч', 'курч'}:
                                    if word_parts[-3] == m:
                                        word_parts[-3] = m[:-1]
                                        word_parts[-2] = 'ч' + k
                    if word_without_ending in {'барчук', 'саранчук'}:
                        word_parts, labels = identify_suffix_simple('чук')
                    elif word_without_ending == 'птенец':
                        word_parts, labels = identify_suffix_simple('енец')
                    elif word_without_ending == 'птенц':
                        word_parts, labels = identify_suffix_simple('енц')
                    for suff in {'ок', 'ек', 'к'}:
                        word_parts, labels = identify_suffix_full(suff, {'а', 'ь', 'о', ''}, set(), 'NOUN')
                    word_parts, labels = identify_suffix_full('ин', {'', 'а', 'ь', 'ы', 'и', 'о', 'ник', 'ка', 'ок'}, set(), 'NOUN')
                    for suff in {'инк', 'инок'}:
                        word_parts, labels = identify_suffix_full(suff, {'а', 'ь', 'о', '', 'й'}, set(), 'NOUN')
                    word_parts, labels = identify_suffix_full('изм',
                                         {'ный', 'ний', 'ьный', 'ьний', 'ной', 'ский', 'ской', 'ичный', 'ический',
                                          'ист'}, set(), 'ADJF')
                    word_parts, labels = identify_suffix_full('ист',
                                         {'ный', 'ний', 'ьный', 'ьний', 'ной', 'ский', 'ской', 'ичный', 'ический'},
                                         set(), 'ADJF')
                    word_parts, labels = identify_suffix_full('оид', {'', 'а'}, set(), 'NOUN')
                    word_parts, labels = identify_suffix_full('иц', {'ый', 'ий', 'ой'}, set(), 'ADJF')  # 296
                    word_parts, labels = identify_suffix_full('иц', {'ный', 'ний', 'ной'}, {'разница'}, 'ADJF')  # 321
                    for suff in {'ость', 'ост', 'ность', 'ност'}:  # 310, 315
                        word_parts, labels = identify_suffix_full(suff, {'ый', 'ий', 'ой'},
                                             {'тягость', 'близость', 'гадость', 'дерзость', 'мерзость', 'робость',
                                              'сладость', 'низость', 'узость', 'узост', 'тягост', 'близост',
                                              'гадост',
                                              'дерзост', 'мерзост', 'робост', 'сладост', 'низост', 'узост'},
                                             'ADJF')  # 310
                        if word_parts[-2] in {'ность', 'ност'} and word_without_ending not in {'современность',
                                                                                               'современност'}:
                            word_parts[-2] = word_parts[-2][1:]
                            word_parts.insert(-2, 'н')
                    for suff in {'есть', 'ест'}:  # 310, 315
                        word_parts, labels = identify_suffix_full(suff, {'ый', 'ий', 'ой'}, {'тяжесть', 'тяжест'}, 'ADJF')
                    for suff in {'ств', 'еств'}:
                        word_parts, labels = identify_suffix_full(suff,
                                             {'ый', 'ий', 'ой', 'ский', 'ской', 'еской', 'еский', 'ный', 'ний',
                                              'ной'},
                                             set(), 'ADJF')
                        word_parts, labels = identify_suffix_full('от', {'ый', 'ий', 'ой'}, {'широт', 'красот', 'тягот', 'духот'},
                                             'ADJF')
                    for suff in {'ишк', 'ишек'}:
                        word_parts, labels = identify_suffix_full(suff, {'', 'а', 'о', 'ы', 'и'},
                                             {'мальчишк', 'пальтишк', 'мальчишек', 'пальтишек', 'ребятишк',
                                              'ребятишек'}, 'NOUN')
                    word_parts, labels = identify_suffix_full('ищ', {'', 'ь', 'а', 'я', 'о', 'е'}, set(), 'NOUN')
                    for suff in {'тель', 'тел'}:  # 211
                        word_parts, labels = identify_suffix_full(suff, {'ть'},
                                             {'гонитель', 'гонител', 'движитель', 'движител', 'сказитель',
                                              'сказител',
                                              'воитель', 'прихлебатель', 'прихлебател', 'блюститель', 'блюстител',
                                              'властитель', 'властител'}, 'INFN')
                        if ('тель' in word_parts or 'тел' in word_parts) and unknown not in {'вои', 'прихлеба',
                                                                                             'блюсти'}:
                            for x in {'и', 'а', 'я', 'е'}:
                                word_parts, labels = identify_suffix_simple(x)
                    if 'тель' not in word_parts and 'тел' not in word_parts:
                        for k in {'итель', 'ител'}:
                            if word_without_ending.endswith(k):
                                new_word = word_without_ending[:-len(k)] + 'еть'
                                if new_word in words:
                                    word_parts, labels = identify_suffix_simple(k[1:])
                                    word_parts, labels = identify_suffix_simple('и')
                    for suff in {'арь', 'ар'}:  # 222
                        word_new, word_new2, word_new3 = '', '', ''
                        if suff not in word_parts and word_without_ending.endswith(suff):
                            for m in {'ь', 'ить', 'ать'}:
                                if suff not in word_parts:
                                    word_new3 = word_without_ending[:-len(suff)] + m
                                    if word_without_ending[:-len(suff)].endswith('к'):
                                        word_new = word_without_ending[:-(len(suff) + 1)] + 'ч' + m
                                    if word_new3 in words or word_new in words:
                                        if word_new3 in words:
                                            word_new = word_new3
                                        elif word_new in words:
                                            word_new = word_new
                                        p_new = morph.parse(word_new)[0]
                                        if 'INFN' in p_new.tag.POS:
                                            word_parts, labels = identify_suffix_simple(suff)
                    if word_without_ending not in {'кубарь', 'кубар'}:  # 340
                        for suff in {'арь', 'ар'}:
                            identify_suffix_full(suff, {'', 'ь', 'а', 'я', 'о', 'е', 'ы', 'и'},
                                                 {'свинарь', 'свинар', 'технарь', 'технар', 'псарь', 'псар',
                                                  'вратарь',
                                                  'вратар', 'шинкарь', 'шинкар'}, 'NOUN')
                    if word_without_ending == 'дороговизн':
                        word_parts, labels = identify_suffix_simple('овизн')
                    else:
                        word_parts, labels = identify_suffix_full('изн', {'ый', 'ий', 'ой'}, set(), 'ADJF')
                    if word_without_ending == 'царевн':
                        word_parts, labels = identify_suffix_simple('н')
                    elif word_without_ending in {'оленух', 'маралух'}:
                        word_parts, labels = identify_suffix_simple('ух')
                    elif word_without_ending in {'синев', 'чернев', 'коричнев'}:
                        word_parts, labels = identify_suffix_simple('ев')
                    elif word_without_ending in {'огнив', 'зарев'}:
                        word_parts, labels = identify_suffix_simple('ев')
                        word_parts, labels = identify_suffix_simple('ив')
                    elif word_without_ending in {'разладиц', 'косовиц'}:  # 283
                        word_parts, labels = identify_suffix_simple('иц')
                    elif word_without_ending == 'крупиц':  # 411
                        word_parts, labels = identify_suffix_simple('иц')
                    elif word_without_ending == 'троиц':  # 444
                        word_parts, labels = identify_suffix_simple('иц')
                    elif word_without_ending == 'своячениц':  # 382
                        word_parts, labels = identify_suffix_simple('ниц')
                        word_parts, labels = identify_suffix_simple('е')
                    elif word_without_ending in {'болезнь', 'боязнь', 'жизнь', 'болезн', 'боязн', 'жизн'}:  # 283
                        word_parts, labels = identify_suffix_simple('знь')
                        word_parts, labels = identify_suffix_simple('зн')
                        if word_without_ending in {'болезнь', 'болезн'}:
                            word_parts, labels = identify_suffix_simple('е')
                    elif unknown in {'знахарь', 'знахар'}:  # 222
                        word_parts, labels = identify_suffix_simple('арь')
                        word_parts, labels = identify_suffix_simple('ар')
                    elif unknown in {'дикарь', 'дикар', 'сизарь', 'сизар', 'сухарь', 'сухар'}:  # 314
                        word_parts, labels = identify_suffix_simple('арь')
                        word_parts, labels = identify_suffix_simple('ар')
                    elif unknown in {'дикарь', 'дикар', 'сизарь', 'сизар', 'сухарь', 'сухар'}:  # 314
                        word_parts, labels = identify_suffix_simple('арь')
                        word_parts, labels = identify_suffix_simple('ар')
                    elif unknown in {'маринад', 'рафинад'}:  # 254
                        word_parts, labels = identify_suffix_simple('ад')
                    elif unknown in {'вражд'}:  # 379
                        word_parts, labels = identify_suffix_simple('ад')
                    elif unknown in {'маскарад'}:  # 379
                        word_parts, labels = identify_suffix_simple('арад')

            elif p.tag.POS == 'ADJF':
                for k in {'лив', 'ив'}:
                    word_parts, labels = identify_suffix_full(k, {'', 'ь', 'о', 'а', 'я', 'ие', 'ье', 'ка'}, set(), 'NOUN')
                for suff in {'ин', 'н'}:
                    word_parts, labels = identify_suffix_full(suff, {'а', 'я', 'ь', 'о', ''}, set(), 'NOUN')
                word_parts, labels = identify_suffix_full('т', {'ть'}, set(), 'VERB')
                word_parts, labels = identify_suffix_full('ист', {'ый', 'ий', 'ой'}, set(), 'ADJF')
                for suff in {'оват', 'еват'}:
                    word_parts, labels = identify_suffix_full(suff, {'ый', 'ий', 'ой'}, set(), 'ADJF')
                for suff in {'ов', 'ев'}:
                    word_parts, labels = identify_suffix_full(suff, {'ый', 'ий', 'ой'}, {'бежев', 'бордов'}, 'ADJF')
                for suff in {'ущ', 'ющ'}:
                    word_parts, labels = identify_suffix_full(suff, {'ый', 'ий', 'ой'}, set(), 'ADJF')
                word_parts, labels = identify_suffix_full('енн', {'ый', 'ий', 'ой'}, set(), 'ADJF')
                for suff in {'ав', 'яв', 'ощав'}:
                    for k in {'ый', 'ий', 'ой'}:
                        if labels[i] == '?' and word_without_ending.endswith(suff):
                            word_new3 = word_without_ending[:-len(suff)] + k
                            if word_without_ending[:-len(suff)].endswith('ж'):
                                word_new = word_without_ending[:-(len(suff) + 1)] + 'д' + k
                            if word_new3 in words or word_new in words or word_without_ending == 'слащав':
                                word_parts[i] = unknown[:-len(suff)]
                                word_parts.insert(i + 1, suff)
                                labels[i] = 'корень'
                                labels.insert(i + 1, 'суффикс')
                word_parts, labels = identify_suffix_full('оньк', {'ый', 'ий', 'ой'}, {'мягоньк', 'махоньк'}, 'ADJF')
                word_parts, labels = identify_suffix_full('еньк', {'ый', 'ий', 'ой'}, {'сладеньк', 'реденьк', 'крепеньк', 'узеньк'},
                                     'ADJF')
                if word_without_ending == 'слащав':
                    word_parts, labels = identify_suffix_simple('ав')
                elif word_without_ending in {'малюсеньк', 'тонюсеньк'}:
                    word_parts, labels = identify_suffix_simple('юсеньк')
                elif word_without_ending == 'такусеньк':
                    word_parts, labels = identify_suffix_simple('усеньк')
                elif word_without_ending in {'первичн', 'вторичн', 'третичн', 'четвертичн'}:
                    word_parts, labels = identify_suffix_simple('ичн')
                elif word_without_ending in {'боязлив', 'ребячлив'}:
                    word_parts, labels = identify_suffix_simple('лив')
                if word_without_ending == 'вдольрядн':
                    word_parts[:-1] = ['вдоль', 'ряд', 'н']
                    labels[:-1] = ['префикс', 'корень', 'суффикс']

            elif p.tag.POS in {'INFN', 'VERB', 'GRND', 'PRTF'}:
                for m in with_ka:
                    if unknown == m + 'ка':
                        word_parts, labels = identify_suffix_simple('ка')
                if 	unknown not in {'кова', 'кива'}:
                    for x in {'ова', 'ева', 'ива', 'ыва'}:
                        word_parts, labels = identify_suffix_simple(x)
                        if x in word_parts and unknown != 'пир':
                            word_parts, labels = identify_suffix_simple('ир')
                            word_parts, labels = identify_suffix_simple('из')
                word_parts, labels = identify_suffix('ва', {'ть'}, {'плава'}, word_without_ending, 'INFN')
                word_parts, labels = suff_nicha(word_without_ending)
                word_parts, labels = suff_nicha(unknown)

                if 'ка' not in word_parts and 'нича' not in word_parts and 'ича' not in word_parts and 'а' not in word_parts:
                    for x in {'и', 'а', 'я', 'е'}:
                        word_parts, labels = identify_suffix_simple(x)
                word_parts, labels = identify_suffix_simple('ну')
                if word_parts[i + 1] == 'ну' and word_parts[i][-1] == 'к' and word_parts[i][:-1] in with_ka:
                    unknown = word_parts[i]
                    word_parts, labels = identify_suffix_simple('к')

            elif p.tag.POS == 'ADJS':
                if word_without_ending in {'малюсенек', 'тонюсенек', 'малюсеньк', 'тонюсеньк'}:
                    word_parts, labels = identify_suffix_simple('юсенек')
                    word_parts, labels = identify_suffix_simple('юсеньк')
                elif word_without_ending in {'такусенек', 'такусеньк'}:
                    word_parts, labels = identify_suffix_simple('усенек')
                    word_parts, labels = identify_suffix_simple('усеньк')

            elif p.tag.POS == 'ADVB':
                if word_without_ending.endswith('а'):
                    comb_pref = {'сыз': ['с', 'ыз'], 'испод': ['ис', 'под'], 'поза': ['по', 'за']}
                    for item in comb_pref:
                        if word_without_ending.startswith(item):
                            word_parts[0] = word_parts[0][len(item):]
                            word_parts.insert(0, comb_pref[item][1])
                            word_parts.insert(0, comb_pref[item][0])
                labels = ['корень', 'суффикс']
            if word == 'сыздетства':
                word_parts = ['c', 'ыз', 'дет', 'ств', 'а']
                labels = ['префикс', 'префикс', 'корень', 'суффикс', 'суффикс']
            elif word == 'сызмальства':
                word_parts = ['c', 'ыз', 'маль', 'ств', 'а']
                labels = ['префикс', 'префикс', 'корень', 'суффикс', 'суффикс']
            elif word == 'втридорога':
                word_parts = ['в', 'тр', 'и', 'дорог', 'а']
                labels = ['префикс', 'корень', 'интерфикс', 'корень', 'суффикс']
            elif word == 'втридешева':
                word_parts = ['в', 'тр', 'и', 'дешев', 'а']
                labels = ['префикс', 'корень', 'интерфикс', 'корень', 'суффикс']
            elif word == 'спросонья':
                word_parts = ['с', 'про', 'сонь', 'я']
                labels = ['префикс', 'префикс', 'корень', 'суффикс']
            elif word == 'наверняка':
                word_parts = ['на', 'вер', 'н', 'як', 'а']
                labels = ['префикс', 'корень', 'суффикс', 'суффикс', 'суффикс']

        if labels[i] in {'?', 'корень?'} and most_probable_variant_mean(word_parts[i]):
            new_lst = most_probable_variant_mean(word_parts[i]).split('/')
            word_parts.remove(word_parts[i])
            for a in range(len(new_lst)):
                word_parts.insert(i, new_lst[a])
                i += 1
        return word_parts

while True:
    word_or_text = input('Выберите режим работы: слово (с) или текст (т) (для выхода наберите "exit"): ')
    if word_or_text == 'exit':
        break
    if word_or_text == 'с':
        word = input('Введите слово: ')
        print(analysis())
    elif word_or_text == 'т':
        file_name = input('Укажите название файла: ')
        morph = pymorphy2.MorphAnalyzer()
        with open(file_name, encoding='utf-8') as f:
            text = f.read().lower()
            text_tokenized = nltk.word_tokenize(text)
            new_text, information = '', []
            for t in text_tokenized:
                p_t = morph.parse(t)[0]
                if t.isalpha() and p_t.tag.POS not in {'PREP', 'CONJ', 'PRCL', "INTJ"}:
                    word = t
                    word_print = word
                    word_parts = analysis()
                    new_text += ' | '.join([word_print, '/'.join(word_parts)]) + '\n'
        annotation_file = input('Введите название файла, в который Вам нужно сохранить результаты аннотирования: ')
        with open(annotation_file, 'w', encoding='utf-8') as f:
            f.write(new_text[:-1])
            print('В файл {} были записаны результаты аннотирования.'.format(annotation_file))


def evaluation (file_name):
    punctuation = '''!"#$%&'()*+,-.:;<=>?@[\]^_`{|}~'''
    hits, insertions, deletions = 0, 0, 0
    dict = {}
    standards, results = [], []

    with open('Морфемно-орфографический словарь А. Н. Тихонова.txt', encoding='windows-1251') as f:
        for line in f:
            word, analysis = line.split(' | ')
            if word[-1].isdigit():
                word = word[:-1]
            analysis = analysis.strip().split()[0]
            if analysis[-1] in punctuation:
                analysis = analysis[:-1]
            if analysis[-1].isdigit():
                analysis = analysis[:-1]
            if analysis[-1] == '☐':
                analysis = analysis[:-1]
            analysis = analysis.replace("'", '')
            dict[word] = analysis

    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split(' | ')
            word_res, analysis_res = line[0], line[1]
            standards.append(dict[word_res])
            results.append(analysis_res)

    for num_word in range(len(standards)):
        for num_symbol in range(len(standards[num_word])):
            if len(standards[num_word]) > len(results[num_word]):
                results[num_word] += ' '
            if results[num_word][num_symbol] == standards[num_word][num_symbol] and results[num_word][num_symbol] == '/':
                hits += 1
            elif results[num_word][num_symbol] != standards[num_word][num_symbol] and results[num_word][num_symbol] == '/':
                insertions += 1
                results[num_word] = results[num_word][:num_symbol] + results[num_word][num_symbol + 1:]
            elif results[num_word][num_symbol] != standards[num_word][num_symbol] and standards[num_word][num_symbol] == '/':
                deletions += 1
                results[num_word] = results[num_word][:num_symbol] + '/' + results[num_word][num_symbol:]

    precision = round(hits / (hits + insertions), 3)
    recall = round(hits / (hits + deletions), 3)
    fmeasure = round(2 * hits / (2 * hits + insertions + deletions), 3)
    return hits, insertions, deletions, precision, recall, fmeasure

file_name = input('Введите имя файла: ')
hits, insertions, deletions, precision, recall, fmeasure = evaluation(file_name)
print('Hits: {}\nInsertions: {}\nDeletions: {}\nPrecision: {}\nRecall: {}\nF-measure: {}'.format(hits, insertions, deletions, precision, recall, fmeasure))

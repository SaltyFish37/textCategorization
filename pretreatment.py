import os
import re
import pickle


def get_stopwords():
    stop_bags = []
    with open("stopwords.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n').lower()
            stop_bags.append(line)
    f.close()
    print('已获得停用词...')
    return stop_bags


def get_ini_data(mode):
    texts = []  # texts：每个文件的词集为一项，texts存储所有项(X)，未处理
    labels = []  # labels：texts中每一项所属的种类(y)
    label_id = 0

    target = "20news-bydate-" + mode
    target_path = os.path.join(os.getcwd(), "20news-bydate", target)

    for type_name in sorted(os.listdir(target_path)):
        type_path = os.path.join(target_path, type_name)
        if os.path.isdir(type_path):
            for file_name in sorted(os.listdir(type_path)):
                file_path = os.path.join(type_path, file_name)
                f = open(file_path, 'r', encoding='utf-8', errors='ignore')
                texts.append(f.read())
                f.close()
                labels.append(label_id)
            label_id += 1
    # print(texts,'\n',labels)
    print('已获得原始数据，正在准备预处理...')
    return texts, labels


def clean_data(texts):
    stop_bags = get_stopwords()
    # 分词
    essay_set = []
    for i in range(len(texts)):
        essay_set.append(texts[i].split())

    # 去邮箱去网址，正则去符号去数字，全小写，去停用词
    clear_words = []
    punctuation = ',:;.\'\"`?/-+*&#()0123456789'
    for essay in essay_set:
        mid_words = []
        for word in essay:
            flag = 1
            for dot in punctuation:
                if dot in word.strip(punctuation):
                    essay.remove(word)
                    flag = 0
                    break
            if flag == 0:
                continue
            word = re.sub('[^a-zA-Z]+', '', word).lower()
            if word in stop_bags or len(word) == 0:
                continue
            else:
                mid_words.append(word)
        clear_words.append(mid_words)
    print('已清洗完成，去除邮箱、停用词、符号、数字等...')
    return clear_words


def get_word_bag(clear_words):
    word_bag = []
    for essay in clear_words:
        for word in essay:
            if word not in word_bag:
                word_bag.append(word)
    print('已获得词袋空间{}...'.format(len(word_bag)))
    return word_bag


def count(clear_words, word_bag):
    print('开始进行特征值记录工作...')
    word_count = []
    for essay in clear_words:
        tmp_count = {}
        for word in essay:
            if word in word_bag and word not in tmp_count.keys():
                tmp_count.update({word:1})
            elif word in word_bag and word in tmp_count.keys():
                tmp_count[word] += 1
        word_count.append(tmp_count)
    print('已完成计数工作...')
    return word_count


if __name__ == '__main__':
    
    save_path = os.path.join(os.getcwd(), 'midfile')

    data_train = 'train'
    texts_train, labels_train = get_ini_data(data_train)
    data_test = 'test'
    texts_test, labels_test = get_ini_data(data_test)
    
    # 存类名
    labels_train_con = pickle.dumps(labels_train)
    filename = os.path.join(save_path, 'labels_train')
    f = open(filename, 'wb')
    f.write(labels_train_con)
    f.close()
    print('数据已保存在文件{}'.format(filename))
    
    labels_test_con = pickle.dumps(labels_test)
    filename = os.path.join(save_path, 'labels_test')
    f = open(filename, 'wb')
    f.write(labels_test_con)
    f.close()
    print('数据已保存在文件{}'.format(filename))

    word_train = clean_data(texts_train)
    word_test = clean_data(texts_test)
    
    # 存洗好的词
    word_train_con = pickle.dumps(word_train)
    filename = os.path.join(save_path, 'word_train')
    f = open(filename, 'wb')
    f.write(word_train_con)
    f.close()
    print('数据已保存在文件{}'.format(filename))
    
    word_test_con = pickle.dumps(word_test)
    filename = os.path.join(save_path, 'word_test')
    f = open(filename, 'wb')
    f.write(word_test_con)
    f.close()
    print('数据已保存在文件{}'.format(filename))
    
    # 存词袋
    word_bag = get_word_bag(word_train)

    word_bag_con = pickle.dumps(word_bag)
    filename = os.path.join(save_path, 'word_bag')
    f = open(filename, 'wb')
    f.write(word_bag_con)
    f.close()
    print('数据已保存在文件{}'.format(filename))

    # 存计数
    word_train_count = count(word_train, word_bag)
    word_test_count = count(word_test, word_bag)

    word_train_count_con = pickle.dumps(word_train_count)
    filename = os.path.join(save_path, 'word_train_count')
    f = open(filename, 'wb')
    f.write(word_train_count_con)
    f.close()
    print('数据已保存在文件{}'.format(filename))

    word_test_count_con = pickle.dumps(word_test_count)
    filename = os.path.join(save_path, 'word_test_count')
    f = open(filename, 'wb')
    f.write(word_test_count_con)
    f.close()
    print('数据已保存在文件{}'.format(filename))

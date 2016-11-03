# -*- coding: utf-8 -*-

import math
from operator import itemgetter as _itemgetter
import numpy as np
import jieba
from sklearn import preprocessing
from collections import Counter
import numpy as np


def sigmoid(value):
    return 1 / (1 + math.exp(-value))

class Word2Vec():
    def __init__(self, vec_len=15000, learn_rate=0.025, win_len=5, model='cbow'):
        self.cutted_text_list = None
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.win_len = win_len
        self.model = model
        self.word_dict = None  # each element is a dict, including: word,possibility,vector,huffmancode
        self.huffman = None  # the object of HuffmanTree

    def build_word_dict(self, word_freq):
        # word_dict = [{word, freq, possibility, init_vector, huffman_code}, ]
        word_dict = {}
        freq_list = [x[1] for x in word_freq]
        sum_count = sum(freq_list)
        for item in word_freq:
            temp_dict = dict(
                word=item[0],
                freq=item[1],
                possibility=item[1] / sum_count,
                vector=np.random.random([1, self.vec_len]),
                Huffman=None
            )
            word_dict[item[0]] = temp_dict
        self.word_dict = word_dict

    def train(self, word_list):
        # build word_dict and huffman tree
        if self.huffman is None:
            if self.word_dict is None:
                counter = WordCounter(word_list)
                self.build_word_dict(counter.count_res.larger_than(5))
                self.cutted_text_list = counter.word_list
            self.huffman = HuffmanTree(self.word_dict, vec_len=self.vec_len)
        # start to train word vector
        before = (self.win_len - 1) >> 1
        after = self.win_len - 1 - before
        # get method
        if self.model == 'cbow':
            method = self.CBOW
        else:
            method = self.SkipGram
        # cut word
        if not self.cutted_text_list:
            # if the text has not been cutted
            for line in word_list:
                line = list(jieba.cut(line, cut_all=False))
                line_len = line.__len__()
                for i in range(line_len):
                    method(line[i], line[max(0, i - before):i] + line[i + 1:min(line_len, i + after + 1)])
        # if the text has been cutted
        total = len(self.cutted_text_list)
        count = 0
        for line in self.cutted_text_list:
            line_len = len(line)
            for i in range(line_len):
                method(line[i], line[max(0, i - before):i] + line[i + 1:min(line_len, i + after + 1)])
            count += 1
            print('{c} of {d}'.format(c=count, d=total))

    def CBOW(self, word, gram_word_list):

        if not self.word_dict.__contains__(word):
            return

        word_huffman = self.word_dict[word]['code']
        gram_vector_sum = np.zeros([1, self.vec_len])
        for i in range(gram_word_list.__len__())[::-1]:
            item = gram_word_list[i]
            if self.word_dict.__contains__(item):
                gram_vector_sum += self.word_dict[item]['vector']
            else:
                gram_word_list.pop(i)

        if gram_word_list.__len__() == 0:
            return

        e = self.__GoAlong_Huffman(word_huffman, gram_vector_sum, self.huffman.root)

        for item in gram_word_list:
            self.word_dict[item]['vector'] += e
            self.word_dict[item]['vector'] = preprocessing.normalize(self.word_dict[item]['vector'])

    def SkipGram(self, word, gram_word_list):

        if not self.word_dict.__contains__(word):
            return

        word_vector = self.word_dict[word]['vector']
        for i in range(gram_word_list.__len__())[::-1]:
            if not self.word_dict.__contains__(gram_word_list[i]):
                gram_word_list.pop(i)

        if gram_word_list.__len__() == 0:
            return

        for u in gram_word_list:
            u_huffman = self.word_dict[u]['code']
            e = self.__GoAlong_Huffman(u_huffman, word_vector, self.huffman.root)
            self.word_dict[word]['vector'] += e
            self.word_dict[word]['vector'] = preprocessing.normalize(self.word_dict[word]['vector'])

    def __GoAlong_Huffman(self, word_huffman, input_vector, root):

        node = root
        item = np.zeros([1, self.vec_len])
        for level in range(word_huffman.__len__()):
            huffman_charat = word_huffman[level]
            q = sigmoid(input_vector.dot(node.value.T))
            grad = self.learn_rate * (1 - int(huffman_charat) - q)
            item += grad * node.value
            node.value += grad * input_vector
            node.value = preprocessing.normalize(node.value)
            if huffman_charat == '0':
                node = node.right
            else:
                node = node.left
        return item


class HuffmanTreeNode:
    def __init__(self, value, possibility):
        self.possibility = possibility
        self.left = None
        self.right = None
        self.value = value  # the value of word
        self.code = "" # huffman code


class HuffmanTree:
    def __init__(self, word_dict, vec_len=15000):
        self.vec_len = vec_len  # the length of word vector
        self.root = None
        #
        word_dict_list = list(word_dict.values())
        node_list = [HuffmanTreeNode(item['word'], item['possibility']) for item in word_dict_list]
        self.build(node_list)
        # self.build_CBT(node_list)
        self.generate_huffman_code(self.root, word_dict)

    def build(self, node_list):
        while node_list.__len__() > 1:
            i1 = 0
            i2 = 1
            if node_list[i2].possibility < node_list[i1].possibility:
                [i1, i2] = [i2, i1]
            for i in range(2, len(node_list)):
                if node_list[i].possibility < node_list[i2].possibility:
                    i2 = i
                    if node_list[i2].possibility < node_list[i1].possibility:
                        [i1, i2] = [i2, i1]
            top_node = self.merge(node_list[i1], node_list[i2])
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0, top_node)
        self.root = node_list[0]

    def build_CBT(self, node_list):  # build a complete binary tree
        node_list.sort(key=lambda x: x.possibility, reverse=True)
        node_num = node_list.__len__()
        before_start = 0
        while node_num > 1:
            for i in range(node_num >> 1):
                top_node = self.merge(node_list[before_start + i * 2], node_list[before_start + i * 2 + 1])
                node_list.append(top_node)
            if node_num % 2 == 1:
                top_node = self.merge(node_list[before_start + i * 2 + 2], node_list[-1])
                node_list[-1] = top_node
            before_start = before_start + node_num
            node_num = node_num >> 1
        self.root = node_list[-1]

    def generate_huffman_code(self, node, word_dict):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            # go along left tree
            while node.left or node.right:
                code = node.code
                node.left.code = code + "1"
                node.right.code = code + "0"
                stack.append(node.right)
                node = node.left
            word = node.value
            code = node.code
            word_dict[word]['code'] = code

    def merge(self, node1, node2):
        top_pos = node1.possibility + node2.possibility
        top_node = HuffmanTreeNode(np.zeros([1, self.vec_len]), top_pos)
        if node1.possibility >= node2.possibility:
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node


class WordCounter:
    def __init__(self, word_list):
        self.word_list = word_list
        self.stop_word = {}
        self.count_res = None
        self.count_word(self.word_list)

    def count_word(self, word_list, cut_all=False):
        filtered_word_list = []
        index = 0
        for line in word_list:
            res = list(jieba.cut(line, cut_all=cut_all))
            word_list[index] = res
            index += 1
            filtered_word_list += res

        self.count_res = MulCounter(filtered_word_list)
        for word in self.count_res:
            if word in self.stop_word:
                self.count_res.pop(word)


class MulCounter(Counter):
    # extends from collections.Counter
    # add some methods, larger_than and less_than
    def __init__(self, element_list):
        super(MulCounter, self).__init__(element_list)

    def larger_than(self, minvalue, ret='list'):
        temp = sorted(self.items(), key=_itemgetter(1), reverse=True)
        low = 0
        high = temp.__len__()
        while high - low > 1:
            mid = (low + high) >> 1
            if temp[mid][1] >= minvalue:
                low = mid
            else:
                high = mid
        if temp[low][1] < minvalue:
            if ret == 'dict':
                return {}
            else:
                return []
        if ret == 'dict':
            ret_data = {}
            for ele, count in temp[:high]:
                ret_data[ele] = count
            return ret_data
        else:
            return temp[:high]

    def less_than(self, maxvalue, ret='list'):
        temp = sorted(self.items(), key=_itemgetter(1))
        low = 0
        high = len(temp)
        while high - low > 1:
            mid = (low + high) >> 1
            if temp[mid][1] <= maxvalue:
                low = mid
            else:
                high = mid
        if temp[low][1] > maxvalue:
            if ret == 'dict':
                return {}
            else:
                return []
        if ret == 'dict':
            ret_data = {}
            for ele, count in temp[:high]:
                ret_data[ele] = count
            return ret_data
        else:
            return temp[:high]


if __name__ == '__main__':
    data = ['Merge multiple sorted inputs into a single sorted output',
            'The API below differs from textbook heap algorithms in two aspects']
    wv = Word2Vec(vec_len=500)
    wv.train(data)
    print(wv.word_dict)
    # FI.save_pickle(wv.word_dict, './static/wv.pkl')
    #
    # data = FI.load_pickle('./static/wv.pkl')
    # x = {}
    # for key in data:
    #     temp = data[key]['vector']
    #     temp = preprocessing.normalize(temp)
    #     x[key] = temp
    # FI.save_pickle(x,'./static/normal_wv.pkl')

    # x = FI.load_pickle('./static/normal_wv.pkl')
    # def cal_simi(data,key1,key2):
    #     return data[key1].dot(data[key2].T)[0][0]
    # keys=list(x.keys())
    # for key in keys:
    #     print(key,'\t',cal_simi(x,'姚明',key))

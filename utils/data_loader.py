# coding=utf-8
# @Author: dcteng
# @Desc: { 模块描述 }
# @Date: 2023/07/18

import os
import json
from functools import reduce

import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict

import torch
from ordered_set import OrderedSet

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.config import *

from utils.graph_loader import ProGraph

from transformers import ElectraTokenizer, XLNetTokenizer, BertTokenizer
from transformers import ElectraTokenizerFast, XLNetTokenizerFast, BertTokenizerFast

'''
    Most of the code in this file is copied from https://github.com/LooperXX/ProSLU/blob/master/utils/loader.py
'''


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


def apply_mapping(mapping, data):
    if isinstance(data, int):
        # 如果data是整数，则根据映射表将其转换
        return mapping.get(data, data)
    elif isinstance(data, list):
        # 如果data是列表，则递归处理其中的元素
        return [apply_mapping(mapping, elem) for elem in data]
    else:
        # 其他类型的数据，保持原样
        return data


def divide_chunks(lengths, n):
    # decresing lengths
    assert n == 2
    assert len(lengths) >= 2
    min_area = len(lengths) * lengths[0]
    divide_index = len(lengths) - 1
    for i, h in enumerate(lengths):
        first_sum = lengths[0] * (i + 1)
        sec_sum = h * (len(lengths) - i - 1)
        area = first_sum + sec_sum
        if area < min_area:
            min_area = area
            divide_index = i
    return [(0, divide_index + 1), (divide_index + 1, len(lengths))]


class Alphabet(object):
    """
    Storage and serialization a set of elements.
    """

    def __init__(self, name, if_use_pad, if_use_unk):

        self.name = name
        self.if_use_pad = if_use_pad
        self.if_use_unk = if_use_unk

        self.index2instance = OrderedSet()
        self.instance2index = OrderedDict()

        # Counter Object record the frequency
        # of element occurs in raw text.
        self.counter = Counter()

        if if_use_pad:
            self.sign_pad = "<PAD>"
            self.add_instance(self.sign_pad)
        if if_use_unk:
            self.sign_unk = "<UNK>"
            self.add_instance(self.sign_unk)

    @property
    def pad_token(self):
        return self.sign_pad

    @property
    def unk_token(self):
        return self.sign_unk

    def index2instance(self):
        return self.index2instance

    def add_instance(self, instance):
        """ Add instances to alphabet.

        1, We support any iterative data structure which
        contains elements of str type.

        2, We will count added instances that will influence
        the serialization of unknown instance.

        :param instance: is given instance or a list of it.
        """

        if isinstance(instance, (list, tuple)):
            for element in instance:
                self.add_instance(element)
            return

        # We only support elements of str type.
        assert isinstance(instance, str)

        # count the frequency of instances.
        self.counter[instance] += 1

        if instance not in self.index2instance:
            self.instance2index[instance] = len(self.index2instance)
            self.index2instance.append(instance)

    def get_index(self, instance):
        """ Serialize given instance and return.

        For unknown words, the return index of alphabet
        depends on variable self.use_unk:

            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.

        :param instance: is given instance or a list of it.
        :return: is the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem) for elem in instance]

        assert isinstance(instance, str)

        try:
            return self.instance2index[instance]
        except KeyError:
            if self.if_use_unk:
                return self.instance2index[self.sign_unk]
            else:
                max_freq_item = self.counter.most_common(1)[0][0]
                return self.instance2index[max_freq_item]

    def get_instance(self, index):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        :param index: is query index, possibly iterable.
        :return: is corresponding instance.
        """

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.index2instance[index]

    def save_content(self, dir_path):
        """ Save the content of alphabet to files.

        There are two kinds of saved files:
            1, The first is a list file, elements are
            sorted by the frequency of occurrence.

            2, The second is a dictionary file, elements
            are sorted by it serialized index.

        :param dir_path: is the directory path to save object.
        """

        # Check if dir_path exists.
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        list_path = os.path.join(dir_path, self.name + "_list.txt")
        with open(list_path, 'w') as fw:
            for element, frequency in self.counter.most_common():
                fw.write(element + '\t' + str(frequency) + '\n')

        dict_path = os.path.join(dir_path, self.name + "_dict.txt")
        with open(dict_path, 'w') as fw:
            for index, element in enumerate(self.index2instance):
                fw.write(element + '\t' + str(index) + '\n')

    def __len__(self):
        return len(self.index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name, len(self), self.index2instance)


class TorchDataset(Dataset):
    def __init__(self, samples, word_alphabet, slot_alphabet, intent_alphabet, plm_tokenizer, char_level):
        super(TorchDataset, self).__init__()
        # transform data from "list of dict" to "dict of list"
        self.data = {}
        for key in samples[0].keys():
            self.data[key] = [sample[key] for sample in samples]

        self.num_total_seqs = len(self.data['text'])
        self.word_alphabet = word_alphabet
        self.slot_alphabet = slot_alphabet
        self.intent_alphabet = intent_alphabet

        self.tokenizer = plm_tokenizer
        self.char_level = char_level

    def __len__(self):
        return self.num_total_seqs

    def __getitem__(self, index):
        # preprocess regular information
        text = self.preprocess(list(self.data['text'][index]), self.word_alphabet, is_graph=False)
        slots = self.preprocess(self.data['slots'][index], self.slot_alphabet, is_graph=False)
        intent = self.intent_alphabet.get_index(self.data['intent'][index])

        # processed information
        data_info = {}
        for k in self.data.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data[k][index]

        # preprocess graph information
        graph_nodes = self.data['graph_nodes'][index]
        profile_span = self.data['profile_span'][index]
        kg_entity_as_token = self.data['kg_entity_as_token'][index]
        if "kg" in profile_span:
            kg_nodes = []
            for st, ed in profile_span["kg"]:
                kg_nodes.extend(graph_nodes[st: ed])
            # check whether the number of elements in each kg node is same
            ens = set([len(node) for node in kg_nodes])
            assert len(ens) == 1, f"the number of elements in each kg node should be same, but got '{ens}'."
            if kg_entity_as_token:
                kg_nodes = self.preprocess(kg_nodes, self.word_alphabet, is_graph=False)
                data_info['kg_nodes'] = (kg_nodes,)
            else:
                kg_texts, kg_element_idx, kg_digital_texts = self.preprocess(kg_nodes, self.word_alphabet, is_graph=True)
                data_info['kg_nodes'] = (kg_texts, kg_element_idx, kg_digital_texts)
        else:
            if kg_entity_as_token:
                data_info['kg_nodes'] = ([],)
            else:
                data_info['kg_nodes'] = ([], [], [])

        sp_nodes = []
        for st, ed in profile_span["up"] + profile_span["ca"]:
            sp_nodes.extend(graph_nodes[st: ed])
        # check whether the number of elements in each special node is same
        ens = set([len(node) for node in sp_nodes])
        assert len(ens) == 1, f"the number of elements in each special node should be same, but got '{ens}'."
        if len(sp_nodes[0]) > 1:
            sp_nodes = self.preprocess(sp_nodes, self.word_alphabet, is_graph=False)
            data_info['sp_nodes'] = (sp_nodes,)
        else:
            sp_texts, sp_element_idx, sp_digital_texts = self.preprocess(sp_nodes, self.word_alphabet, is_graph=True)
            data_info['sp_nodes'] = (sp_texts, sp_element_idx, sp_digital_texts)

        # additional plain information
        data_info['text_plain'] = self.data['text'][index]
        data_info['slots_plain'] = self.data['slots'][index]
        data_info['intent_plain'] = self.data['intent'][index]

        return data_info


    def preprocess(self, sequence, alphabet, is_graph=False):
        if is_graph:
            texts = OrderedSet(['<PAD>'])
            node_element_idx = []
            for i, node in enumerate(sequence):
                node_element_idx.append([])
                for j, element in enumerate(node):
                    if element == alphabet.pad_token:
                        node_element_idx[-1].append(0)
                    else:
                        tidx = texts.append(element)
                        node_element_idx[-1].append(tidx)
            texts = list(texts)
            # sort texts by length
            sorted_texts, ori_indices = zip(*sorted(zip(texts, range(len(texts))), key=lambda x: len(x[0]), reverse=True))
            mapping = dict(zip(ori_indices, range(len(ori_indices))))
            node_element_idx = apply_mapping(mapping, node_element_idx)

            digital_texts = [torch.LongTensor(alphabet.get_index(list(t))) for t in sorted_texts]
            return sorted_texts, node_element_idx, digital_texts

        else:
            digital_seq = alphabet.get_index(sequence)
        digital_seq = torch.LongTensor(digital_seq)
        return digital_seq

    def tokenize_texts(self, text_sequences, char_level=False, return_char_token_mapping=False):
        if char_level:
            text_char_sequences = [list(text) for text in text_sequences]
            piece_batch = []
            piece_spans = []
            # tokenize text_char_sequences
            for chars in text_char_sequences:
                tokenized_chars = self.tokenizer(chars, padding=False, truncation=False,
                                                 return_tensors=None, add_special_tokens=False)
                if isinstance(self.tokenizer, (XLNetTokenizerFast, XLNetTokenizer)):
                    piece_ids = reduce(lambda x, y: x + y, tokenized_chars["input_ids"]) + \
                                [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]
                else:
                    piece_ids = [self.tokenizer.cls_token_id] + reduce(
                        lambda x, y: x + y, tokenized_chars["input_ids"]) + [self.tokenizer.sep_token_id]
                piece_batch.append(piece_ids)
                piece_lens = [0] + [len(mask) for mask in tokenized_chars["attention_mask"]]
                piece_spans.append(np.cumsum(piece_lens).tolist()[:-1])
            # pad piece_batch
            max_len = max([len(piece) for piece in piece_batch])
            for i, piece in enumerate(piece_batch):
                piece_batch[i].extend([self.tokenizer.pad_token_id] * (max_len - len(piece)))
            return torch.LongTensor(piece_batch), piece_spans
        else:
            tokenized_texts = self.tokenizer(text_sequences, padding=True, truncation=True, return_tensors="pt",
                                             add_special_tokens=True, return_offsets_mapping=True,
                                             return_length=True, return_token_type_ids=False)
            if return_char_token_mapping:
                # get mapping from char index to token index
                char_token_mappings = []
                for i, offset_mapping in enumerate(tokenized_texts["offset_mapping"]):
                    char_token_mappings.append([])
                    token_idx = 0
                    for j, (start, end) in enumerate(offset_mapping):
                        if start == end:
                            continue
                        else:
                            char_token_mappings[-1].extend([token_idx] * (end - start))
                            token_idx += 1
                    tokens = [t for t in self.tokenizer.convert_ids_to_tokens(tokenized_texts["input_ids"][i])
                              if t not in [self.tokenizer.pad_token, self.tokenizer.cls_token,
                                           self.tokenizer.sep_token]]
                    assert token_idx == len(
                        tokens), f"token length {token_idx} does not match text length {len(tokens)}: '{text_sequences[i]}' -> '{tokens}'"
                    if isinstance(self.tokenizer, (XLNetTokenizerFast, XLNetTokenizer)):
                        # XLNet regular tokenization: input_ids + sep_token + cls_token
                        if len(char_token_mappings[-1]) == len(text_sequences[i]) + 1:
                            char_token_mappings[-1] = char_token_mappings[-1][1:]
                    # TODO: how to represent whitespaces
                    # '我想用 iPad 搜索古龙的 离别钩' -> [(0, 1), (1, 2), (2, 3), (4, 8), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (15, 16), (16, 17), (17, 18)]
                    assert len(text_sequences[i]) == len(char_token_mappings[-1]), \
                        f"char_token_mappings length {len(char_token_mappings[-1])} does not match text length " \
                        f"{len(text_sequences[i])}: '{text_sequences[i]}' -> '{tokens}'"
                return tokenized_texts["input_ids"], char_token_mappings
            else:
                # based on attention mask, get lengths of each tokenized text (excluding special tokens)
                token_lens = (tokenized_texts["attention_mask"].sum(dim=1) - 2).tolist()
                return tokenized_texts["input_ids"], token_lens

    def collate_fn(self, batch):
        def merge(sequences, is_graph=False, padding_value=0):
            if is_graph:
                plain_texts, digital_sequences = sequences
                lengths = [len(seq) for seq in digital_sequences]
                # divide sequences into multiple parts, based on the sequence lengths
                divide_ranges = divide_chunks(lengths, 2)
                plain_texts_list, padded_seqs_list, lengths_list = [], [], []
                for st, ed in divide_ranges:
                    if st < ed:
                        padded_seqs, lengths = merge(digital_sequences[st: ed], is_graph=False)
                        padded_seqs_list.append(_cuda(padded_seqs))
                        lengths_list.append(lengths)
                        plain_texts_list.append(plain_texts[st: ed])
                return plain_texts_list, padded_seqs_list, lengths_list
            else:
                lengths = [len(seq) for seq in sequences]
                max_len = 1 if max(lengths) == 0 else max(lengths)

                padded_seqs = torch.zeros(len(sequences), max_len).long().fill_(padding_value)

                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
                return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        batch.sort(key=lambda x: len(x['text']), reverse=True)
        item_info = {}
        for key in batch[0].keys():
            item_info[key] = [d[key] for d in batch]

        # merge sequences
        text, text_lengths = merge(item_info['text'], is_graph=False)
        slots, _ = merge(item_info['slots'], is_graph=False)
        intent = torch.LongTensor(item_info['intent']).squeeze(-1)

        text = _cuda(text)
        slots = _cuda(slots)
        intent = _cuda(intent)

        # merge graph nodes
        kg_nodes = []
        if len(item_info['kg_nodes'][0]) == 1:
            kg_nodes = [(_cuda(sample[0]) if len(sample[0]) > 0 else (),) for sample in item_info['kg_nodes']]
        else:
            for kg_text_plain, kg_element_idx, kg_digital_texts in item_info['kg_nodes']:
                if len(kg_text_plain) == 0:
                    kg_nodes.append(())
                    continue
                # divide kg texts into multiple parts, based on the kg text lengths
                kg_text_plain_list, kg_texts_list, kg_lengths_list = merge((kg_text_plain, kg_digital_texts), is_graph=True)
                kg_element_idx = _cuda(torch.LongTensor(kg_element_idx))
                kg_nodes.append((kg_texts_list, kg_lengths_list, kg_element_idx))
                if self.tokenizer is not None:
                    # tokenize kg nodes for pretrained language models
                    kg_tokenized_list, kg_tokenized_lengths_list = [], []
                    for kg_plain_chunk in kg_text_plain_list:
                        kg_tokenized, kg_tokenized_lengths = self.tokenize_texts(
                            kg_plain_chunk,
                            char_level=False,
                            return_char_token_mapping=False)
                        kg_tokenized_list.append(_cuda(kg_tokenized))
                        kg_tokenized_lengths_list.append(kg_tokenized_lengths)
                    kg_nodes[-1] += (kg_tokenized_list, kg_tokenized_lengths_list)

        # merge special nodes
        sp_nodes = []
        if len(item_info['sp_nodes'][0]) == 1:
            sp_nodes = [(_cuda(sample[0]),) for sample in item_info['sp_nodes']]
        else:
            for sp_text_plain, sp_element_idx, sp_digital_texts in item_info['sp_nodes']:
                sp_texts, sp_lengths = merge(sp_digital_texts, is_graph=False)
                sp_texts = _cuda(sp_texts)
                sp_element_idx = _cuda(torch.LongTensor(sp_element_idx))
                sp_nodes.append(([sp_texts], [sp_lengths], sp_element_idx))
                if self.tokenizer is not None:
                    # tokenize special nodes for pretrained language models
                    sp_tokenized, sp_tokenized_lengths = self.tokenize_texts(
                        sp_text_plain,
                        char_level=False,
                        return_char_token_mapping=False)
                    sp_tokenized = _cuda(sp_tokenized)
                    sp_nodes[-1] += ([sp_tokenized], [sp_tokenized_lengths])

        # leaf node ids, padding value is the last value
        max_num_leaf = max([len(sample) for sample in item_info['leaf_node_ids']])
        leaf_node_ids = []
        for sid, sample in enumerate(item_info['leaf_node_ids']):
            leaf_node_ids.append(sample + [sample[-1]] * (max_num_leaf - len(sample)))
        leaf_node_ids = _cuda(torch.LongTensor(leaf_node_ids))

        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        for i, length in enumerate(data_info['node_num']):
            assert length == len(data_info['graph_nodes'][i]), \
                f"node_num {length} does not match graph_nodes length {len(data_info['graph_nodes'][i])}"

        data_info["text_lengths"] = text_lengths

        # merge utterance node ids
        text_node_ids = _cuda(torch.LongTensor([ele[0] for ele in item_info['utterance_node_ids']]))
        data_info["text_node_ids"] = text_node_ids
        if len(item_info['utterance_node_ids'][0]) > 1:
            char_node_ids, _ = merge([torch.LongTensor(ele[1:]) for ele in item_info['utterance_node_ids']], is_graph=False)
            char_node_ids = _cuda(char_node_ids)
            data_info["char_node_ids"] = char_node_ids

        if self.tokenizer is not None:
            # tokenize user utterances for pretrained language models
            text_tokenized, text_tokenized_spans = self.tokenize_texts(
                data_info["text_plain"],
                char_level=self.char_level,
                return_char_token_mapping=True)
            data_info["text_tokenized"] = _cuda(text_tokenized)
            data_info["text_tokenized_spans"] = text_tokenized_spans

        return data_info


class DatasetManager(object):

    def __init__(self, args):
        # Instantiate alphabet objects.
        self.word_alphabet = Alphabet('word', if_use_pad=True, if_use_unk=True)
        self.slot_alphabet = Alphabet('slot', if_use_pad=False, if_use_unk=False)
        self.intent_alphabet = Alphabet('intent', if_use_pad=False, if_use_unk=False)

        self.pro_graph = ProGraph(args, self.word_alphabet.pad_token, self.word_alphabet.unk_token)

        # Record the raw data.
        self.split2data = {}

        self.args = args

    @property
    def num_epoch(self):
        return self.args.num_epoch

    @property
    def batch_size(self):
        return self.args.batch_size

    @property
    def learning_rate(self):
        return self.args.learning_rate

    @property
    def save_dir(self):
        return self.args.save_dir

    @property
    def slot_forcing_rate(self):
        return self.args.slot_forcing_rate

    def quick_build(self):
        """
        Convenient function to instantiate a dataset object.
        """
        file_path_template = lambda split: os.path.join(
            self.args.data_dir, f"{split}_rebuild.json")
        train_path = file_path_template('train')
        dev_path = file_path_template('dev')
        test_path = file_path_template('test')

        self.add_file('train', train_path, is_train=True)
        self.add_file('dev', dev_path, is_train=False)
        self.add_file('test', test_path, is_train=False)

        # Check if save path exists.
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Save alphabet objects.
        alphabet_dir = os.path.join(self.save_dir, "alphabet")
        self.word_alphabet.save_content(alphabet_dir)
        self.slot_alphabet.save_content(alphabet_dir)
        self.intent_alphabet.save_content(alphabet_dir)

    def quick_build_test(self, data_dir, file_name):
        test_path = os.path.join(data_dir, file_name)
        self.add_file('test', test_path, is_train=False)

    def add_file(self, split, file_path, is_train=False):
        """
        Add a data file to the dataset.
        """
        assert split in ['train', 'dev', 'test']
        self.split2data[split] = self.read_data_file(file_path)

        if is_train:
            for sample in self.split2data[split]:
                # Add characters in 'user utterance' to the word alphabet.
                self.word_alphabet.add_instance(list(sample['text']))
                # Add characters in graph nodes to the word alphabet.
                for node in sample["graph_nodes"]:
                    for word in node:
                        assert isinstance(word, str)
                        self.word_alphabet.add_instance(list(word))
                # Add special words to the word alphabet.
                self.word_alphabet.add_instance(sample['special_words'])

                # Add slots to the slot alphabet.
                self.slot_alphabet.add_instance(sample['slots'])
                # Add intents to the intent alphabet.
                self.intent_alphabet.add_instance(sample['intent'])

    def remove_intent_from_slot(self, slots, intent):
        new_slots = []
        for slot in slots:
            fields = slot.split('-')
            assert 0 < len(fields) <= 2, f"tag format wrong. it must be B-xxx.xxx or O, but got '{slot}'."
            if len(fields) == 1:
                # O
                new_slots.append(slot)
                continue
            intent_name, slot_name = fields[1].split('.')
            assert intent_name == intent, f"intent name '{intent_name}' in slot '{slot}' does not match '{intent}'."
            new_slots.append(f"{fields[0]}-{slot_name}")
        return new_slots

    def read_data_file(self, file_path):
        res_data = []
        with open(file_path, 'r', encoding="utf-8") as fin:
            other_info, data = json.load(fin)

            for sid, sample in data.items():
                # TODO: when loading data files, replace whitespace in texts with a predefined rare character
                slots = sample['slot'] if "slot_without_intent" not in self.args or not self.args.slot_without_intent \
                    else self.remove_intent_from_slot(sample['slot'], sample['intent'])
                data_detail = {
                    'text': sample['用户话语'],
                    'slots': slots,
                    'intent': sample['intent'],
                    'KG': sample['KG'],
                    'UP': sample['UP'],
                    'CA': sample['CA']
                }
                # create graph
                self.pro_graph.create_graph(data_detail)
                res_data.append(data_detail)

        return res_data

    def get_dataloader(self, split, plm_tokenizer, char_level=True, batch_size=None):
        """
        Get the dataloader for the specified split.
        """
        if batch_size is None:
            batch_size = self.batch_size
        dataset = TorchDataset(self.split2data[split], self.word_alphabet, self.slot_alphabet, self.intent_alphabet,
                               plm_tokenizer, char_level)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=split == 'train',
            collate_fn=dataset.collate_fn)
        return dataloader

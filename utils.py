#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Shiwen Ni"
# Date: 2021/12/12
import csv
import json
import random

class Datasets():
    def __init__(self, dataset_name=""):
        self.dataset_name = dataset_name
        self.patterns = []
        self.train_path, self.dev_path, self.test_path = "", "", ""
        if (dataset_name == 'SST-2'):
            self.train_path = r"./datasets/GLUE/SST-2/train.tsv"
            self.dev_path = r"./datasets/GLUE/SST-2/train.tsv"
            self.test_path = r"./datasets/GLUE/SST-2/dev.tsv"
            self.metric = 'Acc'
            self.label_texts = ['terrible','great']
            self.templates = ["This movie is [label]!!"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == 'CoLA'):
            self.train_path = r"./datasets/GLUE/CoLA/train.tsv"
            self.dev_path = r"./datasets/GLUE/CoLA/train.tsv"
            self.test_path = r"./datasets/GLUE/CoLA/dev.tsv"
            self.metric = 'Matthews'
            self.label_texts = ["wrong", "correct"]
            self.templates = ["The grammar of the following sentence is [label],"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == 'MR'):
            self.train_path = r"./datasets/others/MR/train.csv"
            self.dev_path = r"./datasets/others/MR/train.csv"
            self.test_path = r"./datasets/others/MR/test.csv"
            self.metric = 'Acc'
            self.label_texts = ["terrible", "great"]
            self.templates = ["It was [label]!"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == 'CR'):
            self.train_path = r"./datasets/others/CR/train.csv"
            self.dev_path = r"./datasets/others/CR/train.csv"
            self.test_path = r"./datasets/others/CR/test.csv"
            self.metric = 'Acc'
            self.label_texts = ["hate", "love"]
            self.templates = ["I really [label] this product."]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == 'MPQA'):
            self.train_path = r"./datasets/others/MPQA/train.csv"
            self.dev_path = r"./datasets/others/MPQA/train.csv"
            self.test_path = r"./datasets/others/MPQA/test.csv"
            self.metric = 'Acc'
            self.label_texts = ["not", "really"]
            self.templates = ["[label] good,"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == 'Subj'):
            self.train_path = r"./datasets/others/Subj/train.csv"
            self.dev_path = r"./datasets/others/Subj/train.csv"
            self.test_path = r"./datasets/others/Subj/test.csv"
            self.metric = 'Acc'
            self.label_texts = ['Objectively',"Subjectively"]
            self.templates = ["[label] speaking."]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == 'TREC'):
            self.train_path = r"./datasets/others/TREC/train.csv"
            self.dev_path = r"./datasets/others/TREC/train.csv"
            self.test_path = r"./datasets/others/TREC/test.csv"
            self.metric = 'Acc'
            self.label_texts = ["definition", "entity", "meaning", "person", "place", "number"]
            self.templates = ["The answer is about a [label],"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == 'SST-5'):
            self.train_path = r"./datasets/others/SST-5/train.csv"
            self.dev_path = r"./datasets/others/SST-5/train.csv"
            self.test_path = r"./datasets/others/SST-5/test.csv"
            self.metric = 'Acc'
            self.label_texts = ["terrible", "bad", "okay", "good", "perfect"]
            self.templates = ["This movie is [label]."]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "QQP"):
            self.train_path = r"./datasets/GLUE/QQP/train.tsv"
            self.dev_path = r"./datasets/GLUE/QQP/train.tsv"
            self.test_path = r"./datasets/GLUE/QQP/dev.tsv"
            self.labels = [0, 1]
            self.metric = 'F1'
            self.label_texts = ["no", "yes"]
            self.templates = ["? [label],"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "MRPC"):
            self.train_path = r"./datasets/GLUE/MRPC/msr_paraphrase_train.txt"
            self.dev_path = r"./datasets/GLUE/MRPC/msr_paraphrase_train.txt"
            self.test_path = r"./datasets/GLUE/MRPC/msr_paraphrase_test.txt"
            self.labels = [0, 1]
            self.metric = 'F1'
            self.label_texts = ["no", "yes"]
            self.templates = ["[label],"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "QNLI"):
            self.train_path = r"./datasets/GLUE/QNLI/train.tsv"
            self.dev_path = r"./datasets/GLUE/QNLI/train.tsv"
            self.test_path = r"./datasets/GLUE/QNLI/dev.tsv"
            self.metric = 'Acc'
            self.text2id = {"not_entailment": 0, "entailment": 1}
            self.labels = [0, 1]
            self.label_texts = ["no", "yes"]
            self.templates = ["? [label]!"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "WNLI"):
            self.train_path = r"./datasets/GLUE/WNLI/train.tsv"
            self.dev_path = r"./datasets/GLUE/WNLI/train.tsv"
            self.test_path = r"./datasets/GLUE/WNLI/dev.tsv"
            self.metric = 'Acc'
            self.labels = [0, 1]
            self.label_texts = ["no", "yes"]
            self.templates = ["? [label],"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "MNLI-mm"):
            self.train_path = r"./datasets/GLUE/MNLI/train.tsv"
            self.dev_path = r"./datasets/GLUE/MNLI/dev_matched.tsv"
            self.test_path = r"./datasets/GLUE/MNLI/dev_matched.tsv"
            self.metric = 'Acc'
            self.text2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
            self.labels = [0, 1, 2]
            self.label_texts = ["no", "maybe", "yes"]
            self.templates = ["? [label],"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "MNLI"):
            self.train_path = r"./datasets/GLUE/MNLI/train.tsv"
            self.dev_path = r"./datasets/GLUE/MNLI/dev_mismatched.tsv"
            self.test_path = r"./datasets/GLUE/MNLI/dev_mismatched.tsv"
            self.metric = 'Acc'
            self.text2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
            self.labels = [0, 1, 2]
            self.label_texts = ["no", "maybe", "yes"]
            self.templates = ["? [label],"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "SNLI"):
            self.train_path = r"./datasets/others/SNLI/train.tsv"
            self.dev_path = r"./datasets/others/SNLI/dev.tsv"
            self.test_path = r"./datasets/others/SNLI/test.tsv"
            self.metric = 'Acc'
            self.text2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
            self.labels = [0, 1, 2]
            self.label_texts = ["no", "maybe", "yes"]
            self.templates = ["? [label],"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "RTE"):
            self.train_path = r"./datasets/GLUE/RTE/train.tsv"
            self.dev_path = r"./datasets/GLUE/RTE/train.tsv"
            self.test_path = r"./datasets/GLUE/RTE/dev.tsv"
            self.metric = 'Acc'
            self.text2id = { "not_entailment": 0, "entailment": 1}
            self.labels = [0, 1]
            self.label_texts = ["no", "yes"]
            self.templates = ["? [label]!"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "STS-B"):
            self.train_path = r"./datasets/GLUE/STS-B/train.tsv"
            self.dev_path = r"./datasets/GLUE/STS-B/train.tsv"
            self.test_path = r"./datasets/GLUE/STS-B/dev.tsv"
            self.metric = 'Pear'
            self.label_texts = ["no"]
            self.templates = ["? [label]!!"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

    def load_data(self, filename, sample_num=-1, is_train=False, is_shuffle=False):
        D = []
        if (self.dataset_name == "QQP"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-3]
                    text_b = rows[-2]
                    label = rows[-1]
                    text_a = text_a 
                    D.append((text_a + "[SEP]" + text_b, int(label)))

        elif (self.dataset_name == "MRPC"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-1]
                    text_b = rows[-2]
                    label = rows[0]
                    text_a = text_a 
                    D.append((text_a + "[SEP]" + text_b, int(label)))

        elif (self.dataset_name == "QNLI"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-3]
                    text_b = rows[-2]
                    label = rows[-1]
                    text_a = text_a
                    D.append((text_a + "[SEP]" + text_b, self.text2id[label] ))

        elif (self.dataset_name == "WNLI"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-3]
                    text_b = rows[-2]
                    label = rows[-1]
                    D.append(("Sentence 1:" +text_a + "[SEP]" + "Sentence 2:" +text_b, int(label)))

        elif (self.dataset_name == "MNLI"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-8]
                    text_b = rows[-7]
                    label = rows[-1]
                    text_a = text_a 
                    D.append(( text_a + "[SEP]"+ text_b, self.text2id[label]))

        elif (self.dataset_name == "MNLI-mm"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-8]
                    text_b = rows[-7]
                    label = rows[-1]
                    text_a = text_a
                    D.append((text_a + "[SEP]" + text_b, self.text2id[label]))

        elif (self.dataset_name == "SNLI"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-8]
                    text_b = rows[-7]
                    label = rows[-1]
                    text_a = text_a 
                    D.append((text_a + "[SEP]" + text_b, self.text2id[label]))

        elif (self.dataset_name == "RTE"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-3]
                    text_b = rows[-2]
                    label = rows[-1]
                    D.append((text_a + "[SEP]" + text_b , self.text2id[label]))

        elif (self.dataset_name == "CoLA"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text = rows[-1]
                    label = rows[-3]
                    D.append((text, int(label)))

        elif (self.dataset_name == "STS-B"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-3]
                    text_b = rows[-2]
                    score = rows[-1]
                    text_a = text_a 
                    D.append((text_a + "[SEP]" + text_b, float(score)))

        elif (self.dataset_name == "SST-2"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text = rows[-2]
                    label = rows[-1]
                    D.append((text, int(label)))

        elif (self.dataset_name == "SST-5"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    text = l[2:]
                    label = l[0]
                    D.append((text, int(label)))

        elif (self.dataset_name == "MR"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    text = l[2:]
                    text = text.lstrip('"').rstrip('"')
                    label = l[0]
                    D.append((text, int(label)))

        elif (self.dataset_name == "CR"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    text = l[2:]
                    text = text.lstrip('"').rstrip('"')
                    label = l[0]
                    D.append((text, int(label)))

        elif (self.dataset_name == "MPQA"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    text = l[2:]
                    text = text.lstrip('"').rstrip('"')
                    label = l[0]
                    D.append((text, int(label)))

        elif (self.dataset_name == "Subj"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    text = l[2:]
                    text = text.lstrip('"').rstrip('"')
                    label = l[0]
                    D.append((text, int(label)))

        elif (self.dataset_name == "TREC"):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    text = l[2:]
                    text = text.lstrip('"').rstrip('"')
                    label = l[0]
                    D.append((text, int(label)))

        # Shuffle the dataset.
        if (is_shuffle):
            random.seed(1)
            random.shuffle(D)

        # Set the number of samples.
        if (sample_num == -1):
            # -1 for all the samples
            return D
        else:
            return D[:sample_num + 1]


class Model():

    def __init__(self, model_name=""):
        self.model_name = model_name
        self.config_path, self.checkpoint_path, self.dict_path = "", "", ""

        if (model_name == 'electra-small'):
            self.config_path = './models/electra_small/small_discriminator_config.json'
            self.checkpoint_path = './models/electra_small/electra_small'
            self.dict_path = './models/electra_small/vocab.txt'

        elif (model_name == 'electra-base'):
            self.config_path = './models/electra_base/base_discriminator_config.json'
            self.checkpoint_path = './models/electra_base/electra_base'
            self.dict_path = './models/electra_base/vocab.txt'

        elif (model_name == 'electra-large'):
            self.config_path = './models/electra_large/large_discriminator_config.json'
            self.checkpoint_path = './models/electra_large/electra_large'
            self.dict_path = './models/electra_large/vocab.txt'


def read_labels(label_file_path):
    labels_text = []
    text2id = {}
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            label = line.strip('\n')
            labels_text.append(label)
            text2id[label] = index
    return labels_text, text2id


def sample_dataset(data: list, k_shot: int, label_num=-1):
    if(k_shot==-1):
        return data
    label_set = set()
    label2samples = {}
    for d in data:
        (text, label) = d
        label_set.add(label)
        if (label in label2samples):
            label2samples[label].append(d)
        else:
            label2samples[label] = [d]
    if (label_num != -1):
        assert len(label_set) == label_num
    new_data = []
    for label in label_set:
        if (isinstance(label, float)):
            random.seed(0)
            new_data = random.sample(data, k_shot)
            random.shuffle(new_data)
            return new_data
        random.seed(0)
        new_data += random.sample(label2samples[label], k_shot)
    random.seed(0)
    random.shuffle(new_data)
    return new_data

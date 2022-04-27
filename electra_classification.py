#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Shiwen Ni"
# Date: 2021/12/14

import numpy
from tqdm import tqdm
from sklearn import metrics
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from utils import *
import os
# Choose which GPU card to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class data_generator(DataGenerator):
    """Data Generator"""
    def __init__(self, pattern="", is_pre=True, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.pattern = pattern
        self.is_pre = is_pre
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if (self.is_pre):
                token_ids, segment_ids = tokenizer.encode(first_text=self.pattern, second_text=text, maxlen=maxlen)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=text, second_text=self.pattern, maxlen=maxlen)
            source_ids, target_ids = token_ids[:], token_ids[:]
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids, batch_output_ids], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


def evaluate(data_generator_list, data, note=""):
    print("\n*******************Start to Zero-Shot predict on 【{}】*******************".format(note), flush=True)
    patterns_logits = [[] for _ in patterns]
    for i in range(len(data_generator_list)):
        print("\nPattern{}".format(i), flush=True)
        data_generator = data_generator_list[i]
        counter = 0
        for (x, _) in tqdm(data_generator):
            outputs = model.predict(x[:2])
            loc_all = []
            for tokens_batch in x[:1]:
                for tokens in tokens_batch:
                    loc = tokens.tolist().index(102) + 2   # ⭐️ Locate label words,  [CLS]: 101  [SEP]: 102
                    loc_all.append(loc)
            for (out,loc) in zip(outputs, loc_all):
                logit_pos = (out[loc].T)      # [CLS]sentence1[SEP]sentence2[SEP] 
                patterns_logits[i].append(logit_pos)
                counter += 1


    # Evaluate the results
    trues = [d[1] for d in data]
    preds = []
    for i in range(len(patterns_logits[0])):
        pred = numpy.argmin([logits[i] for logits in patterns_logits])  # ⭐️ min
        preds.append(int(pred))

    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
    print("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
    if (dataset.metric == 'Matthews'):
        matthews_corrcoef = metrics.matthews_corrcoef(trues, preds)
        print("Matthews Corrcoef:\n{}".format(matthews_corrcoef), flush=True)
    else:
        acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
        print("Acc.:\t{:.4f}".format(acc), flush=True)
        return acc


if __name__ == "__main__":

    # Load the hyper-parameters-----------------------------------------------------------
    maxlen = 256  # The max length 128 is used in our paper
    batch_size = 40  # Will not influence the results

    # Choose a dataset----------------------------------------------------------------------
    # dataset_names = ['CoLA', 'SST-2', 'SST-5', 'MR', 'CR', 'MPQA', 'Subj', 'TREC']
    dataset_name = 'MPQA'

    # Choose a model----------------------------------------------------------------------
    # model_names = ['electra-small', 'electra-base', 'electra-large']
    model_name = 'electra-large'

    # Load model and dataset class
    pre_model = Model(model_name=model_name)
    dataset = Datasets(dataset_name=dataset_name)

    # Choose a template------------------------------------------------------------------
    patterns = dataset.patterns[0]
    # Prefix or Suffix-------------------------------------------------------------------
    is_pre = False

    # Load the dev set--------------------------------------------------------------------
    # -1 for all the samples
    dev_data = dataset.load_data(dataset.dev_path, sample_num=-1, is_shuffle=True)
    dev_data = sample_dataset(dev_data, 16)
    dev_generator_list = []
    for p in patterns:
        dev_generator_list.append(data_generator(pattern=p, is_pre=is_pre, data=dev_data, batch_size=batch_size))

    # Load the test set--------------------------------------------------------------------
    # -1 for all the samples
    test_data = dataset.load_data(dataset.test_path, sample_num=-1, is_shuffle=True)
    test_generator_list = []
    # class * numbers of data
    for p in patterns:
        test_generator_list.append(data_generator(pattern=p, is_pre=is_pre, data=test_data, batch_size=batch_size))

    # Build ELECTRA model---------------------------------------------------------------------
    tokenizer = Tokenizer(pre_model.dict_path, do_lower_case=True)
    # Load ELECTRA model with RTD head

    model = build_transformer_model(
        config_path=pre_model.config_path, 
        checkpoint_path=pre_model.checkpoint_path, 
        model='electra', with_discriminator=True,
    )
    print(model_name, model.summary())
    # Zero-Shot predict and evaluate-------------------------------------------------------
    # evaluate(dev_generator_list, dev_data, note="Dev Set")
    evaluate(test_generator_list, test_data, note="Test Set")

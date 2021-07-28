import re

import torch
from fastai.callback.tracker import SaveModelCallback
from fastai.text.learner import language_model_learner
from fastai.text.models import AWD_LSTM
from pathlib import Path
from pandas.io import pickle
from sklearn.datasets._base import load_data


from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertTokenizer

from ..BERT_WSD.script.utils.dataset import GlossSelectionRecord, _create_features_from_records
from ..BERT_WSD.script.utils.model import BertWSD, forward_gloss_selection
from ..BERT_WSD.script.utils.wordnet import get_glosses



from fastai.text.all import *


# import torch
# from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
# import pandas as pd
# import math
# from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
# from arabert.preprocess import ArabertPreprocessor


MAX_SEQ_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_wsd_predictions(model, tokenizer, sentence):
    re_result = re.search(r"\[TGT\](.*)\[TGT\]", sentence)
    if re_result is None:
        print("\nIncorrect input format. Please try again.")
        return

    ambiguous_word = re_result.group(1).strip()
    sense_keys = []
    definitions = []
    for sense_key, definition in get_glosses(ambiguous_word, None).items():
        sense_keys.append(sense_key)
        definitions.append(definition)

    record = GlossSelectionRecord("test", sentence, sense_keys, definitions, [-1])
    features = _create_features_from_records([record], MAX_SEQ_LENGTH, tokenizer,
                                             cls_token=tokenizer.cls_token,
                                             sep_token=tokenizer.sep_token,
                                             cls_token_segment_id=1,
                                             pad_token_segment_id=0,
                                             disable_progress_bar=True)[0]

    with torch.no_grad():
        logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
        for i, bert_input in tqdm(list(enumerate(features)), desc="Progress"):
            logits[i] = model.ranking_linear(
                model.bert(
                    input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
                )[1]
            )
        scores = softmax(logits, dim=0)

    return sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True)


def load_bert_wsd_model(model_dir):
    print("Loading model...")

    model = BertWSD.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    model.to(DEVICE)
    model.eval()
    return model, tokenizer





# def load_arabert_lm_model(model_dir):
#     print("Loading model...")
#
#     model_name = "aubmindlab/bert-large-arabertv02"
#     lm_bertMaskedLM = AutoModelForMaskedLM.from_pretrained(model_name)
#     lm_tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#
#     model = BertWSD.from_pretrained(model_dir)
#     tokenizer = BertTokenizer.from_pretrained(model_dir)
#
#     model.to(DEVICE)
#     model.eval()
#     return model, tokenizer
#
# def get_score(sentence, lm_tokenizer, lm_bertMaskedLM):
#     tokenize_input = lm_tokenizer.tokenize(sentence)
#     tensor_input = torch.tensor([lm_tokenizer.convert_tokens_to_ids(tokenize_input)])
#     predictions=lm_bertMaskedLM(tensor_input)
#     #print(predictions[0])
#     loss_fct = torch.nn.CrossEntropyLoss()
#     loss = loss_fct(predictions[0].squeeze(),tensor_input.squeeze()).data
#     #print(loss.data)
#     return math.exp(loss)
#


# model_name = "aubmindlab/bert-base-arabertv2"
# lm_bertMaskedLM = None #AutoModelForMaskedLM.from_pretrained(model_name)
# lm_tokenizer = None #AutoTokenizer.from_pretrained(model_name)
#
# def get_score(sentence):
#     tokenize_input = lm_tokenizer.tokenize(sentence)
#     tensor_input = torch.tensor([lm_tokenizer.convert_tokens_to_ids(tokenize_input)])
#     predictions=lm_bertMaskedLM(tensor_input)
#     #print(predictions[0])
#     loss_fct = torch.nn.CrossEntropyLoss()
#     loss = loss_fct(predictions[0].squeeze(),tensor_input.squeeze()).data
#     return math.exp(loss)


import random
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import functional as F

from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModel


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            return ind, ind + sll - 1


# 0. settings
# for training and model related
model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
per_device_train_batch_size = 16
per_device_eval_batch_size = 64
model_name_short = "blueBERTnorm"
num_train_epochs = 3
bert_frozen = False
train_model = True
checkpoint_path = "rd-fine-tune-ckpts-and-res"
# for mention representation inside the model
use_doc_struc = False
window_size = 5
masking = False
saved_model_path = "./fine-tuned-rd-ph-model%s%s" % (
    "-masked" if masking else "",
    "-ds" if use_doc_struc else "",
)  # saved best model path

fill_data = True  # if True, fill the prediction into the .xlsx file

# 1. load data, encoding, and train model
# load data

df = pd.read_csv("window_size_16.csv")

num_sample_tr = len(df)  # 9000 #len(data_list_tuples) #500 #len(data_list_tuples) #1000
num_sample_eval = 2000

df_train, df_test = train_test_split(df, test_size=0.1, random_state=1234)


data_list_tuples_cw_train = df_train[
    0:num_sample_tr
]  # only train on a fixed part of randomly shuffled data
data_list_tuples_cw_valid_in_train = df_test[
    0:num_sample_eval
]  # only eval (in train) on a fixed part of randomly shuffled data

print(len(data_list_tuples_cw_valid_in_train))


class DatasetMentFiltGen(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_list_tuples_cw, use_doc_struc, verbo=True):
        "Initialization"

        self.list_sent_cw = [
            data_tuple_cw[0] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != ""
        ]
        self.list_doc_struc = [
            data_tuple_cw[3] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != ""
        ]
        self.mention = [
            data_tuple_cw[4] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != ""
        ]
        self.list_bi_labels = [
            data_tuple_cw[5] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != ""
        ]

        if verbo:
            print(
                "original",
                len(data_list_tuples_cw),
                "after eliminating empty sents",
                len(self.list_sent_cw),
            )
            print(self.list_sent_cw[0:5])
            print(self.list_bi_labels[0:5])
            print("pos:", self.list_bi_labels.count(1), "neg:", self.list_bi_labels.count(0))

        # tokenisation
        # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-models/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
        )
        if use_doc_struc:
            self.encodings = tokenizer(
                self.list_doc_struc, self.list_sent_cw, truncation=True, padding=True
            )
        else:
            self.encodings = tokenizer(self.list_sent_cw, truncation=True, padding=True)
        # print('self.encodings:',self.encodings.items())
        self.labels = [int(bool_label) for bool_label in self.list_bi_labels]

        # get the new mention offsets after the tokenisation
        mention_encodings = tokenizer(
            self.mention, add_special_tokens=False, truncation=True, padding=False
        )
        # print(mention_encodings)
        # to do

        token_input_ids_cw = self.encodings["input_ids"]
        token_input_ids_men = mention_encodings["input_ids"]
        # print('token_input_ids_cw',token_input_ids_cw)
        self.list_offset_tuples = [
            find_sub_list(list_men_token_id, list_cw_token_id)
            for list_men_token_id, list_cw_token_id in zip(token_input_ids_men, token_input_ids_cw)
        ]
        # print(self.list_offset_tuples)

        self.list_offset_tuples = [
            offset_tuple if offset_tuple != None else (0, 0)
            for offset_tuple in self.list_offset_tuples
        ]

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_sent_cw)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Load data and get label
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        item["begin_offsets"] = self.list_offset_tuples[index][0]
        item["end_offsets"] = self.list_offset_tuples[index][1]
        return item


train_dataset = DatasetMentFiltGen(data_list_tuples_cw_train, use_doc_struc)
val_for_tr_dataset = DatasetMentFiltGen(data_list_tuples_cw_valid_in_train, use_doc_struc)

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
# model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
# model = AutoModelForSequenceClassification.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")


class mentionBERT(nn.Module):
    def __init__(self):
        super(mentionBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)

        self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.Tanh()

        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.num_labels)
        # self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        begin_offsets=0,
        end_offsets=0,  # add begin and ending offsets as input here
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,  # here as True
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states
        )

        hidden_states = (
            outputs.hidden_states
        )  # a tuple of k+1 layers (k=12 for bert-base) and each is a shape of (batch_size,sent_token_len,hidden_note_size)
        second_to_last_layer_hs = hidden_states[-2]

        sequence_output_cont_emb = [
            torch.mean(second_to_last_layer_hs[ind][offset_start : offset_end + 1], dim=0)
            for ind, (offset_start, offset_end) in enumerate(zip(begin_offsets, end_offsets))
        ]  # here offset_end needs to add 1, since the offsets from find_sub_list() subtracted 1 for the end offset
        # print('sequence_output_cont_emb',sequence_output_cont_emb)
        sequence_output_cont_emb = torch.stack(sequence_output_cont_emb)

        # here also has a dense layer with tanh activation - HD
        pooled_output = self.dense(sequence_output_cont_emb)
        pooled_output = self.activation(pooled_output)

        logits = self.classifier(pooled_output)

        loss_fct = CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return (loss,) + output


model = mentionBERT()

# if keep the model part frozen - see https://huggingface.co/transformers/training.html
if bert_frozen:
    for param in model.bert.base_model.parameters():
        param.requires_grad = False


def compute_metrics(pred):
    labels = pred.label_ids
    # print('pred.predictions',pred.predictions)

    preds = pred.predictions[0].argmax(
        -1
    )  # get the binary prediction from the softmax output # here get the first element of predictions as the hidden_states are also predicted
    # preds = (pred.predictions[0] > 0).astype(int) # for BCEWithLogitsLoss
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    # acc = accuracy_score(labels, preds)
    confusion_mat_tuple = confusion_matrix(labels, preds).ravel()
    # only unpack the confusion matrix when there are enough to unpack
    if len(confusion_mat_tuple) == 4:
        tn, fp, fn, tp = confusion_mat_tuple
    else:
        tn, fp, fn, tp = None, None, None, None
    return {
        #'accuracy': acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if train_model:
    # using with the parameters in https://huggingface.co/transformers/custom_datasets.html#fine-tuning-with-trainer
    # for a full list of the arguments, see https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir=checkpoint_path,  #'./results',          # output directory
        overwrite_output_dir=True,  # If True, overwrite the content of the output directory.
        num_train_epochs=num_train_epochs,  # total number of training epochs
        per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=500,  # logging step
        evaluation_strategy="steps",  # eval by step
        eval_accumulation_steps=1,  # accumulate results to CPU every k steps to save memoty
        save_strategy="epoch",  # save model every epoch
        load_best_model_at_end=True,  # load the best model at end
        metric_for_best_model="f1",  # with metric F1
    )

    trainer = Trainer(
        model=model,  # the instantiated ðŸ¤— Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_for_tr_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # trainer.train(checkpoint_path + '/checkpoint-3792') #train from a certain checkpoint
    trainer.save_model(saved_model_path)
else:
    training_args = TrainingArguments(
        output_dir=checkpoint_path,  #'./results',          # output directory
        num_train_epochs=0.00001,  # total number of training epochs # set this to an extremely small number
        per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=500,  # logging step
        evaluation_strategy="steps",  # eval by step
        eval_accumulation_steps=1,  # accumulate results to CPU every k steps to save memoty
        save_strategy="epoch",  # save model every epoch
        load_best_model_at_end=True,  # load the best model at end
        metric_for_best_model="f1",  # with metric F1
    )
    trainer = Trainer(
        model=model,  # the instantiated ðŸ¤— Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_for_tr_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )
    # trainer.train(checkpoint_path + '/checkpoint-3792') #eval with a certain checkpoint
    trainer.train(saved_model_path)

print(
    trainer.evaluate()
)  # see https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.predict

# 2. load testing data and predict results:
# load data from .xlsx and save the results to a specific column
# get a list of data tuples from an annotated .xlsx file
# data format: a 6-element tuple of section_text, document_structure_name, mention_name, UMLS_code, UMLS_desc, label (True or False)
df = pd.read_excel("for validation - SemEHR ori.xlsx")
# change nan values into empty strings in the two rule-based label columns
df[["neg label: only when both rule 0", "pos label: both rules applied"]] = df[
    ["neg label: only when both rule 0", "pos label: both rules applied"]
].fillna("")
# if the data is not labelled, i.e. nan, label it as -1 (not positive or negative)
df[["manual label from ann1"]] = df[["manual label from ann1"]].fillna(-1)

data_list_tuples_valid = []
data_list_tuples_test = []
for i, row in df.iterrows():
    doc_struc = row["document structure"]
    text = row["Text"]
    mention = row["mention"]
    UMLS_code = row["UMLS with desc"].split()[0]
    UMLS_desc = " ".join(row["UMLS with desc"].split()[1:])
    label = row["gold text-to-UMLS label"]
    label = 0 if label == -1 else label  # assume that the inapplicable (-1) entries are all False.
    data_tuple = (text, doc_struc, mention, UMLS_code, UMLS_desc, label)
    if i < 400:
        data_list_tuples_valid.append(data_tuple)
    else:
        data_list_tuples_test.append(data_tuple)
data_list_tuples_test_whole = data_list_tuples_valid + data_list_tuples_test
print(
    "valid data %s, test data %d, whole eval data %s"
    % (len(data_list_tuples_valid), len(data_list_tuples_test), len(data_list_tuples_test_whole))
)

data_list_tuples_cw_valid = get_context_window_from_data_list_tuples(
    data_list_tuples_valid, window_size=window_size, masking=masking
)
data_list_tuples_cw_test = get_context_window_from_data_list_tuples(
    data_list_tuples_test, window_size=window_size, masking=masking
)
data_list_tuples_cw_test_whole = get_context_window_from_data_list_tuples(
    data_list_tuples_test_whole, window_size=window_size, masking=masking
)

valid_dataset = DatasetMentFiltGen(data_list_tuples_cw_valid, use_doc_struc)
test_dataset = DatasetMentFiltGen(data_list_tuples_cw_test, use_doc_struc)
test_whole_dataset = DatasetMentFiltGen(data_list_tuples_cw_test_whole, use_doc_struc)

print(trainer.evaluate(valid_dataset))
print(trainer.evaluate(test_dataset))
print(trainer.evaluate(test_whole_dataset))

predictions, _, metrics = trainer.predict(
    test_whole_dataset
)  # see https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.predict
# print(predictions,metrics)

# fill results to the excel sheet - to do (to record results of bluebert-base-fine-tune for non-mask and mask).
list_ind_empty_cw_whole_test = [
    ind
    for ind, datalist_tuple_cw_ in enumerate(data_list_tuples_cw_test_whole)
    if datalist_tuple_cw_[0] == ""
]
y_pred_test = predictions[0].argmax(-1)
# for ind, data_list_tuple in enumerate(data_list_tuples_cw_test_whole): # to do to make this by batch
# if data_list_tuple[0] != '':
# test_dataset_from_one_tuple = DatasetMentFiltGen([data_list_tuple], use_doc_struc, verbo=False) # not verbo here
# pred_,_,_ = trainer.predict(test_dataset_from_one_tuple)
# pred = pred_[0].argmax(-1)[0]
# else:
# pred = 0
# y_pred_test.append(pred)
print("y_pred_test", y_pred_test)

if fill_data:
    # fill the prediction into the .xlsx file
    result_column_name = "model %s%s hf prediction%s%s%s" % (
        model_name_short,
        "finetune" if bert_frozen else "",
        " (masked training)" if masking else "",
        " ds" if use_doc_struc else "",
        " tr%s" % str(num_sample_tr) if num_sample_tr < len(data_list_tuples) else "",
    )
    if not result_column_name in df.columns:
        df[result_column_name] = ""
    ind_y_pred_test = 0
    for i, row in df.iterrows():
        if i in list_ind_empty_cw_whole_test:
            continue
        if row[result_column_name] != y_pred_test[ind_y_pred_test]:
            print(
                "row %s results changed %s to %s"
                % (str(i), row[result_column_name], y_pred_test[ind_y_pred_test])
            )
        df.at[i, result_column_name] = y_pred_test[ind_y_pred_test]
        ind_y_pred_test = ind_y_pred_test + 1
    df.to_excel(
        "for validation - SemEHR ori - hf - predicted%s%s.xlsx"
        % (" - masked" if masking else "", " - ds" if use_doc_struc else ""),
        index=False,
    )
    # hf stands for huggingface, it actually means fine-tuning.

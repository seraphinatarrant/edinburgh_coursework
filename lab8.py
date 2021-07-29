import argparse
import itertools
import re
import os
import json
import pickle

import numpy as np
import spacy
from spacy import displacy
from spacy.language import Language
from spacy.pipeline import Pipe
from pathlib import Path
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing

### Globals strings ###
ONTONOTES_LABELS = [
    'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL',
    'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

ontonotes_json = "data/ontonotes5_reduced.json"


# TODO have a longer span of text for the second part of this question
example_text = "On March 8, 2021, a group of hackers including Kottmann and calling themselves " \
       "'APT - 69420 Arson Cats' gained 'super admin' rights in the network of Verkada, a " \
       "cloud-based security camera company, using credentials they found on the public " \
       "internet. They had access to the network for 36 hours. The group collected about 5 " \
       "gigabytes of data, including live security camera footage and recordings from more" \
       " than 150,000 cameras in places like a Tesla factory, a jail in Alabama, a Halifax " \
       "Health hospital, and residential homes. The group also accessed a list of Verkada " \
       "customers and the company's private financial information, and gained superuser " \
       "access to the corporate networks of Cloudflare and Okta through their Verkada cameras."

# Choices of combination function for part 3: average, sum, first, last, maxpool
COMBINATION_FUNCTION = "average"

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['1','2','3','4'], required=True)
    p.add_argument('--ents', choices=ONTONOTES_LABELS, nargs='+')
    p.add_argument('--viz_output', default='entity_viz_example.html',
                   help='Name of output file for the visualisation')
    p.add_argument('--embed_func', choices=['first', 'last', 'sum', 'max', 'avg'],
                   help='the type of function to use to combine multiple embeddings into one entity'
                        'representation')
    p.add_argument('--corpus', help='name of corpus file to load in')
    p.add_argument('--classifier_path', help='name for path to save classifier')
    p.add_argument('--test', action='store_true', help='whether to print metrics on test set')
    return p.parse_args()

#####

def part_1(args, nlp):
    doc = nlp(example_text)
    ent_list, output_file = args.ents, args.viz_output
    options = {"ents": ent_list} if ent_list else {"ents": ONTONOTES_LABELS}

    html = displacy.render(doc, style="ent", options=options)

    output_path = Path(output_file)
    output_path.open("w").write(html)


def part_2(args, nlp):
    # These are special characters used by the tokenizer, ignore them
    special_chars = re.compile("Ġ|<pad>|<s>|</s>")
    doc = nlp(example_text)

    print("List of Entities:")
    print(doc.ents)

    print("\nStandard Tokenisation:")
    print(" ".join([tok.text for tok in doc]))

    print("\nSubword Tokenisation:")
    subword_string = " ".join([tok for tok in itertools.chain(*doc._.trf_data.wordpieces.strings)])
    cleaned_subword_string = special_chars.sub("", subword_string).strip()

    print(cleaned_subword_string)

### This is for Part 3 ###
class ContextualVectors(Pipe):
    def __init__(self, nlp):
        self._nlp = nlp
        self.combination_function = COMBINATION_FUNCTION ### modify this here for different versions of part 3

    def __call__(self, doc):
        if type(doc) == str:
            doc = self._nlp(doc)
        self.lengths = doc._.trf_data.align.lengths
        self.tensors = doc._.trf_data.tensors
        doc.user_token_hooks["vector"] = self.vector
        return doc

    ### HERE is where vectors are set
    def vector(self, token):
        trf_vector = []
        for len_idx in range(self.lengths[token.i]):
            try:
                trf_vector.append(self.tensors[0][0][token.i+len_idx])
            except IndexError:
                print("Error")
                print(token)
                return None
        trf_vector = np.array(trf_vector)
        return self.combine_vectors(trf_vector)

    def combine_vectors(self, trf_vector):
        if self.combination_function == "first":
            return trf_vector[0]
        if self.combination_function == "last":
            return trf_vector[-1]
        if self.combination_function == "maxpool":
            return np.maximum(trf_vector, axis=0)
        if self.combination_function == "average":
            return np.average(trf_vector, axis=0)
        if self.combination_function == "sum":
            return np.sum(trf_vector, axis=0)


@Language.factory("trf_vector_hook", assigns=["doc.user_token_hooks"])
def create_contextual_hook(nlp, name):
    return ContextualVectors(nlp)


def part_3(args, nlp):

    nlp.add_pipe("trf_vector_hook", last=True)
    max_tok = 145  # max tokens per chunk
    def chunks(tokens, n):
        for i in range(0, len(tokens), n):
            yield tokens[i:i+n]


    with open(ontonotes_json) as fin:
        f = json.load(fin)
    # process all the data
    corpus = dict.fromkeys(f.keys())
    for key in f.keys():
        #print("loading {}".format(key))
        embeddings, labels = [], []
        corpus_split = f[key]
        for entry in tqdm(corpus_split, desc=f"Processing {key}"):
            if not entry.get("entities"):
                continue
            this_string = entry["text"]
            # BERT max is 512 wordpiece tokens at once
            if len(this_string.split()) > max_tok:
                text_chunks = chunks(this_string, max_tok)
            else:
                text_chunks = [this_string]
            for c in text_chunks:
                this_doc = nlp("".join(c))

                # TODO if want gold labels, now have to get the entity that is the offset in order to make it gold label? ONTONOTES json is 'entities': {'DATE': [[177, 186]]}
                # TODO use start_char and end_char on both?
                # TODO also deal with token length maximum of 512 (and or 145) - one thing exceeds it. Also check strides
                # for silver labels:
                for ent in this_doc.ents:
                    try:
                        if not ent.vector.any(): # TODO this happens 4 times in ontonotes, but technically should not with subwords so it's weird
                            continue
                    except:
                        print(f"Error on entity '{ent}' in document: {this_doc}")
                        continue
                    # validation check for nans
                    if np.isnan(ent.vector).any() or np.isinf(ent.vector.any()):
                        print(f"Skipping entry, found nan or inf in vector for entity '{ent}' "
                              f"in document: {this_doc}")
                        continue
                    embeddings.append(ent.vector)
                    labels.append(ent.label_)
        # save processed split of corpus, with matrix of number_samples x features, list of labels
        corpus[key] = [np.vstack(embeddings), labels]

    # print number of entities found in each section for information
    for key in corpus.keys():
        print("{}: {} entities".format(key, len(corpus[key][0])))

    with open(f"models/corpus_{COMBINATION_FUNCTION}.pkl", "wb") as fout:
        pickle.dump(corpus, fout)

### Errors
# Token indices sequence length is longer than the specified maximum sequence length for this model (720 > 512). Running this sequence through the model will result in indexing errors


def part_4(args, nlp):
    # this involves reading in ontonotes data, getting embeddings for the entities,
    # then training a classifier with the paired embeddings and labels.
    classifier = LogisticRegression()  # TODO make better default params

    # This loads a dict of TESTING, TRAINING, VALIDATION keys and values as a nested list of
    # 0 as embeddings and 1 as labels (co-indexed, equal length)
    with open(args.corpus, "rb") as fin:
        corpus = pickle.load(fin)

    # process data
    label_encoder = preprocessing.LabelEncoder()  # labels need to be ints not strings
    all_labels = list(itertools.chain(*[corpus[split][1] for split in corpus.keys()]))
    label_encoder.fit(all_labels)

    train_data, train_labels_ = corpus["TRAINING"]
    #TODO check this is temporary validation
    print(np.isnan(train_data).any(), np.isinf(train_data).any())
    nan_loc = np.argwhere(np.isnan(train_data))
    remove_rows = sorted(list(set([row[0] for row in nan_loc])), reverse=True) # so remove last first
    for row in remove_rows:
        del train_labels_[row]
        train_data = np.delete(train_data, row, 0)  # index to delete, and axis

    train_labels = label_encoder.transform(train_labels_)  # inverse_transform restores to strings

    print("Training classifier with params:")
    print(classifier.get_params())

    classifier.fit(train_data, train_labels)

    print("Saving classifier to {}".format(args.classifier_path))
    with open(args.classifier_path, "wb") as fout:
        pickle.dump(classifier, fout)

    # visualise features
    # self.plot_coefficients()
    # print out test accuracy if set to test
    if args.test:
        test_data, test_labels_ = corpus["TESTING"]
        predictions = classifier.predict(test_data)

        # TODO check if this works with NER confusion matrix and if it does make a function and use twice
        predictions_ = label_encoder.inverse_transform(predictions)  # transform to strings for printing
        accuracy = np.mean(predictions_ == test_labels_)
        # matrix_labels = (ONTONOTES_LABELS
        #     [label.name for label in Label] + [] if not conf_thresh else [label.name for label in
        #                                                                   Label] + ["below thresh"]
        # )
        print("Classifier Accuracy: {}".format(accuracy))
        print("-" * 89)
        print("Classification Report:")
        print(metrics.classification_report(test_labels_, predictions_,
                                            target_names=label_encoder.classes_))
        print("Confusion Matrix:")
        print(metrics.confusion_matrix(test_labels_, predictions_, labels=[label_encoder.classes_]))


def main(args, nlp):
    dict2func = {
        "1": part_1,
        "2": part_2,
        "3": part_3,
        "4": part_4,
    }

    dict2func[args.part](args, nlp)


if __name__ == "__main__":
    args = setup_argparse()

    # validation checks
    # that model is downloaded
    spacy_model_name = 'en_core_web_trf'
    if not spacy.util.is_package(spacy_model_name):
        spacy.cli.download(spacy_model_name)
    # that relevant directories exist
    for d in ["models", "data"]:
        if not os.path.exists(d):
            os.makedirs(d)

    # load spacy model
    nlp = spacy.load('en_core_web_trf')

    main(args, nlp)

import argparse
import itertools
import re
import json

import numpy as np
import spacy
from spacy import displacy
from spacy.language import Language
from spacy.pipeline import Pipe
from pathlib import Path
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


### This is for Part 3 ###
class ContextualVectors(Pipe):
    def __init__(self, nlp):
        self._nlp = nlp
        self.combination_function = "average"

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
def create_contextual_hook(nlp, name): # TODO how do I pass through the type of hook to make it customisable?
   return ContextualVectors(nlp)


### Globals until flags are sorted out ###
ONTONOTES_LABELS = [
    'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL',
    'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

ontonotes_json = "data/ontonotes5_reduced.json"

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe("trf_vector_hook", last=True)

# TODO have a longer span of text for the second part of this question
text = "On March 8, 2021, a group of hackers including Kottmann and calling themselves " \
       "'APT - 69420 Arson Cats' gained 'super admin' rights in the network of Verkada, a " \
       "cloud-based security camera company, using credentials they found on the public " \
       "internet. They had access to the network for 36 hours. The group collected about 5 " \
       "gigabytes of data, including live security camera footage and recordings from more" \
       " than 150,000 cameras in places like a Tesla factory, a jail in Alabama, a Halifax " \
       "Health hospital, and residential homes. The group also accessed a list of Verkada " \
       "customers and the company's private financial information, and gained superuser " \
       "access to the corporate networks of Cloudflare and Okta through their Verkada cameras."
doc = nlp(text)


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['1','2','3','4'], required=True)
    p.add_argument('--ents', choices=ONTONOTES_LABELS, nargs='+')
    p.add_argument('--viz_output', default='entity_viz_example.html',
                   help='Name of output file for the visualisation')
    p.add_argument('--embed_func', choices=['first', 'last', 'sum', 'max', 'avg'],
                   help='the type of function to use to combine multiple embeddings into one entity'
                        'representation')
    return p.parse_args()


def part_1(args):
    ent_list, output_file = args.ents, args.viz_output
    options = {"ents": ent_list} if ent_list else {"ents": ONTONOTES_LABELS}

    html = displacy.render(doc, style="ent", options=options)

    output_path = Path(output_file)
    output_path.open("w").write(html)


def part_2(args):
    # These are special characters used by the tokenizer, ignore them
    special_chars = re.compile("Ġ|<pad>|<s>|</s>")

    print("List of Entities:")
    print(doc.ents)

    print("\nStandard Tokenisation:")
    print(" ".join([tok.text for tok in doc]))

    print("\nSubword Tokenisation:")
    subword_string = " ".join([tok for tok in itertools.chain(*doc._.trf_data.wordpieces.strings)])
    cleaned_subword_string = special_chars.sub("",subword_string).strip()

    print(cleaned_subword_string)

def part_3(args):
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
                #print(doc)
                # TODO now have to get the entity that is the offset in order to make it gold label? ONTONOTES json is 'entities': {'DATE': [[177, 186]]}
                # TODO use start_char and end_char on both?
                # TODO also deal with token length maximum of 512 (and or 145)
                # for silver labels:
                for ent in this_doc.ents:
                    try:
                        if not ent.vector.any(): #TODO this should never happen, sort out error
                            continue
                    except:
                        print(f"Error on entity '{ent}' in document: {this_doc}")
                        continue
                    embeddings.append(ent.vector)
                    labels.append(ent.label_)
        corpus[key] = [embeddings, labels]

    # TODO this is just validation
    for key in corpus.keys():
        print("{}: {} entities".format(key, len(corpus[key][0])))

    with open("corpus.json", "w") as fout:
        json.dump(corpus, fout)

### Errors
# Token indices sequence length is longer than the specified maximum sequence length for this model (720 > 512). Running this sequence through the model will result in indexing errors




def part_4(args):
    # TODO this involves reading in ontonotes data, getting embeddings for the entities, then training a classifier with the paired embeddings and labels.
    # TODO they should then try with different combination functions and see what they think
    # take extracted representations and labels, and train a classifier
    classifier = LogisticRegression() # TODO make better default params

    # TODO READ IN TRAINING DATA
    # make training data
    text_data, text_labels = format_data(train_items)
    # make pipeline

    logging.info("Training classifier with params:")
    logging.info(classifier.get_params())
    classifier.fit(text_data, text_labels)
    logging.info("Saving classifier to {}".format(self.path))
    save_pkl(self, self.path)
    # visualise features
    self.plot_coefficients()
    # print out test accuracy if exists
    if test_items:
        self.classify_documents(test_items, has_labels=True)


def main(args):
    dict2func = {
        "1": part_1,
        "2": part_2,
        "3": part_3,
        "4": part_4,
    }
    
    dict2func[args.part](args)


if __name__ == "__main__":
    args = setup_argparse()

    main(args)

"""
Preprocess the Quora dataset and word embeddings to be used by the ESIM model.
"""
import os
import pickle
import argparse
import fnmatch
import json

from mfae.data import Preprocessor


def preprocess_quora_data(inputdir,
                         embeddings_file,
                         targetdir,
                         lowercase=False,
                         ignore_punctuation=False,
                         num_words=None,
                         stopwords=[],
                         labeldict={},
                         bos=None,
                         eos=None):
    """
    Preprocess the data from the SNLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        stopwords: A list of words that must be ignored when preprocessing
            the data. Defaults to an empty list.
        bos: A string indicating the symbol to use for beginning of sentence
            tokens. If set to None, bos tokens aren't used. Defaults to None.
        eos: A string indicating the symbol to use for end of sentence tokens.
            If set to None, eos tokens aren't used. Defaults to None.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = "train.tsv"
    dev_file = "dev.tsv"
    test_file = "test.tsv"
    # for file in os.listdir(inputdir):
    #     if fnmatch.fnmatch(file, "train.tsv"):
    #         train_file = file
    #     elif fnmatch.fnmatch(file, "dev.tsv"):
    #         dev_file = file
    #     elif fnmatch.fnmatch(file, "test.tsv"):
    #         test_file = file

    # -------------------- Train data preprocessing -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words,
                                stopwords=stopwords,
                                labeldict=labeldict,
                                bos=bos,
                                eos=eos)

    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    # print(os.path.join(inputdir, train_file))
    data = preprocessor.read_data_quora_bert(os.path.join(inputdir, train_file)) #read_data_quora
    with open(os.path.join(targetdir, "bert_train_data.pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data_quora_bert(os.path.join(inputdir, dev_file))
    with open(os.path.join(targetdir, "bert_dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20*"=", " Preprocessing test set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data_quora_bert(os.path.join(inputdir, test_file))
    with open(os.path.join(targetdir, "bert_test_data.pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)


if __name__ == "__main__":
    default_config = "../../config/preprocessing/quora_preprocessing.json"

    parser = argparse.ArgumentParser(description="Preprocess the Quora dataset")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing SNLI"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    preprocess_quora_data(
        os.path.normpath(os.path.join(script_dir, config["data_dir"])),
        os.path.normpath(os.path.join(script_dir, config["embeddings_file"])),
        os.path.normpath(os.path.join(script_dir, config["target_dir"])),
        lowercase=config["lowercase"],
        ignore_punctuation=config["ignore_punctuation"],
        num_words=config["num_words"],
        stopwords=config["stopwords"],
        labeldict=config["labeldict"],
        bos=config["bos"],
        eos=config["eos"]
    )

import pandas as pd
import numpy as np
# import bert
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tqdm import tqdm
import re
import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig


MAX_SEQ_LEN = 300  # max sequence length


def get_masks(tokens):
    """Masks: 1 for real tokens and 0 for paddings"""
    return [1] * len(tokens) + [0] * (MAX_SEQ_LEN - len(tokens))


def get_segments(tokens):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (MAX_SEQ_LEN - len(tokens))


def get_ids(tokens, ids):
    """Token ids from Tokenizer vocab"""
    token_ids = ids
    input_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
    return input_ids


def create_single_input(sentence, tokenizer, max_len):
    """Create an input from a sentence"""
    # stokens = tokenizer.tokenize(sentence)
    # stokens = stokens[:max_len]
    # stokens = ["[CLS]"] + stokens + ["[SEP]"]

    encoded = tokenizer.encode(sentence)

    ids = get_ids(encoded.tokens, encoded.ids)
    masks = get_masks(encoded.tokens)
    segments = get_segments(encoded.tokens)

    return ids, masks, segments


def convert_sentences_to_features(sentences, tokenizer):
    """Convert sentences to features: input_ids, input_masks and input_segments"""
    input_ids, input_masks, input_segments = [], [], []

    for sentence in tqdm(sentences, position=0, leave=True):
        ids, masks, segments = create_single_input(sentence, tokenizer, MAX_SEQ_LEN - 2)
        assert len(ids) == MAX_SEQ_LEN
        assert len(masks) == MAX_SEQ_LEN
        assert len(segments) == MAX_SEQ_LEN
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]


# def create_tonkenizer(bert_layer):
#     """Instantiate Tokenizer with vocab"""
#     vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
#     do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
#     tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
#     return tokenizer


def nlp_model(callable_object):
    # Load the pre-trained BERT base model
    bert_layer = hub.KerasLayer(handle=callable_object, trainable=True)

    # BERT layer three inputs: ids, masks and segments
    input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")
    input_masks = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_masks")
    input_segments = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")

    inputs = [input_ids, input_masks, input_segments]  # BERT inputs
    pooled_output, sequence_output = bert_layer(inputs)  # BERT outputs

    # Add a hidden layer
    x = Dense(units=768, activation="relu")(pooled_output)
    x = Dropout(0.1)(x)

    # Add output layer
    outputs = Dense(2, activation="softmax")(x)

    # Construct a new model
    model = Model(inputs=inputs, outputs=outputs)
    return model


# model = nlp_model("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
model = nlp_model("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2")
model.summary()

# https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
movie_reviews = pd.read_csv("bert/IMDB Dataset.csv")
movie_reviews.head(5)
movie_reviews = movie_reviews.sample(frac=1)

print(movie_reviews.isnull().values.any())

print(movie_reviews.shape)


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub("[^a-zA-Z]", " ", sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)

    # Removing multiple spaces
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence


TAG_RE = re.compile(r"<[^>]+>")


def remove_tags(text):
    return TAG_RE.sub("", text)


reviews = []
sentences = list(movie_reviews["review"])
for sen in sentences:
    reviews.append(preprocess_text(sen))

print(movie_reviews.columns.values)

print(movie_reviews.sentiment.unique())

y = movie_reviews["sentiment"]

y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))

slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
tokenizer.enable_truncation(MAX_SEQ_LEN - 2)

train_count = 40000 # 40000
test_count = 2000 #

# X_train = convert_sentences_to_features(reviews[:40000], tokenizer)
# X_test = convert_sentences_to_features(reviews[40000:], tokenizer)

X_train = convert_sentences_to_features(reviews[:train_count], tokenizer)
X_test = convert_sentences_to_features(reviews[train_count:train_count+test_count], tokenizer)

one_hot_encoded = to_categorical(y)
# one_hot_encoded = tf.one_hot(y, 1)

# y_train = one_hot_encoded[:40000]
# y_test = one_hot_encoded[40000:]

y_train = one_hot_encoded[:train_count]
y_test = one_hot_encoded[train_count:train_count + test_count]

BATCH_SIZE = 8
EPOCHS = 1

opt = Adam(learning_rate=2e-5)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# Fit the data to the model
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Save the trained model
# model.save('nlp_model.h5')

pred_test = np.argmax(model.predict(X_test), axis=1)
print(pred_test[:10])
# print(reviews[40000:40010])

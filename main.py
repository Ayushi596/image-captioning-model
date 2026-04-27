import os
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
import tensorflow as tf

# ── PATHS (update based on environment) ──
# For Colab:
IMAGES_DIR    = '/content/Images'
CAPTIONS_FILE = '/content/drive/MyDrive/flickr8k/captions.txt'

# ── LOAD CAPTIONS ──
def load_captions(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    captions = {}
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        image_name = parts[0].split('#')[0].strip()
        caption = parts[1].strip().lower()
        caption = re.sub(r"[^a-z\s]", "", caption)
        caption = "startseq " + caption + " endseq"
        if image_name not in captions:
            captions[image_name] = []
        captions[image_name].append(caption)
    return captions

# ── EXTRACT FEATURES ──
def extract_features(images_dir):
    base_model = VGG16()
    model = Model(inputs=base_model.inputs,
                  outputs=base_model.layers[-2].output)
    print("VGG16 loaded ✅")
    features = {}
    for img_name in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_name)
        img = Image.open(img_path).resize((224, 224))
        img = np.array(img)
        if img.shape != (224, 224, 3):
            continue
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img, verbose=0)
        features[img_name] = feature
    return features

# ── BUILD VOCABULARY ──
def build_vocab(captions):
    all_captions = []
    for caps in captions.values():
        all_captions.extend(caps)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in all_captions)
    return tokenizer, vocab_size, max_length

# ── DATA GENERATOR ──
def data_generator(captions, features, tokenizer, max_length, vocab_size):
    for img_name, caps in captions.items():
        if img_name not in features:
            continue
        feat = features[img_name][0]
        for cap in caps:
            seq = tokenizer.texts_to_sequences([cap])[0]
            for t in range(1, len(seq)):
                in_seq   = pad_sequences([seq[:t]], maxlen=max_length)[0]
                out_word = to_categorical([seq[t]], num_classes=vocab_size)[0]
                yield (feat, in_seq), out_word

# ── BUILD MODEL ──
def build_model(vocab_size, max_length):
    img_input  = Input(shape=(4096,))
    img_dense  = Dense(256, activation='relu')(img_input)
    img_drop   = Dropout(0.4)(img_dense)

    text_input = Input(shape=(max_length,))
    text_embed = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text_drop  = Dropout(0.4)(text_embed)
    text_lstm  = LSTM(256, use_cudnn=False)(text_drop)

    merged = Add()([img_drop, text_lstm])
    dense1 = Dense(256, activation='relu')(merged)
    output = Dense(vocab_size, activation='softmax')(dense1)

    model = Model(inputs=[img_input, text_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# ── MAIN TRAINING FLOW ──
if __name__ == '__main__':
    # Load data
    captions = load_captions(CAPTIONS_FILE)
    print(f"Captions loaded: {len(captions)} images")

    # Extract or load features
    FEATURES_PATH = '/content/drive/MyDrive/flickr8k/features.pkl'
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'rb') as f:
            features = pickle.load(f)
        print(f"Features loaded: {len(features)}")
    else:
        features = extract_features(IMAGES_DIR)
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(features, f)
        print("Features saved ✅")

    # Build vocabulary
    tokenizer, vocab_size, max_length = build_vocab(captions)
    print(f"Vocab size: {vocab_size}, Max length: {max_length}")

    # Save tokenizer
    with open('/content/drive/MyDrive/flickr8k/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Build tf.data dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(captions, features, tokenizer,
                               max_length, vocab_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(4096,),       dtype=tf.float32),
                tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
            ),
            tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
        )
    ).batch(64).repeat().prefetch(tf.data.AUTOTUNE)

    # Train
    model = build_model(vocab_size, max_length)
    model.summary()

    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.fit(dataset, epochs=1, steps_per_epoch=7450, verbose=1)
        model.save(f'/content/drive/MyDrive/flickr8k/model_epoch_{epoch+1}.h5')
        print(f"Epoch {epoch+1} saved ✅")
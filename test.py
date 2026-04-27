import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

# ── PATHS ──
MODEL_PATH    = '/content/drive/MyDrive/flickr8k/model_epoch_20.h5'
FEATURES_PATH = '/content/drive/MyDrive/flickr8k/features.pkl'
TOKENIZER_PATH= '/content/drive/MyDrive/flickr8k/tokenizer.pkl'
IMAGES_DIR    = '/content/Images'

# ── LOAD SAVED FILES ──
model = load_model(MODEL_PATH)
with open(FEATURES_PATH, 'rb') as f:
    features = pickle.load(f)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# ── CAPTION GENERATOR ──
def generate_caption(model, tokenizer, photo_feature,
                     max_length, temperature=0.5):
    in_text = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([photo_feature, seq], verbose=0)[0]
        yhat = np.log(yhat + 1e-10) / temperature
        yhat = np.exp(yhat) / np.sum(np.exp(yhat))
        yhat_index = np.random.choice(len(yhat), p=yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()

# ── TEST ON RANDOM IMAGES ──
def test_random_images(n=5):
    import random
    keys = list(features.keys())
    for _ in range(n):
        img_name    = random.choice(keys)
        feature     = features[img_name]
        caption     = generate_caption(model, tokenizer,
                                       feature, max_length=37)
        img = Image.open(f'{IMAGES_DIR}/{img_name}')
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(caption, fontsize=13)
        plt.show()
        print(f"Caption: {caption}\n")

# ── BLEU SCORE ──
def evaluate_bleu(captions, n=500):
    actual, predicted = [], []
    test_images = list(captions.keys())[:n]
    for img_name in tqdm(test_images):
        if img_name not in features:
            continue
        real_caps = []
        for cap in captions[img_name]:
            clean = cap.replace('startseq','').replace('endseq','').strip()
            real_caps.append(clean.split())
        actual.append(real_caps)
        generated = generate_caption(model, tokenizer,
                                     features[img_name], max_length=37)
        predicted.append(generated.split())

    print(f"BLEU-1: {corpus_bleu(actual, predicted, weights=(1,0,0,0)):.4f}")
    print(f"BLEU-2: {corpus_bleu(actual, predicted, weights=(0.5,0.5,0,0)):.4f}")
    print(f"BLEU-3: {corpus_bleu(actual, predicted, weights=(0.33,0.33,0.33,0)):.4f}")
    print(f"BLEU-4: {corpus_bleu(actual, predicted, weights=(0.25,0.25,0.25,0.25)):.4f}")

if __name__ == '__main__':
    test_random_images(5)
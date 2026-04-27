# 🖼️ Image Captioning Model — CNN + LSTM

Automatically generates natural language captions for images using deep learning. Built with VGG16 (CNN) for image feature extraction and LSTM for sequence generation, trained on the Flickr8k dataset.

---

## 🧠 Model Architecture

```
Image → VGG16 (pretrained) → 4096-dim feature vector → Dense(256)
                                                              ↓
                                                           Add() → Dense(256) → Softmax → Word
                                                              ↑
Caption → Embedding(256) → Dropout → LSTM(256)
```

- **CNN:** VGG16 pretrained on ImageNet (transfer learning)
- **Text:** Embedding layer + LSTM decoder
- **Merge:** Add layer combines image and text features
- **Output:** Softmax over vocabulary (~8781 words)
- **Parameters:** 6,144,589 trainable params

---

## 📊 Results

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.3911 |
| BLEU-2 | 0.1917 |
| BLEU-3 | 0.0909 |
| BLEU-4 | 0.0450 |
| Training Loss (Epoch 1) | 3.9707 |
| Training Loss (Epoch 20) | 2.4949 |

Trained for **20 epochs** on T4 GPU (Google Colab). Loss reduced by **37%** over training.

---

## 📁 Project Structure

```
image-captioning-model/
├── main.py                       ← Training pipeline
├── test.py                       ← Caption generation + BLEU evaluation
├── Image_Captioning_Model.ipynb  ← Full Colab notebook
├── requirements.txt              ← Dependencies
└── README.md
```

---

## 🗂️ Dataset

**Flickr8k** — 8,092 images with 5 human-written captions each (40,460 captions total)

Download from Kaggle:
```bash
kaggle datasets download -d adityajn105/flickr8k
```

Or from the original source:
- Images: `Flickr8k_Dataset.zip`
- Captions: `Flickr8k_text.zip` → use `Flickr8k.token`

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Ayushi596/image-captioning-model.git
cd image-captioning-model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Organize dataset
```
flickr8k/
├── Images/        ← all .jpg files
└── captions.txt   ← renamed from Flickr8k.token
```

---

## 🚀 Running on Google Colab (Recommended)

1. Open `Image_Captioning_Model.ipynb` in Google Colab
2. Set runtime to **T4 GPU** → `Runtime → Change runtime type → T4 GPU`
3. Upload dataset to Google Drive at `MyDrive/flickr8k/`
4. Run all cells in order

---

## 🏋️ Training

```bash
python main.py
```

Handles full pipeline:
1. Load and clean captions
2. Extract VGG16 features (saved to `features.pkl`)
3. Build vocabulary and tokenizer
4. Train CNN+LSTM model for 20 epochs
5. Save model checkpoint after each epoch

---

## 🔍 Generate Captions

```bash
python test.py
```

---

## 🛠️ Tech Stack

| Tool | Use |
|------|-----|
| Python 3.12 | Core language |
| TensorFlow / Keras | Model building & training |
| VGG16 | Image feature extraction (transfer learning) |
| LSTM | Sequence/caption generation |
| NumPy | Numerical operations |
| Pillow | Image loading & preprocessing |
| NLTK | BLEU score evaluation |
| Google Colab | GPU training environment |
| Google Drive | Dataset & model storage |

---

## 🔮 Future Improvements

- [ ] Train for 40+ epochs to improve BLEU score
- [ ] Add sample output images with generated captions
- [ ] Implement beam search for better caption quality
- [ ] Replace VGG16 with ResNet50 or EfficientNet
- [ ] Add attention mechanism
- [ ] Build a simple web demo with Streamlit

---

## 👩‍💻 Author

**Ayushi** — [GitHub](https://github.com/Ayushi596)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
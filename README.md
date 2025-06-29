# 🧠 Image Captioning AI — Describe Images with Words

This project is an AI-powered system that generates human-like descriptive captions for images by combining **computer vision** and **natural language processing**. It analyzes images and produces meaningful sentences, making it ideal for accessibility, social media automation, and image search optimization.

## 📸 What Is Image Captioning?

Image captioning bridges two powerful AI domains:
- **Computer Vision**: Extracts visual features from images.
- **Natural Language Processing (NLP)**: Generates coherent, contextually relevant text.

### Use Cases
- **Accessibility**: Helps visually impaired users understand image content.
- **Social Media**: Automatically generates captions for photos.
- **Search Optimization**: Enhances image indexing for search engines and apps.

## 🧰 Project Structure

| File/Folder             | Purpose                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| `Image_Captioning.ipynb`| Jupyter Notebook for training, testing, and generating captions          |
| `best_caption_model.h5` | Pre-trained model file (saved after training)                           |
| `tokenizer.pkl`         | Tokenizes text for model input/output                                   |
| `flickr8k/`             | Dataset folder containing 8,000 images and their captions                |
| `utils/`                | Helper scripts for image feature extraction and preprocessing            |

## 🛠️ How It Works

1. **Image Feature Extraction**:
   - Utilizes **ResNet50**, a pre-trained convolutional neural network, to extract key visual features from images.
   - Converts images into numerical representations for the model.

2. **Caption Generation**:
   - Employs a **Long Short-Term Memory (LSTM)** network to generate captions word-by-word based on extracted image features.
   - Combines visual and textual data to produce coherent sentences.

3. **Training & Evaluation**:
   - Trained on the **Flickr8k dataset**, which includes 8,000 images, each paired with 5 human-written captions.
   - Evaluates caption quality using **BLEU scores** (a metric for comparing generated text to human references).

## 🚀 Getting Started (Local Setup)

### Prerequisites
- **Python 3.8+**
- Required libraries: `tensorflow`, `keras`, `numpy`, `nltk`, `pillow`
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-captioning-ai.git
   cd image-captioning-ai
   ```

2. Install **Git LFS** for large files (e.g., `best_caption_model.h5`):
   ```bash
   git lfs install
   git lfs pull
   ```

3. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook Image_Captioning.ipynb
   ```

4. Follow the notebook instructions to:
   - Download the Flickr8k dataset.
   - Preprocess images and captions.
   - Train the model or load the pre-trained model.
   - Generate captions for new images.

## 🧠 Technologies Used

- **Python**: Core programming language.
- **TensorFlow / Keras**: Deep learning frameworks for model building.
- **ResNet50**: Pre-trained model for image feature extraction.
- **LSTM**: Recurrent neural network for sequence (text) generation.
- **NLTK**: Evaluates caption quality using BLEU scores.
- **Git LFS**: Manages large model files.

## 📁 Managing Large Files

The `best_caption_model.h5` file exceeds 25MB and is tracked using **Git LFS**. To ensure proper download:
```bash
git lfs install
git lfs pull
```
**Note**: Without Git LFS, the model file will not download correctly.

## 💡 Sample Output

**Image**: `3525417522_7beb617f8b.jpg`  
**Generated Caption**:
> *"A dog is running through the grass."*

For more examples, run the notebook and test with your own images!

## 🧩 Future Enhancements

- **Web Interface**: Integrate a user-friendly interface using **Gradio** or **Streamlit** for real-time captioning.
- **Multilingual Support**: Extend caption generation to support multiple languages.
- **Voice Output**: Add text-to-speech functionality for audio descriptions.
- **Improved Models**: Experiment with advanced architectures like Vision Transformers or GPT-based models.
- **Real-Time Processing**: Optimize for faster caption generation on low-resource devices.

## 📚 Resources & Acknowledgments

- **Dataset**: [Flickr8k Dataset](https://github.com/jbrownlee/Datasets)
- **Frameworks**: TensorFlow, Keras
- **Community**: Thanks to the open-source AI community for inspiration and tools

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

*Built with 💻 and ☕ by [Laasvitha]*

# AI Commerce Assistant

An AI-powered shopping assistant that supports product recommendation and image-based search.

## Project Overview

This project implements a conversational AI assistant for an e-commerce setting, inspired by Amazon Rufus. It can answer user questions, recommend products based on text or image inputs, and only recommends items from a predefined catalog.

## Features

-  General conversation with AI agent
-  Text-based product recommendation
-  Image-based product search using GPT-4 vision
-  Only recommends from a predefined product catalog

## Demo

Streamlit Cloud Deployment: [https://d3bgufv6gothniutjlksud.streamlit.app](https://d3bgufv6gothniutjlksud.streamlit.app/))  

## Getting Started

### Run locally

```bash
git clone https://github.com/amenotensa/ai-commerce
cd ai-commerce-assistant
pip install -r requirements.txt
streamlit run app.py
```

### Run on Streamlit Cloud

Just upload the files to your GitHub repo and connect to [Streamlit Cloud](https://streamlit.io/cloud). It will auto-run using `app.py`.

## Tech Stack & Design

- **Frontend**: Streamlit for fast UI
- **LLM**: OpenAI GPT-4o for chat and image understanding
- **Embedding**: `text-embedding-3-small` for product search
- **Vector Search**: NumPy cosine similarity over pre-generated embeddings
- **Deployment**: Streamlit Cloud

## Agent API Documentation

The application internally uses the following OpenAI APIs:

- **Chat API**: `openai.chat.completions.create()`  
  - Used for conversation and image-to-text
  - Model: `gpt-4o-mini`
- **Embedding API**: `openai.embeddings.create()`  
  - Used to embed user queries and catalog items
  - Model: `text-embedding-3-small`

Example usage:
```python
openai.embeddings.create(
    model="text-embedding-3-small",
    input="Recommend a waterproof backpack"
)
```

## Project Structure

```bash
.
├── app.py               # Main application file
├── catalog.json         # Product catalog (name, desc, image path)
├── catalog_vectors.npy  # Precomputed product vectors
├── embed_catalog.py     # Script to generate catalog vectors
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 📄 License

MIT License  
Thanks to OpenAI for GPT-4o APIs.

---

## API Documentation

# Agent API Documentation

This document describes the core AI agent interfaces used in the AI Commerce Assistant application.

## `detect_category(text: str, thresh=0.25) -> Optional[str]`
Detect the most likely product category from a user's query using OpenAI Embeddings.

- **Input:**
  - `text` (str): Natural language query (e.g., `"Recommend a t-shirt for summer"`)
  - `thresh` (float): Similarity threshold for confirming category match

- **Returns:**
  - A category string (e.g., `"t-shirt"`) or `None` if not confidently matched.

---

## `knn(vec: np.ndarray, k=10) -> tuple[list[int], list[float]]`
Find top-k similar items in the catalog using cosine similarity on embeddings.

- **Input:**
  - `vec`: Embedding vector of the query
  - `k`: Number of top results to return

- **Returns:**
  - A tuple of `(indices, similarities)` where `indices` point to catalog items.

---

## `product_matches(cat: str, product: dict) -> bool`
Check if a product matches the given category.

- **Input:**
  - `cat` (str): Category keyword
  - `product` (dict): A catalog product entry

- **Returns:**
  - True if the product matches the category; otherwise False.

---

## `safe_image(path_or_url: str, caption: str = "", width: int = 180, keyword_fallback: str = None)`
Safely load and display a product image in Streamlit.

- **Input:**
  - `path_or_url` (str): Local image path or URL
  - `caption` (str): Optional caption for display
  - `width` (int): Width of the displayed image
  - `keyword_fallback` (str): Fallback keyword for Unsplash image if loading fails

- **Output:**
  - Displays an image in the Streamlit app.

---

## Vision Integration (Image-Based Search)
A call to OpenAI's vision model (GPT-4o with image input) is used to describe uploaded images.

- **Steps:**
  1. Upload an image file in the UI.
  2. The image is encoded and sent to OpenAI Vision API.
  3. Description is parsed to obtain semantic embedding.
  4. Embedding is passed to `knn()` for product matching.

---

## Session-Based Chat (`st.session_state.chat`)
Maintains conversation history for multi-turn chat.

- Detects user intent (e.g., `"Recommend…"`) to trigger recommendation logic.
- If no intent to recommend, passes messages to OpenAI Chat for general conversation.



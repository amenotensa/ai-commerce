import os, json, base64, requests, numpy as np, streamlit as st, openai
from io import BytesIO
from pathlib import Path
from PIL import Image

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("❌ OPENAI_API_KEY is missing."); st.stop()

# ---------- Product Data ----------
catalog = json.load(open("catalog.json"))
prod_vecs = np.load("catalog_vectors.npy")

def knn(vec: np.ndarray, k=10):
    sims = prod_vecs @ vec
    idx = sims.argsort()[-k:][::-1]
    return idx, sims[idx]

# ---------- Category Vectors ----------
CATEGORIES = {
    "t-shirt":      "Short-sleeve tee shirt for sports or daily wear.",
    "backpack":     "Backpack carried on the back for commuting or travel.",
    "earbuds":      "Wireless in-ear earbuds or earphones.",
    "yoga mat":     "Non-slip yoga or workout mat.",
    "wallet":       "Leather wallet or card holder."
}
@st.cache_resource(show_spinner="🔄 Loading category vectors...")
def load_cat_vecs():
    emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=list(CATEGORIES.values())
    ).data
    return np.array([e.embedding for e in emb]), list(CATEGORIES.keys())
cat_vecs, cat_keys = load_cat_vecs()

def detect_category(text: str, thresh=0.25):
    q_vec = openai.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding
    sims = cat_vecs @ np.array(q_vec)
    best = sims.argmax()
    return cat_keys[best] if sims[best] > thresh else None

def product_matches(cat, product):
    if not cat: return True
    return cat in f"{product['name']} {product['desc']}".lower()

# ---------- Safe Image Loading ----------
def safe_image(path_or_url, caption="", width=180, keyword_fallback=None):
    try:
        if Path(path_or_url).is_file():
            img = Image.open(path_or_url)
        else:
            data = requests.get(path_or_url, timeout=8)
            data.raise_for_status()
            img = Image.open(BytesIO(data.content))
    except Exception:
        if keyword_fallback:
            try:
                alt = f"https://source.unsplash.com/400x400?{keyword_fallback}"
                data = requests.get(alt, timeout=8); data.raise_for_status()
                img = Image.open(BytesIO(data.content))
                st.info(f"Image unavailable. Replaced with a random image of“{keyword_fallback}”.")
            except Exception as e2:
                st.warning(f"🖼️ Failed to load image:{e2}"); return
        else:
            return
    st.image(img, caption=caption, width=width)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Commerce Assistant")
st.title("🛍️ AI Commerce Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role":"system",
         "content":"You are an enthusiastic e-commerce assistant. "
                   "Always reply in the same language as the user."}]

tab1, tab2 = st.tabs(["Chat / Recommend", "Image Search"])

# ===== Tab1: Chat + Product Recommendation =====
with tab1:
    for m in st.session_state.chat[1:]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Ask a question or enter 'recommend ...'")
    if user_msg:
        st.chat_message("user").markdown(user_msg)
        st.session_state.chat.append({"role":"user","content":user_msg})

        if user_msg.lower().startswith(("recommend")):
            cat = detect_category(user_msg)
            q_vec = openai.embeddings.create(
                model="text-embedding-3-small", input=user_msg
            ).data[0].embedding
            idxs, sims = knn(np.array(q_vec), k=15)

            results = []
            for i,s in zip(idxs,sims):
                if s < 0.22: continue
                if product_matches(cat,catalog[i]):
                    results.append(catalog[i])
                if len(results)==5: break

            with st.chat_message("assistant"):
                if results:
                    st.markdown(f"Here are some **{cat or 'related'}** products we picked for you:")
                    for p in results:
                        col1, col2 = st.columns([1,3], gap="small")
                        with col1:
                            safe_image(
                                p["img"], width=120,
                                keyword_fallback=p["name"].split()[0]
                            )
                        with col2:
                            st.markdown(f"**{p['name']}**  \n￥{p['price']}")
                else:
                    st.markdown("Sorry, no matching products found in the catalog.")

                st.session_state.chat.append({"role":"assistant","content":"(image list rendered)"})
        else:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.chat,
                temperature=0.7
            )
            ans = resp.choices[0].message.content
            st.chat_message("assistant").markdown(ans)
            st.session_state.chat.append({"role":"assistant","content":ans})

# ===== Tab2：Image Search =====
with tab2:
    img_file = st.file_uploader("Upload product image", type=["jpg", "png"])
    if img_file:
        img_bytes = img_file.read()
        vision = openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role":"system","content":"Describe the product in the image (40 words)."},
                {"role":"user","content":[
                    {"type":"text","text":"What is this product?"},
                    {"type":"image_url","image_url":{
                        "url":"data:image/png;base64,"+base64.b64encode(img_bytes).decode()
                    }}
                ]}
            ]
        ).choices[0].message.content.strip()
        st.write("**Image Description:** ", vision)

        q_vec = openai.embeddings.create(
            model="text-embedding-3-small", input=vision
        ).data[0].embedding
        best = knn(np.array(q_vec), k=1)[0][0]
        prod = catalog[best]

        safe_image(
            prod["img"],
            caption=f"{prod['name']}  ￥{prod['price']}",
            keyword_fallback=prod["name"].split()[0]
        )
import json, numpy as np, openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")

catalog = json.load(open("catalog.json"))
texts = [f"{p['name']} - {p['desc']}" for p in catalog]

resp = openai.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
vecs = np.array([d.embedding for d in resp.data])
np.save("catalog_vectors.npy", vecs)
print("✅ embeddings saved :", vecs.shape)
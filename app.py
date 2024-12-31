import streamlit as st
import requests
import numpy as np
import os

# Load environment variables for OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1"

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# Sample documents (replace with your dataset)
documents = [
    """
    Call us for Appointments
    051-8351111

    Monday - Saturday
    9:00 AM – 9:00 PM

    Root Canals
    Affordable prices
    Routine Check-ups
    Comprehensive dental check-ups
    Teeth Whitening
    High-quality teeth whitening

    Welcome to Smile Center Dental Clinic in Islamabad
    Quality and affordable dentistry
    Smile Center Dental Clinic dentists are committed to your individual attention and strive to provide personalized oral health care tailored to your unique dental needs and goals. We work hard to create the ultimate stress-free and comfortable dental experience. Your health and comfort are our top priorities, and we do what it takes to help every patient understand their symptoms and treatment options in a relaxed office setting.

    It is our mission to educate and provide our patients with the best dental care. Achieving quality dental care shouldn’t be difficult, so at Smile Center, we make it simple to achieve your healthiest, most beautiful smile.

    Quality and affordable dentistry

    Dental Fillings
    Treatment to restore the function, integrity, and morphology of missing tooth structure

    Orthodontics
    We specialize in the diagnosis, prevention, and correction of malpositioned teeth and jaws.

    Tooth Extraction
    We have state-of-the-art equipment in our facility, which makes the procedure pain-free.

    Root Canal Treatment
    Nerve and pulp are removed, and the inside of the tooth is cleaned and sealed by our expert dentists.

    Teeth Whitening
    This is treated by changing the intrinsic color or by removing the formation of extrinsic stains.

    Routine Dental Exam & Check-Up
    Routine check-ups are a regular procedure towards your dental care and healthy teeth.

    Make your dream smile a reality!
    Call us or book your appointment today.

    Why should I go to the dentist regularly?
    Many people do not see a dentist on a regular basis. They only go when they have a problem. This is known as “crisis treatment” versus “preventive treatment.” While these patients may feel they are saving money, it often ends up costing much more in dollars and time. This is because many dental problems do not have symptoms until they reach the advanced stages of the disease process.

    How can I prevent cavities?
    Why does the dentist take X-rays?
    What should I do about bleeding gums?

    Our Patients
    Smiles to be proud of

    Our Dental Team
    Professional and highly trained

    Address
    Office 37 & 38 Ground Floor, Al-Babar Center, Park Road, F-8 Markaz Islamabad.
    Near OGDCL Center & Rayyan’s Restaurant and Opposite OPF School.

    Phone
    051-8351111, 0321-5212690

    Email
    info@smilecenterislamabad.com

    Opening Hours
    Mon-Sat: 09:00 AM – 9:00 PM
    Sun: Closed
    24/7 availability in case of emergency

    Smile Center Islamabad

    Smile Center encourages clients to contact us whenever they have an interest or concern about dentistry procedures.
    """
]

# Function to get embeddings using OpenAI's REST API
def get_openai_embeddings(texts):
    url = f"{OPENAI_API_URL}/embeddings"
    payload = {
        "model": "text-embedding-ada-002",
        "input": texts
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    embeddings = [np.array(data['embedding']) for data in response.json()['data']]
    return embeddings

# Embed documents (cache this for efficiency)
@st.cache_resource
def embed_documents(docs):
    return get_openai_embeddings(docs)

document_embeddings = embed_documents(documents)

# Streamlit UI
st.title("Smile Center AI Chatbot")

# User input
query = st.text_input("Ask a question:", "")

if query:
    # Get query embedding
    query_embedding = get_openai_embeddings([query])[0]

    # Compute cosine similarity
    scores = np.dot(document_embeddings, query_embedding) / (
        np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    best_match_idx = np.argmax(scores)
    relevant_document = documents[best_match_idx]

    # Generate a response using OpenAI's GPT-4 REST API
    url = f"{OPENAI_API_URL}/chat/completions"
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": f"Based on the following document: \"{relevant_document}\", answer the question: \"{query}\""}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    generated_text = response.json()['choices'][0]['message']['content']

    # Display the response
    st.write("### Answer:")
    st.write(generated_text)
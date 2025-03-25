import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import os
from datetime import datetime   
from transformers import pipeline

# 🔹 Initialisation de session_state
if "base" not in st.session_state:
    with open("botbase.json", "r", encoding="utf-8") as f:
        st.session_state.base = json.load(f)

if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if "questions" not in st.session_state:
    st.session_state.questions = []
    st.session_state.reponses = []
    for item in st.session_state.base:
        for variante in item["questions"]:
            st.session_state.questions.append(variante)
            st.session_state.reponses.append(item["answer"])
    st.session_state.embeddings = st.session_state.model.encode(st.session_state.questions, convert_to_tensor=True)

if "historique" not in st.session_state:
    st.session_state.historique = []

if "paraphraser" not in st.session_state:
    st.session_state.paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")

def generer_paraphrases(phrase):
    resultats = st.session_state.paraphraser(f"paraphrase: {phrase} </s>", max_length=50, num_return_sequences=3)
    return [r['generated_text'] for r in resultats]

def ajouter_nouvelle_entree(question, reponse, fichier="botbase.json"):
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            base = json.load(f)
    except FileNotFoundError:
        base = []

    nouvelle_entree = {
        "questions": question,
        "answer": reponse
    }

    base.append(nouvelle_entree)

    with open(fichier, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=4, ensure_ascii=False)

def log_apprentissage(question, reponse, source="Utilisateur", fichier="logs_apprentissage.json"):
    log = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "reponse": reponse,
        "source": source
    }
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []
    logs.append(log)
    with open(fichier, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

#  Interface type ChatGPT
st.title("💬 Mon Chatbot Style GPT")

response_generated = False

#  Formulaire d'entrée utilisateur
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Pose ta question :", key="input")
    submitted = st.form_submit_button("Envoyer")

#  Traitement de la question une fois soumise
if submitted and user_input:
    emb_user = st.session_state.model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(emb_user, st.session_state.embeddings)[0]
    score_max = scores.max().item()
    idx = scores.argmax().item()

    if score_max > 0.7:
        reponse = st.session_state.reponses[idx]
    else:
        reponse = "Je ne connais pas encore cette question. Tu peux m'apprendre la réponse en dessous."

    st.session_state.historique.append({"role": "user", "content": user_input})
    st.session_state.historique.append({"role": "bot", "content": reponse})
    response_generated = True

    
st.markdown("---")
st.subheader("🗂️ Historique de la conversation")

for message in st.session_state.historique:
    if message["role"] == "user":
        st.markdown(f"<div style='background-color:#DCF8C6; padding:8px; border-radius:10px; margin:5px 0'><b>🧍‍ Toi :</b> {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#F1F0F0; padding:8px; border-radius:10px; margin:5px 0'><b>🤖 Bot :</b> {message['content']}</div>", unsafe_allow_html=True)

# Affichage de l'historique à jour (après traitement)
#for message in st.session_state.historique:
  #  if message["role"] == "user":
    #    markdown_placeholder1.markdown(f"**🧜‍♂️ Toi :** {message['content']}")
    #else:
     #   markdown_placeholder2.markdown(f"**🤖 Bot :** {message['content']}")
   # markdown_placeholder3.markdown("---")

# Apprentissage si non connu
if response_generated and score_max <= 0.7:
    with st.expander("Ajouter une réponse pour cette question"):
        new_answer = st.text_input("✍️ Ta réponse :", key="new_answer")
        if new_answer:
            paraphrases = generer_paraphrases(user_input)
            st.session_state.base.append({
                "questions": [user_input] + paraphrases,
                "answer": new_answer
            })
            st.session_state.questions.append(user_input)
            st.session_state.reponses.append(new_answer)
            for p in paraphrases:
                st.session_state.questions.append(p)
                st.session_state.reponses.append(new_answer)
            st.session_state.embeddings = st.session_state.model.encode(st.session_state.questions, convert_to_tensor=True)
            ajouter_nouvelle_entree([user_input] + paraphrases, new_answer)
            log_apprentissage([user_input] + paraphrases, new_answer)
            st.success("Merci ! J'ai appris une nouvelle réponse")

st.divider()
if st.button("🔄 Réinitialiser la session"):
    st.session_state.clear()
    

if st.button("📃 Voir toutes les questions connues"):
    for item in st.session_state.base:
        st.markdown("- " + ", ".join(item["questions"]))

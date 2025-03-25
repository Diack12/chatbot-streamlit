#  Importation des librairies nécessaires
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import requests
from datetime import datetime
from transformers import pipeline

#  Chargement ou initialisation de la base de données
@st.cache_data
def charger_faq():
    try:
        with open("botbase.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

#  Enregistrement dans le fichier JSON
def ajouter_nouvelle_entree(question, reponse):
    base = charger_faq()
    base.append({"questions": question, "answer": reponse})
    with open("botbase.json", "w", encoding="utf-8") as f:
        json.dump(base, f, indent=4, ensure_ascii=False)

#  Logging d'apprentissage
def log_apprentissage(question, reponse, source):
    log = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "reponse": reponse,
        "source": source
    }
    try:
        with open("logs_apprentissage.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []
    logs.append(log)
    with open("logs_apprentissage.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

#  Recherche intelligente web
@st.cache_data(show_spinner=False)
def reponse_wikipedia_intelligente(question):
    search_url = "https://fr.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": question, "format": "json"}
    try:
        r = requests.get(search_url, params=params)
        titre = r.json()["query"]["search"][0]["title"]
        summary_url = f"https://fr.wikipedia.org/api/rest_v1/page/summary/{titre.replace(' ', '_')}"
        r = requests.get(summary_url)
        return r.json().get("extract")
    except:
        return None

@st.cache_data(show_spinner=False)
def recherche_duckduckgo(query):
    try:
        url = "https://api.duckduckgo.com/?q=" + query + "&format=json&no_redirect=1&no_html=1"
        r = requests.get(url)
        j = r.json()
        return j.get("AbstractText") or j.get("Answer")
    except:
        return None

def cherche_partout(question):
    wiki = reponse_wikipedia_intelligente(question)
    if wiki:
        return wiki, "Wikipedia"
    ddg = recherche_duckduckgo(question)
    if ddg:
        return ddg, "DuckDuckGo"
    return None, None

#  Paraphraseur
@st.cache_resource
def get_paraphraser():
    return pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")

#  Préparation du modèle d'embedding
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def preparer_questions_reponses(base):
    q, r = [], []
    for item in base:
        for variante in item["questions"]:
            q.append(variante)
            r.append(item["answer"])
    return q, r

#  Initialisation
st.set_page_config(page_title="Chatbot IA", page_icon="🔎")
st.title("💬 Chatbot Intelligent (local + web)")


if "historique" not in st.session_state:
    st.session_state.historique = []

base = charger_faq()
questions, reponses = preparer_questions_reponses(base)
model = get_model()
embeddings = model.encode(questions, convert_to_tensor=True)
paraphraser = get_paraphraser()

#  Entrée utilisateur
with st.form("formulaire", clear_on_submit=True):
    user_input = st.text_input("Pose ta question :", key="input")
    submitted = st.form_submit_button("Envoyer")

if submitted and user_input:
    emb_user = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(emb_user, embeddings)[0]
    score_max = scores.max().item()
    idx = scores.argmax().item()

    if score_max > 0.7:
        reponse = reponses[idx]
        st.session_state.historique.append((user_input, reponse))
    else:
        reponse, source = cherche_partout(user_input)
        if reponse:
            st.info(f"{reponse}\n(Source : {source})")
            if st.button("Apprendre cette réponse ?"):
                variantes = [user_input] + [r['generated_text'] for r in paraphraser(f"paraphrase: {user_input} </s>", max_length=50, num_return_sequences=2, do_sample=True)]
                ajouter_nouvelle_entree(variantes, reponse)
                log_apprentissage(variantes, reponse, source)
                st.success("Réponse apprise !")
        else:
            st.warning("Aucune réponse trouvée sur le web. Apprends-la moi !")
            new_answer = st.text_input("Ta réponse :")
            if new_answer:
                variantes = [user_input] + [r['generated_text'] for r in paraphraser(f"paraphrase: {user_input} </s>", max_length=50, num_return_sequences=2, do_sample=True)]
                ajouter_nouvelle_entree(variantes, new_answer)
                log_apprentissage(variantes, new_answer, "Utilisateur")
                st.success("Merci, j'ai appris une nouvelle réponse !")
                st.session_state.historique.append((user_input, new_answer))

#  Affichage de l'historique
st.divider()
st.subheader("📖 Historique de la conversation")
for q, r in st.session_state.historique[::-1]:
    st.markdown(f"**🧍️ Toi :** {q}")
    st.markdown(f"**🤖 Bot :** {r}")
    st.markdown("---")

#  Reset session
if st.button("🔄 Réinitialiser"):
    st.session_state.historique = []
    st.session_state.clear()

import json
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
from datetime import datetime


from transformers import pipeline
import requests

def recherche_duckduckgo(query):
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "no_html": 1
    }
    try:
        r = requests.get(url, params=params)
        data = r.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        elif data.get("Answer"):
            return data["Answer"]
        else:
            return None
    except:
        return None


def reponse_wikipedia_intelligente(question):
    search_url = "https://fr.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": question,
        "format": "json"
    }

    try:
        response = requests.get(search_url, params=search_params)
        data = response.json()

        titre = data["query"]["search"][0]["title"]
        summary_url = f"https://fr.wikipedia.org/api/rest_v1/page/summary/{titre.replace(' ', '_')}"
        summary_response = requests.get(summary_url)
        summary_data = summary_response.json()

        return summary_data.get("extract", None)

    except:
        return None


def cherche_partout(question):
    print("🤖 Je cherche une réponse sur le web...")

    wiki_result = reponse_wikipedia_intelligente(question)
    if wiki_result:
        print("📚 Trouvé sur Wikipédia :")
        print(wiki_result)
        return wiki_result, "Wikipedia"

    ddg_result = recherche_duckduckgo(question)
    if ddg_result:
        print("🔎 Trouvé via DuckDuckGo :")
        print(ddg_result)
        return ddg_result, "DuckDuckGo"

    return None, None


paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")

def generer_paraphrases(phrase):
    resultats = paraphraser(
        f"paraphrase: {phrase} </s>",
        max_length=50,
        num_return_sequences=3,
        do_sample=True  # 🔧 ajout obligatoire pour générer plusieurs sorties
    )
    paraphrases = [r['generated_text'] for r in resultats]

    # Filtrage facultatif
    mots_origine = set(phrase.lower().split())
    paraphrases_filtrees = [
        p for p in paraphrases
        if any(mot in p.lower() for mot in mots_origine)
    ]

    return paraphrases_filtrees if paraphrases_filtrees else paraphrases



def ajouter_nouvelle_entree(question, reponse, fichier="botbase.json"):
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            base = json.load(f)
    except FileNotFoundError:
        base = []

    # On ajoute la question comme nouvelle entrée avec une seule variante
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




# Chargement des données
def charger_faq(fichier="botbase.json"):
    with open(fichier, "r", encoding="utf-8") as f:
        return json.load(f)
        

# Embeddings des questions de la base


def preparer_questions_reponses(faq):
    questions = []
    reponses = []
    for item in faq:
        for variante in item["questions"]:
            questions.append(variante)
            reponses.append(item["answer"])  # Répéter la réponse pour chaque variante
    return questions, reponses


# Chargement de la base
faq = charger_faq()
questions, reponses = preparer_questions_reponses(faq)

# Chargement du modèle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embeddings des questions

embeddings = model.encode(questions, convert_to_tensor=True)


# Boucle principale
while True:
    user_input = input("Pose ta question (ou 'quit') : ")
    if user_input.lower() in ['quit', 'exit']:
        break
   

    elif user_input.lower() == "#connait":
     print("🤖 Je connais les questions suivantes :")
     for item in faq:
        print("🗨️ ", ", ".join(item["questions"]))
     continue
    
    emb_user = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(emb_user, embeddings)[0]

    # Trouver l'indice de la question la plus proche
    idx_max = scores.argmax().item()
    score_max = scores[idx_max].item()

    if score_max > 0.7:  # seuil de confiance
        index = scores.argmax().item()
        print("🤖", reponses[index])

        
    else:
      reponse, source = cherche_partout(user_input)

      if reponse:
        validation = input("Souhaites-tu que j'apprenne cette réponse ? (oui/non) : ").strip().lower()
        if validation in ["oui", "yes", "y", "o"]:
            variantes = generer_paraphrases(user_input)
            ajouter_nouvelle_entree([user_input] + variantes, reponse)
            log_apprentissage([user_input] + variantes, reponse, source=source)
            # Rechargement
            faq = charger_faq()
            questions, reponses = preparer_questions_reponses(faq)
            embeddings = model.encode(questions, convert_to_tensor=True)
            print("✅ Réponse apprise automatiquement.")
        else:
            print("D'accord, je ne retiendrai pas cette réponse.")
      else:
        print("❌ Je n’ai trouvé aucune réponse automatique.")
        response_input = input("Peux-tu m’enseigner la bonne réponse ? : ")
        variantes = generer_paraphrases(user_input)
        ajouter_nouvelle_entree([user_input] + variantes, response_input)
        log_apprentissage([user_input] + variantes, response_input)
        faq = charger_faq()
        questions, reponses = preparer_questions_reponses(faq)
        embeddings = model.encode(questions, convert_to_tensor=True)
        print("✅ Réponse ajoutée manuellement.")

            

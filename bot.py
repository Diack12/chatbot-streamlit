import json
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
from datetime import datetime


from transformers import pipeline

paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase-Paws")

def generer_paraphrases(phrase):
    resultats = paraphraser(f"paraphrase: {phrase} </s>", max_length=50, num_return_sequences=5)
    return [r['generated_text'] for r in resultats]



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

        
    else:    # seuil trop bas
        
        response_input=input("Je ne comprends pas la question, peut tu me donner la réponse ? ")
        variantes=generer_paraphrases(user_input)
        
        ajouter_nouvelle_entree([user_input] + variantes,response_input)
        log_apprentissage([user_input] + variantes,response_input)
        # Rechargement de la base mise à jour
        faq = charger_faq()
        questions, reponses = preparer_questions_reponses(faq)

        # Mise à jour des embeddings
        embeddings = model.encode(questions, convert_to_tensor=True)

            

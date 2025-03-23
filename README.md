# 🤖 Chatbot IA avec Streamlit

Un petit assistant intelligent construit avec **Python**, **Streamlit**, et **Transformers**.  
Il est capable de répondre à des questions, d’apprendre de nouvelles réponses, et d’enrichir sa base de connaissances automatiquement.

---

##  Fonctionnalités

-  Reconnaissance de similarité sémantique via `sentence-transformers`
-  Interface utilisateur en Streamlit (style ChatGPT)
-  Historique des échanges affiché en direct
-  Apprentissage automatique si une question n’est pas reconnue
-  Génération automatique de variantes de questions via un modèle de paraphrase

---

##  Lancer le projet localement

### 1. Cloner le dépôt

```bash
git clone https://github.com/Diack12/clean-chatbot.git
cd clean-chatbot

### 2. Installer les dépendances

```bash
pip install -r requirements.txt

### 3. Lancer l'application Streamlit

```bash
streamlit run app_streamlit.py

##  Structure du projet
.
├── app_streamlit.py        # Code principal de l'application
├── bot.py                  # Code du bot sans app/textuel ( sur le terminal)
├── botbase.json            # Base de questions/réponses
├── logs_apprentissage.json # Historique des ajouts utilisateurs
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
## Auteur
Faye Papa Djidiack
Étudiant en ingénierie passionné par l'IA, la programmation et les systèmes embarqués.
##  Déploiement
Ce projet peut être déployé gratuitement sur Streamlit Cloud.
1. Poussez votre code sur GitHub

2. Connectez Streamlit Cloud à votre compte GitHub

3. Choisissez le fichier app_streamlit.py

4. Cliquez sur Deploy 

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Projet ML - LDA", layout="wide")

# --- BARRE LATÉRALE DE NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Khtar ach bghiti t-chouf :", ["1. Présentation PPT (PDF)", "2. Cas Pratique LDA"])

# Smiya dyal l-PDF exact kima 3ndek f GitHub
pdf_filename = "Projet LDA – Discrimination Linéaire_compresse.pdf"

# --- SECTION 1 : AFFICHAGE DU PPT ---
if page == "1. Présentation PPT (PDF)":
    st.title("📂 Présentation du Projet")
    st.write("Voici les diapositives théoriques de mon projet de Machine Learning.")
    
    try:
        with open(pdf_filename, "rb") as f:
            pdf_data = f.read()
        
        # Bouton pour télécharger le PPT direct (Hada khdam 100%)
        st.download_button(
            label="📥 Télécharger la présentation PDF complète",
            data=pdf_data,
            file_name=pdf_filename,
            mime="application/pdf"
        )
        
        st.markdown("---")
        st.info("💡 Vous pouvez aussi lire la présentation ou la télécharger directement via le bouton ci-dessus.")
        
        # Solution de secours visuelle pour le prof :
        st.write("Si le lecteur intégré ne s'affiche pas, vous pouvez directement télécharger le document ci-dessus.")
        
    except FileNotFoundError:
        st.error(f"Le fichier '{pdf_filename}' est introuvable sur GitHub. Vérifiez son nom.")

# --- SECTION 2 : CAS PRATIQUE JUPYTER (LDA) ---
elif page == "2. Cas Pratique LDA":
    st.title("Presentation LDA")
    st.write("Exécution en direct de l'algorithme sur le Dataset Iris :")

    # 1. Dataset & Split
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 2. Modèle LDA avec Shrinkage
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X_train, y_train)

    # 3. PCA pour la visualisation 2D
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)
    lda_2d = LinearDiscriminantAnalysis().fit(X_test_pca, y_test)

    # 4. Création du Dashboard de résultats avec Matplotlib / Seaborn
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # A. Decision Boundary
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax1.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    ax1.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', cmap='viridis')
    ax1.set_title(f'LDA Decision Boundary\n(Test Acc: {lda.score(X_test, y_test)*100:.1f}%)', fontweight='bold')

    # B. Confusion Matrix
    cm = confusion_matrix(y_test, lda.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=iris.target_names, yticklabels=iris.target_names)
    ax2.set_title('Confusion Matrix', fontweight='bold')

    # C. Performance Bar Chart
    train_acc = accuracy_score(y_train, lda.predict(X_train)) * 100
    test_acc = accuracy_score(y_test, lda.predict(X_test)) * 100
    ax3.bar(['Entraînement', 'Test'], [train_acc, test_acc], color=['#3498db', '#e74c3c'], width=0.5)
    ax3.set_ylim(0, 110)
    ax3.set_ylabel('Précision (%)')
    ax3.set_title('Analyse de la Performance', fontweight='bold')
    for i, v in enumerate([train_acc, test_acc]):
        ax3.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()

    # --- AFFICHAGE DU GRAPHIQUE DANS STREAMLIT ---
    st.pyplot(fig)
    st.success(f"✅ Modèle entraîné avec succès ! Précision globale sur le Test : {test_acc:.1f}%")

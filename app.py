import streamlit as st
import joblib

# Carrega modelo e vetorizador
modelo = joblib.load("model/modelo_final.pkl")
vetorizador = joblib.load("model/vetorizador.pkl")

# Configuração da página
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
)

# Título
st.title("🎬 Movie Review Sentiment Analysis")

st.write("""
Esta aplicação utiliza Machine Learning para classificar avaliações
de filmes como positivas ou negativas.
""")

st.divider()

# Entrada de texto
texto = st.text_area(
    "Digite uma avaliação de filme em inglês:",
    height=200
)

# Botão
if st.button("Analisar Sentimento"):

    if texto.strip() == "":
        st.warning("Digite um texto para análise.")
    else:

        st.subheader("Texto informado")
        st.info(texto)

        # Vetorização
        texto_vetorizado = vetorizador.transform([texto])

        # Predição
        resultado = modelo.predict(texto_vetorizado)[0]

        st.subheader("Resultado")

        if resultado == 1:
            st.success(
                "O modelo classificou a avaliação como POSITIVA."
            )
        else:
            st.error(
                "O modelo classificou a avaliação como NEGATIVA."
            )

        # Confiança da previsão
        try:
            probabilidades = modelo.predict_proba(texto_vetorizado)

            confianca = max(probabilidades[0]) * 100

            st.metric(
                label="Confiança da previsão",
                value=f"{confianca:.2f}%"
            )

        except:
            st.info(
                "Este modelo não disponibiliza probabilidades de classificação."
            )
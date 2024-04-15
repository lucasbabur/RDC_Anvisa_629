import streamlit as slt
import langchain_helper as lc

slt.title("Dúvidas sobre a RDC 629/2022")

question = slt.text_input(label="Digite sua dúvida", max_chars=200)

if slt.button("Enviar"):
    response = lc.invoke(question)  
    slt.write(response)  
from groq import Groq
from dotenv import load_dotenv
import os 
import streamlit as st
from create_vectors import embed_text,vector_index
load_dotenv()
Groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
st.title("A cool rag app by student at infosys ")
st.text("Ask me anythin about student at infosys")

user_query = st.text_input("Enter your query here: ")
submit_button = st.button("Submit")

if submit_button and user_query:
    if 'history' not in st.session_state:
        st.session_state.history = []
    vector = embed_text(user_query)
    vector_search_reponse = vector_index.query(vector=vector,top_k=2,include_metadata=True)
    similar_texts = ""
    
    for match in vector_search_reponse['matches']:
        text = match['metadata']['text']
        similar_texts += text + "\n\n"
  
    system_prompt = f"""You are a helpful assistant. Use the following context to answer the user's questions.
     # some of the documents for the context
     {similar_texts}
     # """
    system_context = {
        "role":'system',
        'content':system_prompt
    }
    st.session_state.history.append(system_context)
    user_context = {
        "role":"user",
        "content":user_query
    }
    st.session_state.history.append(user_context)
    response = Groq_client.chat.completions.create(
            model = "llama-3.3-70b-versatile",
            messages = st.session_state.history,
            max_tokens = 1500,
            temperature = 0.3
        )
    llm_answer = response.choices[0].message.content
    llm_context = {
        "role":"assistant",
        "content":llm_answer
    }
    st.session_state.history.append(llm_context)
    st.write(f"'LLm answer ' : {llm_answer}")
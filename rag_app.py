import os 
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone
from google import genai
groq_client = Groq(api_key = os.getenv("GROQ_API_KEY"))
Pinecone_client = Pinecone(api_key= os.getenv("PINECONE_API"))
google_client = genai.Client(api_key = os.getenv("GEMINI_API"))
vector_index = Pinecone_client.Index("student-kb")

def user_query_match(query_text):
    
    response = google_client.models.embed_content(
        model = "gemini-embedding-001",
        contents = query_text,
        config={
            "output_dimensionality" : 768
        }
    )
    query_embedding = response.embeddings[0].values
    return query_embedding

def get_result_after_embeding(query_text):

    query_embedding = user_query_match(query_text)
    result = vector_index.query(
        vector=query_embedding,
        top_k=5,  # number of similar items to retrieve
        include_metadata=True  # return your original text or extra info
    )
    return result

def get_match_text(query_text):
   
    result = get_result_after_embeding(query_text)
    result_1 = []
    for match in result['matches']:
        # print(f"ID: {match['id']} | Score: {match['score']} | Metadata: {match['metadata']['text']}")
        result_1.append(match['metadata']['text'])
    result2 = '/n/n'.join(result_1)
    return result2


history = []

while True:
    
    query_text = input("")
    system_prompt = get_match_text(query_text)
    system_context = {
        'role':'system',
        'content' : system_prompt
    }
    history.append(system_context)
    user_prompt = query_text
    user_context = {
        'role':'user',
        'content': user_prompt
    }
    history.append(user_context)
    response = groq_client.chat.completions.create(
            model = "llama-3.3-70b-versatile",
            messages = history,
            max_tokens = 1500,
            temperature = 0.3
        )
    llm_answer = response.choices[0].message.content
    llm_context = {
        'role':'assistant',
        'content':llm_answer
    }
    history.append(llm_context)
    print("AI :: ", llm_answer)
# system , user and llm 
# system ko 
# we need to create match
# step 1 take the input query from the user.
# convert it to the vector embeding using the gemini
# match the common with the metadata in the pine_code database.


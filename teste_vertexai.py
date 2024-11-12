import os
import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import SystemMessage, HumanMessage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mamae.json'

vertexai.init(project='', location='')

# Inicializa o cliente de LLM
llm_client = ChatVertexAI(
    model="gemini-1.5-flash-001",
    temperature=1,
    max_tokens=2000,
    max_retries=3,
    stop=None,
)

content = "You are a helpful assistant that translates English to French. Translate the user sentence."
user_question = "I love programming."
messages = [SystemMessage(content=content), HumanMessage(content=user_question)]

answer = llm_client.invoke(messages)
print(answer.content)







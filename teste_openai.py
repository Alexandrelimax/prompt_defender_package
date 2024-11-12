# !pip install langchain-openai
import os
from langchain_openai import ChatOpenAI

from langchain.schema import SystemMessage, HumanMessage

os.environ["OPENAI_API_KEY"] = "sua-chave-de-api"

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.5,
    max_tokens=100,
    timeout=10,
    max_retries=3
)

content = "You are a helpful assistant that translates English to French. Translate the user sentence."
user_question = "I love programming."

messages = [SystemMessage(content=content), HumanMessage(content=user_question)]
response = llm.invoke(messages)
print(response.content)





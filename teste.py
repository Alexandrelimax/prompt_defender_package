import os
import vertexai
from llm_defender.prompt_defender import PromptDefender
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI


os.environ["OPENAI_API_KEY"] = "API-KEY"

llm_openai = ChatOpenAI(
    model="gpt-4",
    temperature=0.5,
    max_tokens=100,
    timeout=10,
    max_retries=3
)

defender_openai = PromptDefender(llm_client=llm_openai)

#---------------------------------------------------------------------------------------------------

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

vertexai.init(project='', location='')

llm_vertex = ChatVertexAI(
    model="gemini-1.5-flash-001",
    temperature=1,
    max_tokens=2000,
    max_retries=3,
    stop=None,
)

defender_vertex = PromptDefender(llm_client=llm_vertex)

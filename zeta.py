from langchain_core.pydantic_v1 import BaseModel, Field

class RetornaTemperatura(BaseModel):
    location: str = Field(..., description="A cidade e estado, ex: Porto Alegre, RS")

llm_with_tools = llm.bind_tools([RetornaTemperatura])
response = llm_with_tools.invoke("Qual é o clima em São Paulo?")
print(response.tool_calls)
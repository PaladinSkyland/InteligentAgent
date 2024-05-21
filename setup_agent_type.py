from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from typing import Annotated

def setup_openai_agent(
        llm : Annotated[any, "The language model to be used."],
        tools : Annotated[list, "The tools to be used."],
        prompt : Annotated[ChatPromptTemplate, "The prompt to be used."]
    ) -> AgentExecutor:
    """
    This function sets up an OpenAI agent 
    """
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    from langchain.agents import AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


from langchain.agents import create_structured_chat_agent

def setup_simple_agent(
        llm : Annotated[any, "The language model to be used."],
        tools : Annotated[list, "The tools to be used."],
        prompt : Annotated[ChatPromptTemplate, "The prompt to be used."]
    ) -> AgentExecutor:
    """
    This function sets up a simple agent
    """
    agent = create_structured_chat_agent(llm, tools, prompt)
    
    from langchain.agents import AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from typing import Annotated, List, Tuple, Union

import matplotlib.pyplot as plt
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain


from langchain_openai.chat_models import ChatOpenAI


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

search = DuckDuckGoSearchRun()

search_tool = Tool.from_function(
    func=search.run,
    name="Search",
    description="useful for when you need to search the internet for information"
)

tools = [search_tool]

def create_agent_simple(
    llm,
    tools,
):
    from langchain.prompts import SystemMessagePromptTemplate
    from langchain.prompts import PromptTemplate
    from langchain.prompts import MessagesPlaceholder
    from langchain.prompts import HumanMessagePromptTemplate
    from langchain.agents import create_openai_functions_agent, initialize_agent

    message = [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
               MessagesPlaceholder(variable_name='chat_history', optional=True),
               HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
               MessagesPlaceholder(variable_name='agent_scratchpad')]

    from langchain.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate(messages=message)
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    from langchain.agents import AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


agent = create_agent_simple(llm, tools)
agent.invoke({"input": """Using the search tool, find the different between create_structured_chat_agent vs create_openai_functions_agent"""
              })
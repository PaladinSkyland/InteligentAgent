from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from langchain_openai.chat_models import ChatOpenAI


llm = ChatOpenAI(model="gpt-3.5-turbo")

from tools.usefultools import init_math_tool, init_duckduckgo_tool, init_word_problem_tool

math_tool = init_math_tool(llm)
word_problem_tool = init_word_problem_tool(llm)

from setup_agent_type import setup_openai_agent


from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import HumanMessagePromptTemplate

message = [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template="""
    You are a helpful assitant who is good in mathematics and good in solving problems.
    Do not calculate yourself let the tool do the calculation. Call one tool at a time.
    Logically arrive at the solution, and be factual. 
    In your answers, clearly detail the steps involved and give the final answer.""")),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['message'], template="{user_message}"))]

from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate(messages=message)

#print(word_problem_tool.run({"querry": "You have a fruit basket containing 5 different types of fruit: apples, oranges, bananas, grapes, and kiwis. Here's the scenario: You start with 6 apples, 8 oranges, 10 bananas, 20 grapes, and 15 kiwis. You decide to make fruit baskets for your friends. You make 3 fruit baskets, each containing an equal number of each type of fruit. After making the fruit baskets, you realize you want to add more variety. So, you go to the market and buy 4 pineapples, 3 watermelons, and 2 packs of mixed berries. Each pack of mixed berries contains 50 berries. You decide to distribute the new fruits evenly among the existing fruit baskets. Question: How many pieces of fruit are there in each of the updated fruit baskets, and how many pieces of each type of fruit do you have left over?"}))

agent = setup_openai_agent(llm,[math_tool],prompt)

probleme1 = """You have a fruit basket containing 5 different types of fruit: apples, oranges, bananas, grapes, and kiwis.

    Here's the scenario:

    You start with 6 apples, 8 oranges, 10 bananas, 20 grapes, and 15 kiwis.
    You decide to make fruit baskets for your friends. You make 3 fruit baskets, each containing an equal number of each type of fruit.
    After making the fruit baskets, you realize you want to add more variety. So, you go to the market and buy 4 pineapples, 3 watermelons, and 2 packs of mixed berries. Each pack of mixed berries contains 50 berries.
    You decide to distribute the new fruits evenly among the existing fruit baskets.

    Question: How many fruits are in each of the updated fruit baskets and how many pieces of each fruit in a single basket?"""

probleme2 = """Consider a triangle ABC where angle A is 60∘, side AC has a length of 5 units, and side BC has a length of 7 units.

    Find the length of side AB.
    Determine the measures of angles B and C.
    Calculate the area of triangle ABC.
    """


print(agent.invoke(
    {"user_message": probleme1}
    ))
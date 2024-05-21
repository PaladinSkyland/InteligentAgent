from langchain_community.tools import DuckDuckGoSearchRun, Tool

def init_duckduckgo_tool():
    search = DuckDuckGoSearchRun()

    search_tool = Tool.from_function(
        func=search.run,
        name="duck_duck_go",
        description="useful for when you need to search the internet for information"
    )
    return search_tool

from langchain.chains import LLMMathChain

def init_math_tool(llm):
    problem_chain = LLMMathChain.from_llm(llm=llm)
    math_tool = Tool.from_function(
        name="Calculator",
        func=problem_chain.run,
        description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions."
    )
    return math_tool



from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
def init_word_problem_tool(llm):
    word_problem_template = """You are a reasoning agent tasked with solving 
    the user's logic-based questions. Logically arrive at the solution, and be 
    factual. In your answers, clearly detail the steps involved and give the 
    final answer. Provide the response in bullet points. 
    Question  {question} Answer"""

    math_assistant_prompt = PromptTemplate(input_variables=["question"],
                                           template=word_problem_template
                                           )
    word_problem_chain = LLMChain(llm=llm,
                                  prompt=math_assistant_prompt)
    word_problem_tool = Tool.from_function(name="Reasoning_Tool",
                                           func=word_problem_chain.run,
                                           description="Useful for when you need to answer logic-based/reasoning questions.",
                                        )
    return word_problem_tool
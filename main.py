import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

load_dotenv()

def run_real_agent(task_description: str):
    print(f"Initializing REAL {name}...")
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    # Load actual working tools
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    try:
        result = agent.run(task_description)
        return result
    except Exception as e:
        return f"Agent execution failed. Did you set OPENAI_API_KEY? Error: {e}"

if __name__ == "__main__":
    print(run_real_agent("Who is the current CEO of Microsoft and what is their background?"))

import os
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def ask_langchain_gemini(featured_stock_data, start, end, show_intermediate_steps: False):
        
    # Load a DataFrame
    df = featured_stock_data.copy()
    
    df = df.iloc[int(start*len(df)/100):int(end*len(df)/100)]

    # Initialize the Gemini model via LangChain
    llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model="gemini-2.0-flash",
        temperature=0.0
    )

    # Create the DataFrame agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=show_intermediate_steps,
        allow_dangerous_code=True
    )

    # Ask a question!
    response = agent.invoke("What was the highest Close price last month?")
    return response

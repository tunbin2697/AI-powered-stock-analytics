import os
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class ChatBot:
    def __init__(self):
        # Initialize the Gemini model via LangChain
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model="gemini-2.0-flash",
            temperature=0.0
        )
            
    def ask_langchain_gemini(self, featured_stock_data, start, end, user_promt, show_intermediate_steps: False):
            
        # Load a DataFrame
        df = featured_stock_data.copy()
        
        df = df.iloc[int(start*len(df)/100):int(end*len(df)/100)]


        # Create the DataFrame agent
        agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=show_intermediate_steps,
            allow_dangerous_code=True
        )

        # Ask a question!
        response = agent.invoke(user_promt)
        return response

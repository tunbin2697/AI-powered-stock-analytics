import os
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
import requests

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "")
GEMINI_API_KEYS = [k.strip() for k in GEMINI_API_KEYS.split(",") if k.strip()]


class ChatBot:
    def __init__(self):
        # Initialize the Gemini model via LangChain
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEYS[0],
            model="gemini-2.0-flash",
            temperature=0.0
        )
            
    def ask_langchain_gemini(
        self,
        featured_stock_data,
        start,
        end,
        user_promt,
        show_intermediate_steps=False,
        chart_description=None
    ):
        df = featured_stock_data.copy()
        df = df.iloc[int(start*len(df)/100):int(end*len(df)/100)]

        try:
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                return_intermediate_steps=show_intermediate_steps,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
            response = agent.invoke(user_promt)
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            return response
        except Exception as e:
            # Fallback 1: Use chart_description as context for Gemini (LangChain)
            context = chart_description or ""
            context += f"\nUser Question: {user_promt}"
            try:
                llm_response = self.llm.invoke(context)
                if isinstance(llm_response, dict) and "content" in llm_response:
                    return str(llm_response["content"])
                return str(llm_response)
            except Exception as e2:
                # Fallback 2: Direct Gemini API call with key rotation
                prompt_text = user_promt + "\n" + (chart_description or "")
                api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt_text}
                            ]
                        }
                    ]
                }
                headers = {"Content-Type": "application/json"}
                last_error = ""
                for api_key in GEMINI_API_KEYS:
                    try:
                        resp = requests.post(
                            f"{api_url}?key={api_key}",
                            headers=headers,
                            json=payload,
                            timeout=30
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            # Gemini API returns candidates[0].content.parts[0].text
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts:
                                    return parts[0].get("text", "")
                            return str(data)
                        else:
                            last_error = f"HTTP {resp.status_code}: {resp.text}"
                            # If quota/limit error, try next key
                            if resp.status_code in (429, 403, 401, 503):
                                continue
                            else:
                                break
                    except Exception as e3:
                        last_error = str(e3)
                        continue
                return f"❌ All Gemini API keys failed. Last error: {last_error}"

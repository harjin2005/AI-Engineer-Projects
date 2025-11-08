import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError("Set GROQ_API_KEY in your .env file for best practice!")

llm = ChatOpenAI(
    openai_api_key=groq_key,
    openai_api_base="https://api.groq.com/openai/v1",
    model="qwen/qwen3-32b",   
    temperature=0.2,
    max_tokens=500,
)

prompts = {
    "market": '''You are a Market Researcher specializing in the hotel and PG sector.
Find the average monthly rent and key pricing trends for PGs and hotels in Bengaluru.
Summarize top competitors and their rates, and spot any pricing anomalies or seasonal swings.
Respond as a market expert.''',

    "occupancy": '''You are an Occupancy Analyst for PGs/hotels in Bengaluru.
Report recent occupancy rates and major seasons. Highlight periods of high/low demand and predict next month's expected occupancy.
Respond as an occupancy analyst.''',

    "accounting": '''You are the Accountant for a mid-sized PG/hotel operation in Bengaluru.
Summarize last quarter's profit, loss, and main expenses. Flag major cost drivers and offer suggestions to improve profit margin for owners.
Respond as a finance expert.''',

    "review": '''You are a Review/Sentiment Analyst for PGs/hotels in Bengaluru.
Analyze and summarize the latest online reviews (2025). Highlight customer complaints, satisfaction drivers, and actionable improvements.
Respond as a review analyst.''',
}

def run_agent(role, prompt_text):
    print(f"\n--- {role} Agent ---")
    chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt_text, input_variables=[]))
    result = chain.invoke({})
    print(result["text"])

if __name__ == "__main__":
    for key, label in [
        ("market", "Market Research"),
        ("occupancy", "Occupancy Analysis"),
        ("accounting", "Accounting"),
        ("review", "Review/Sentiment Analysis"),
    ]:
        run_agent(label, prompts[key])

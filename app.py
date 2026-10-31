import os
os.environ["OPENAI_API_KEY"] = "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"
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
parser = StrOutputParser()

parse_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract demand and technology gap from the user's message. "
     "Demand must be High, Medium, Low, or None. "
     "Gap must be Small, Medium, Large, or None. "
     "Reply in exactly this format: demand=<value>, gap=<value>"),
    ("human", "{question}"),])

parse_chain = parse_prompt | llm | parser

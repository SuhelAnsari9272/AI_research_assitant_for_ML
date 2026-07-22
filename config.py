
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model = "llama-3.3-70b-versatile",  #"llama-3.1-8b-instant",
               temperature=0.0, 
            #    max_tokens= 200
               )
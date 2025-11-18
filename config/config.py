# from config import client
# from openai import OpenAI
from groq import Groq
# from openai import OpenAI
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client =Groq(api_key=GROQ_API_KEY)


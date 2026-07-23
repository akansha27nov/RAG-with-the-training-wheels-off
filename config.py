import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = "./input_data/ai_hleg_ethics_guidelines_for_trustworthy_ai-en_87F84A41-A6E8-F38C-BFF661481B40077B_60419.pdf"

assert OPENAI_API_KEY, "Missing OPENAI_API_KEY in .env"
assert PDF_PATH, "Missing PDF_PATH in config file"
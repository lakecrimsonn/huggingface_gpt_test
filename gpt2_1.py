import os
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

pdf_summary_text = ""
pdf_file_path = r"pdf\paper.pdf"
pdf_file = open(pdf_file_path, 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

output_dir = "outputs" 
os.makedirs(output_dir, exist_ok=True)

for page_num in range(len(pdf_reader.pages)):
    page_text = pdf_reader.pages[page_num].extract_text().lower()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": f"Summarize this: {page_text}"},
        ],
    )
    
    page_summary = response.choices[0].message.content
    pdf_summary_text += page_summary + "\n"

pdf_summary_file = os.path.join(output_dir, os.path.basename(pdf_file_path).replace(os.path.splitext(pdf_file_path)[1], "_summary.txt"))

with open(pdf_summary_file, "w", encoding='utf-8') as file:
    file.write(pdf_summary_text)

pdf_file.close()

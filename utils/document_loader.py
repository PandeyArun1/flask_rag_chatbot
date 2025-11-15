import os
from PyPDF2 import PdfReader


def load_documents(folder_path):
    documents=[]
    for filename in os.listdir(folder_path):
        file_path=os.path.join(folder_path,filename)

        if filename.endswith('.txt'):
            with open(file_path,"r",encoding="utf-8") as f:
                text=f.read()
                documents.append({"filename":filename,"content":text})
        elif filename.endswith('.pdf'):
            reader=PdfReader(file_path)
            text=""

            for page in reader.pages:
                page_text = page.extract_text()
                if isinstance(page_text, list):
                    page_text = " ".join(page_text)  
                if page_text:
                    text += page_text + "\n"

            text = str(text).strip()
            documents.append({"filename": filename, "content": text})
    
    return documents
    





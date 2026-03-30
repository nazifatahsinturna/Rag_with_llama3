from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#reading the pdf
reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text+= page.extract_text()

text_spilitter = RecursiveCharacterTextSplitter (
    chunk_size = 500, # in each chunk there are 500 chars
    chunk_overlap = 100 #overlap chars to save the context
)

chunks = text_spilitter.split_text(text) #spliting the pdf text

print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks[:3]): #printing only first 3 chunks
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print("\n")
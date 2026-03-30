from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

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


#using ollama embeddings
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text" #llama3 is not ideal for embeddings
)

#storing in chromaDB
vector_db = Chroma.from_texts(
    texts = chunks,
    embedding = embedding_model,
    persist_directory="chroma_db"
)

#vector_db.persist()

print("The vector embeddings are created")
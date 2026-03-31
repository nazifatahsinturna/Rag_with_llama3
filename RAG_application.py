from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

#using ollama embeddings
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text" #llama3 is not ideal for embeddings
)

#loading the vector database
vector_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

#retriver creation
retriver = vector_db.as_retriever(search_kwargs={'k': 3})

#loaading the llm model
llm = OllamaLLM(model="llama3")

while True:
    question = input("Ask a question about your document or bye to exit: ");
    if question.lower() == 'bye':
        break
    #getting the relevent chunks
    docs = retriver.invoke(question)

    #context = "\n\n".join([doc.page_content for doc in docs])
    seen = set()
    unique_context = []
    for doc in docs:
        if doc.page_content not in seen:
            unique_context.append(doc.page_content)
            seen.add(doc.page_content)

    context = "\n\n".join(unique_context)

    #sending to the llm

    prompt = f"""
    Use the following context to answer the question.
    Context:{context}
    Question: {question}
    """

    response = llm.invoke(prompt)

    #printing the answer
    print("\n==========Answer==========\n")
    print(response)
from langchain.vectorstores import FAISS

from langchain.llms import GooglePalm
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.embeddings import GooglePalmEmbeddings

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
api_key="AIzaSyAZUYEICc9qGQSXXMtrvB4uvExddr-_TZ8"
llm = GooglePalm(google_api_key=api_key, temperature=0.1)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="bert-base-uncased",
                                                      cache_folder="instructor_cache")
e = instructor_embeddings.embed_query("What is the tip for bright skin ")

vectordb_file_path = "faiss_index"

def create_vectordb():
    loader = UnstructuredExcelLoader("dataset.xlsx", source_column="problem")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    global vectordb  # Declare vectordb as global if needed

    # Load the vector database from the local folder
    try:
        new_db = FAISS.load_local(vectordb_file_path, instructor_embeddings,allow_dangerous_deserialization=True)

        vectordb = new_db  # Move this line outside the try block
        print("Loaded vector database from faiss_index")
    except FileNotFoundError:
        print("Vector database 'faiss_index' not found. Please create it first.")
        return None

    retriever = vectordb.as_retriever(score_threshold=0.7)  # Now vectordb is accessible
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain




if __name__ == "__main__":
    create_vectordb()
    chain = get_qa_chain()
    print(chain("what to do for dark circles"))
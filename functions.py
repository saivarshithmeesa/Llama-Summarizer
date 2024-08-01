from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
import numpy as np
import PyPDF2
import re
import pytesseract
from PIL import Image
MODEL_PATH="./model/llama-2-7b-chat.Q8_0.gguf"

def load_moadel()-> LlamaCpp:
    """Loads Llama model"""
    callback:CallbackManager=CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers=50
    n_batch=3000
    Llama_model: LlamaCpp =LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        # n_gpu_layers=n_gpu_layers,
        # n_batch=n_batch,
        max_tokens=1000,
        n_ctx=3000,
        top_p=0.5,
        callback_manager=callback,
        verbose=True
    )
    return Llama_model

# def file_preprocessing(file):
#     querySplit = RecursiveCharacterTextSplitter (chunk_size=512,chunk_overlap  = 20,length_function = len,separators = ['\n'])
#     loader=PyPDFLoader(file)
#     data=loader.load()
#     queryChunks = querySplit.split_documents(data)
#     docs=[Document(page_content=str(t.page_content)) for t in queryChunks]
#     return docs

def extract_text_from_pdf(pdf_file):
    text = ""
    if str(pdf_file.name).split(".")[-1]=='pdf':
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            text+=pdf_reader.pages[page_num].extract_text()
        orgtext=' '.join(text.split())
        text=orgtext
    elif str(pdf_file.name).split(".")[-1]=='jpg' or str(pdf_file.name).split(".")[-1]=='jpeg' or str(pdf_file.name).split(".")[-1]=='png':
        photo=Image.open(pdf_file)
        docs=pytesseract.image_to_string(photo)
        text=docs
    # print(orgtext)
    return text

def extract_text_from_all_pdf(pdf_files):
    text=""
    for pdf in pdf_files:
        if str(pdf.name).split(".")[-1]=='pdf':
            pdf_reader=PyPDF2.PdfReader(pdf)
            for page in pdf_reader.pages:
                text+=page.extract_text()
        elif str(pdf.name).split(".")[-1]=='jpg' or str(pdf.name).split(".")[-1]=='jpeg' or str(pdf.name).split(".")[-1]=='png':
            photo=Image.open(pdf)
            docs=pytesseract.image_to_string(photo)
            text+=docs
    orgtext=' '.join(text.split())
    return orgtext

def text_to_chunk(text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunk=text_splitter.split_text(text)
    return chunk

def get_vectorstore(chunk):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'})
    vectorstore=FAISS.from_texts(texts=chunk,embedding=embeddings)
    return vectorstore

def get_conv(vector_store):
    llm=load_moadel()
    # memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    # conv_chain=ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vector_store.as_retriever(),
    #     memory=memory
    # )
    # templates="""
    # Use the following pieces of context to answer the question at the end.
    # If you DONOT KNOW THE ANSWER just say you DONOT KNOW THE ANSWER,DONOT try to make up answer.

    # Context:{context}
    # Question:{question}

    # Only return the helpful answer not in points below and nothing else.
    # Helpful and indepth answers
    # """
    templates="""
    [INST]<<SYS>>
    Use the following piece of context to answer the question at the end.If you don't know the answer, just say that you don't know,don't try to make up an answer.
    <</SYS>>
    {context} 
    Question:{question}[/INST]
    """.strip()
    qa_prompt=PromptTemplate(template=templates,input_variables=["context","question"])
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    chain=RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=vector_store.as_retriever(search_kwargs={'k':2}),chain_type_kwargs={'prompt':qa_prompt},memory=memory)
    return chain

def load_model(pdf_file,modes):
    llm=load_moadel()
    template = """
        [INST] <<SYS>>
        You are tasked with summarizing the following text in a {mode} manner within {lessword} to {words} words. Please ensure that your responses are socially unbiased and positive in nature.
        <</SYS>>
        {text}[/INST]
        """.strip()
    # max_word_count = 20  # Replace with your desired value
    # template = template.replace("{max}", str(max_word_count))
    prompt = PromptTemplate(template=template, input_variables=["text","mode","lessword","words"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(llm_chain)
    ans=llm_chain.run(text=extract_text_from_pdf(pdf_file),mode=modes,lessword=20,words=50)
    return ans

def load_modeltext(texts,modes):
    llm=load_moadel()
    template = """
        [INST] <<SYS>>
        You are tasked with summarizing the following text in a {mode} manner within {lessword} to {words} words. Please ensure that your responses are socially unbiased and positive in nature.
        <</SYS>>
        {text}[/INST]
        """.strip()
    # max_word_count = 20  # Replace with your desired value
    # template = template.replace("{max}", str(max_word_count))
    prompt = PromptTemplate(template=template, input_variables=["text","mode","lessword","words"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(llm_chain)
    ans=llm_chain.run(text=texts,mode=modes,lessword=93-10,words=93)
    return ans


def similarity_check(actual,pred):
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    query1=embeddings.embed_query(actual)
    query2=embeddings.embed_query(pred)
    vector1 = np.array(query1)
    vector2 = np.array(query2)
    cosine_sim_the = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cosine_sim_the
from functions import extract_text_from_all_pdf,extract_text_from_pdf,text_to_chunk,get_vectorstore,get_conv,load_model,load_modeltext,similarity_check
import streamlit as st
import base64
from mtranslate import translate
from htmlTemplates import css,bot_template,user_template,page_bg_image


def handleuser_input(user_input):
    response=st.session_state.conversation({'query':user_input})
    st.session_state.chat_history=response["chat_history"]
    st.write(user_template.replace("{{MSG}}",response["query"]),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}",response["result"]),unsafe_allow_html=True)
    


# @st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)


# st.set_page_config(layout="wide")
def main_section_one(uploaded_file,placeholder,section_options,text):
    # if "summary" not in st.session_state:
    #     st.session_state.summary=None
    # if "file_name" not in st.session_state:
    #     st.session_state.file_name=None
    st.title("Document Summarization using LLama-2:llama:")

    if uploaded_file:
        if(st.session_state.summary):
            st.write("Previous Summary Result:books:")
            for i in range (len(st.session_state.summary)):
                col1, col2 = st.columns(2)
                filepath = "data/"+st.session_state.file_name[i]
                with col1:
                    if(st.session_state.file_name[i].split(".")[-1]=='pdf'):
                        st.info("Uploaded File "+str(st.session_state.file_name[i]))
                        displayPDF(filepath)
                    elif(st.session_state.file_name[i].split(".")[-1]=='jpg' or st.session_state.file_name[i].split(".")[-1]=='jpeg' or st.session_state.file_name[i].split(".")[-1]=='png'):
                        st.info("Uploaded File "+str(st.session_state.file_name[i]))
                        st.image(filepath)
                with col2:
                    st.info("Summarization from File "+str(st.session_state.file_name[i]))
                    st.success(st.session_state.summary[i])
    if text:
        if(st.session_state.summary):
            st.write("Previous Summary Result:books:")
            for i in range (len(st.session_state.summary)):
                col1, col2 = st.columns(2)
                with col1:
                    st.info("Uploaded Text")
                    st.write(text)
                with col2:
                    st.info("Summarization for text")
                    st.success(st.session_state.summary[i])
    if(uploaded_file or text):
        option=st.selectbox("Choose the language",("English","Telugu"),placeholder="Select Language")
        modes=st.selectbox("Choose the Mode",("Detailed","Simple","Formal","Creative"),placeholder="Select Mode")
        if st.button("Summarize"):
            placeholder.empty()
            sum=[]
            if uploaded_file:
                file=[]
                for i in uploaded_file:
                    col1, col2 = st.columns(2)
                    filepath = "data/"+i.name
                    file.append(i.name)
                    with open(filepath, "wb") as temp_file:
                        temp_file.write(i.read())
                    with col1:
                        if(str(i.name).split(".")[-1]=='pdf'):
                            st.info("Uploaded File "+str(i.name))
                            displayPDF(filepath)
                        elif(str(i.name).split(".")[-1]=='jpg' or str(i.name).split(".")[-1]=='jpeg' or str(i.name).split(".")[-1]=='png'):
                            st.info("Uploaded File "+str(i.name))
                            st.image(i)
                    with col2:
                        a=st.info("Summarization in progress...")
                        summary=load_model(i,modes)
                        if(summary):
                            if(option=='Telugu'):
                                translation = translate(summary, 'te')
                                sum.append(translation)
                            else:
                                sum.append(summary)
                            a.empty()
                            st.info("Summarization Complete from File "+str(i.name))
                            if(option=='Telugu'):
                                st.success(translation)
                            else:
                                st.success(summary)
                st.session_state.file_name=file
            if text:
                textin=[]
                col1, col2 = st.columns(2)
                textin.append(text)
                with col1:
                    st.info("Uploaded Text")
                    st.write(text)
                with col2:
                    a=st.info("Summarization in progress...")
                    summary=load_modeltext(text,modes)
                    if(summary):
                        if(option=='Telugu'):
                            translation = translate(summary, 'te')
                            sum.append(translation)
                        else:
                            sum.append(summary)
                        a.empty()
                        st.info("Summarization Complete from Text")
                        st.success(summary)
                st.session_state.textin=text

            st.session_state.summary=sum
            placeholder.selectbox("Select an Task",section_options)


def main_section_two(pdf_docs,textin):
    st.write(css,unsafe_allow_html=True)
    st.header("Chat with multiple pdfs using LLama-2 :books:")
    user_question=st.text_input("ask about your pdf")
    if(st.session_state.chat_history):
        for i,message in enumerate(st.session_state.chat_history):
            if i%2 ==0:
                st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
    if user_question:
        if(st.session_state.conversation!=None):
            handleuser_input(user_question)
        else:
            st.warning('Please press Process button')
    with st.sidebar:
        st.subheader("Press Process Button Before Asking Questions")
        if st.button("Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    text=extract_text_from_all_pdf(pdf_docs)
                    chunk=text_to_chunk(text)
                else:
                    chunk=text_to_chunk(textin)
                vector_store=get_vectorstore(chunk)
                st.session_state.conversation=get_conv(vector_store)



st.set_page_config(page_title="S&A With LLama-2" , page_icon=":llama:",layout="wide")
st.write(page_bg_image,unsafe_allow_html=True)
def main():
    if "summary" not in st.session_state:
        st.session_state.summary=None
    if "file_name" not in st.session_state:
        st.session_state.file_name=None
    if "textin" not in st.session_state:
        st.session_state.textin=None
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    section_options = ["Summarize", "QA with PDF"]
    with st.sidebar:
        st.subheader("Select File")
        pdf_docs=st.file_uploader("Upload Your pdf here",accept_multiple_files=True)
        input_text=st.text_area("Enter text to Summarize")
    if pdf_docs or input_text:
        st.sidebar.title("Summarizar OR QA with PDF")
        select_box_placeholder = st.sidebar.empty()
        selected_section = select_box_placeholder.selectbox("Select Task", section_options)
        if selected_section == "Summarize":
            main_section_one(pdf_docs,select_box_placeholder,section_options,input_text)
        elif selected_section == "QA with PDF":
            main_section_two(pdf_docs,input_text)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if(st.session_state.summary):
                select_pred=st.selectbox("Select summary",st.session_state.summary)
            else:
                st.info("please sumummarize first")
        with col2:
            actual_text=st.text_input("Actual summary or abstract")
        if(actual_text):
            col3, col4 = st.sidebar.columns(2)
            with col3:
                st.info("Your similarity score")
            with col4:
                st.success(similarity_check(actual_text,select_pred))
    else:
        st.session_state.conversation=None
if __name__ =="__main__":
    main()
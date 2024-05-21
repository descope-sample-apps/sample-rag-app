import streamlit as st 
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain
from streamlit_oauth import OAuth2Component
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from descope import DescopeClient


DB_FAISS_PATH = './vectorstore/db_faiss'

st.title("Query Descope Docs")

AUTHORIZE_URL = st.secrets.get('AUTHORIZE_URL')
TOKEN_URL = st.secrets.get('TOKEN_URL')
REFRESH_TOKEN_URL = st.secrets.get('REFRESH_TOKEN_URL')
REVOKE_TOKEN_URL = st.secrets.get('REVOKE_TOKEN_URL')
CLIENT_ID = st.secrets.get('CLIENT_ID')
CLIENT_SECRET = st.secrets.get('CLIENT_SECRET')
REDIRECT_URI = st.secrets.get('REDIRECT_URI')
SCOPE = st.secrets.get('SCOPE')
PROJECT_ID = st.secrets.get('PROJECT_ID')

oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL)
descope_client = DescopeClient(project_id=PROJECT_ID)
loader = WebBaseLoader(["https://docs.descope.com/manage/idpapplications/oidc/", "https://docs.descope.com/manage/testusers/"])
data = loader.load()

if 'token' not in st.session_state:
    # If not, show authorize button
    result = oauth2.authorize_button(
        name="Continue with Descope",
        icon="https://images.ctfassets.net/xqb1f63q68s1/7D1PYGYvVgRNOBeiA6USQM/68b572056b5d38a769c71b0fba63b4e5/Descope_RGB_Icon-ForDarkBackground.svg",
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        key="descope",
        use_container_width=False,
        pkce='S256',
    )
    if result and 'token' in result:
        # If authorization successful, save token in session state
        st.session_state.token = result.get('token')
        st.experimental_rerun()
else:
    # If token exists in session state, show the token
    token = st.session_state['token']
    jwt_response = descope_client.validate_session(session_token=token.get("access_token"), audience=PROJECT_ID)
    roles = jwt_response.get("roles")

    list_of_documents = [
        Document(page_content=data[0].page_content, metadata=dict(role="Dev")),
        Document(page_content=data[1].page_content, metadata=dict(role="QA"))]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=st.secrets.openai_key
    )

    vector = FAISS.from_documents(list_of_documents, embeddings)
    vector.save_local(DB_FAISS_PATH)


    llm = ChatOpenAI(api_key=st.secrets.openai_key, temperature=0, model="gpt-4-turbo")
    retriever = vector.as_retriever()


    def conversational_chat(query):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an engineer. Answer the question based Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),])
        
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector.as_retriever(
            search_kwargs={'filter': {'role': roles[0]}}), verbose=True)
        response = chain({
            "question": query, "chat_history":[]
        })
        return response["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about Descope docs 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Enter your query:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")





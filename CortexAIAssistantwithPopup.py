# Import necessary libraries for Streamlit UI, JSON handling, regex, HTTP requests,
# Snowflake connectivity, pandas for data manipulation, Plotly for visualizations, and typing.
import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from snowflake.core import Root
from typing import Any, Dict, List, Optional, Tuple
import plotly.express as px
import time

# --- Snowflake/Cortex Configuration ---
# Define constants for Snowflake connection and Cortex API settings.
# These specify the host, database, schema, API endpoint, and semantic model for procurement data.
HOST = "gbjyvct-lsb50763.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000  # in milliseconds
CORTEX_SEARCH_SERVICES = "PROC_SERVICE"
SEMANTIC_MODEL = '@"AI"."DWH_MART"."PROCUREMENT_SEARCH"/procurement.yaml'

# --- Model Options ---
# List available Cortex language models for user selection.
MODELS = [
    "mistral-large",
    "snowflake-arctic",
    "llama3-70b",
    "llama3-8b",
]

# --- Streamlit Page Config ---
# Configure Streamlit app with title, wide layout, and auto sidebar.
st.set_page_config(
    page_title="Welcome to Cortex AI Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Session State Initialization ---
# Initialize session state to manage authentication, connections, chat history, and app settings.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None
    st.session_state.chat_history = []
    st.session_state.messages = []
if "last_suggestions" not in st.session_state:
    st.session_state.last_suggestions = []
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = None
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = None
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_sql" not in st.session_state:
    st.session_state.current_sql = None
if "current_summary" not in st.session_state:
    st.session_state.current_summary = None
if "service_metadata" not in st.session_state:
    st.session_state.service_metadata = [{"name": "PROC_SERVICE", "search_column": ""}]
if "selected_cortex_search_service" not in st.session_state:
    st.session_state.selected_cortex_search_service = "PROC_SERVICE"
if "model_name" not in st.session_state:
    st.session_state.model_name = "mistral-large"
if "num_retrieved_chunks" not in st.session_state:
    st.session_state.num_retrieved_chunks = 100
if "num_chat_messages" not in st.session_state:
    st.session_state.num_chat_messages = 10
if "use_chat_history" not in st.session_state:
    st.session_state.use_chat_history = True
if "clear_conversation" not in st.session_state:
    st.session_state.clear_conversation = False
if "rerun_trigger" not in st.session_state:
    st.session_state.rerun_trigger = False

# --- CSS Styling ---
# Apply custom CSS to hide Streamlit branding, prevent chat message shading, disable copy buttons,
# and fix the header (title + semantic model text) at the top of the page.
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    background-color: transparent !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow: hidden !important;
}
[data-testid="stChatMessageContent"] {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow: hidden !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
.copy-button, [data-testid="copy-button"], [title="Copy to clipboard"], [data-testid="stTextArea"] {
    display: none !important;
}
/* Fix the header (title + semantic model text) at the top */
.fixed-header {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    background-color: #ffffff !important; /* White background, adjust for dark theme if needed */
    z-index: 9999 !important; /* Ensure header stays on top */
    padding: 0.8rem 1.5rem !important; /* Responsive padding */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important; /* Subtle shadow */
    box-sizing: border-box !important;
}
/* Adjust width to account for sidebar */
@media (min-width: 992px) {
    .fixed-header {
        left: 250px !important; /* Match sidebar width */
        width: calc(100% - 250px) !important; /* Adjust for sidebar */
    }
}
/* On mobile, when sidebar collapses, use full width */
@media (max-width: 991px) {
    .fixed-header {
        left: 0 !important;
        width: 100% !important;
    }
}
/* Responsive font size for the title */
.fixed-header h1 {
    font-size: clamp(1.5rem, 2.5vw, 2rem) !important; /* Responsive font size */
    margin: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important; /* Handle long titles */
}
/* Ensure semantic model text doesn't break layout */
.fixed-header div:nth-child(2) {
    font-size: 0.9rem !important;
    color: #666 !important; /* Match the gray color from the screenshot */
    margin-top: 0.2rem !important;
}
/* Add padding to main content to prevent overlap with fixed header */
main {
    padding-top: 80px !important; /* Adjust based on header height */
}
</style>
""", unsafe_allow_html=True)

# --- Stream Text Function ---
# Stream text output in chunks with a delay for a typewriter effect.
def stream_text(text: str, chunk_size: int = 1, delay: float = 0.02):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(delay)

# --- Start New Conversation ---
# Reset session state to clear chat history, query results, and chart settings for a new conversation.
def start_new_conversation():
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.current_query = None
    st.session_state.current_results = None
    st.session_state.current_sql = None
    st.session_state.current_summary = None
    st.session_state.chart_x_axis = None
    st.session_state.chart_y_axis = None
    st.session_state.chart_type = "Bar Chart"
    st.session_state.last_suggestions = []
    st.session_state.clear_conversation = False
    st.session_state.rerun_trigger = True

# --- Initialize Service Metadata ---
# Fetch and store metadata for the Cortex search service, including the search column.
def init_service_metadata():
    st.session_state.service_metadata = [{"name": "PROC_SERVICE", "search_column": ""}]
    st.session_state.selected_cortex_search_service = "PROC_SERVICE"
    try:
        svc_search_col = session.sql("DESC CORTEX SEARCH SERVICE PROC_SERVICE;").collect()[0]["search_column"]
        st.session_state.service_metadata = [{"name": "PROC_SERVICE", "search_column": svc_search_col}]
    except Exception as e:
        st.error(f"❌ Failed to verify PROC_SERVICE: {str(e)}. Using default configuration.")

# --- Initialize Config Options ---
# Set up sidebar controls for clearing conversations and configuring model and context settings.
def init_config_options():
    st.sidebar.button("Clear conversation", on_click=start_new_conversation)
    st.sidebar.toggle("Use chat history", key="use_chat_history", value=True)
    with st.sidebar.expander("Advanced options"):
        st.selectbox("Select model:", MODELS, key="model_name")
        st.number_input(
            "Select number of context chunks",
            value=100,
            key="num_retrieved_chunks",
            min_value=1,
            max_value=400
        )
        st.number_input(
            "Select number of messages to use in chat history",
            value=10,
            key="num_chat_messages",
            min_value=1,
            max_value=100
        )

# --- Query Cortex Search Service ---
# Query the Cortex search service to retrieve relevant procurement data context for a given query.
def query_cortex_search_service(query):
    try:
        db, schema = session.get_current_database(), session.get_current_schema()
        root = Root(session)
        cortex_search_service = (
            root.databases[db]
            .schemas[schema]
            .cortex_search_services["PROC_SERVICE"]
        )
        context_documents = cortex_search_service.search(
            query, columns=[], limit=st.session_state.num_retrieved_chunks
        )
        results = context_documents.results
        service_metadata = st.session_state.service_metadata
        search_col = service_metadata[0]["search_column"]
        context_str = ""
        for i, r in enumerate(results):
            context_str += f"Context document {i+1}: {r[search_col]} \n" + "\n"
        return context_str
    except Exception as e:
        st.error(f"❌ Error querying Cortex Search service: {str(e)}")
        return ""

# --- Get Chat History ---
# Retrieve recent chat history based on user-specified message limits.
def get_chat_history():
    start_index = max(
        0, len(st.session_state.chat_history) - st.session_state.num_chat_messages
    )
    return st.session_state.chat_history[start_index : len(st.session_state.chat_history) - 1]

# --- Make Chat History Summary ---
# Summarize chat history and current question into a single query using Cortex.
def make_chat_history_summary(chat_history, question):
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extends the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history_str}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """
    summary = complete(st.session_state.model_name, prompt)
    return summary

# --- Create Prompt ---
# Construct a prompt for Cortex, combining chat history and search service context if applicable.
def create_prompt(user_question):
    chat_history_str = ""
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context = query_cortex_search_service(question_summary)
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        else:
            prompt_context = query_cortex_search_service(user_question)
    else:
        prompt_context = query_cortex_search_service(user_question)
        chat_history = []
    
    if not prompt_context.strip():
        return complete(st.session_state.model_name, user_question)
    
    prompt = f"""
        [INST]
        You are a helpful AI chat assistant with RAG capabilities. When a user asks you a question,
        you will also be given context provided between <context> and </context> tags. Use that context
        with the user's chat history provided in the between <chat_history> and </chat_history> tags
        to provide a summary that addresses the user's question. Ensure the answer is coherent, concise,
        and directly relevant to the user's question.

        If the user asks a generic question which cannot be answered with the given context or chat_history,
        just respond directly and concisely to the user's question using the LLM.

        <chat_history>
        {chat_history_str}
        </chat_history>
        <context>
        {prompt_context}
        </context>
        <question>
        {user_question}
        </question>
        [/INST]
        Answer:
    """
    return complete(st.session_state.model_name, prompt)

# --- Authentication Logic ---
# Handle user authentication with Snowflake and set up Snowpark session on success.
if not st.session_state.authenticated:
    st.title("Welcome to Snowflake Cortex AI")
    st.write("Please login to interact with your data")
    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")
    if st.button("Login"):
        try:
            conn = snowflake.connector.connect(
                user=st.session_state.username,
                password=st.session_state.password,
                account="gbjyvct-lsb50763",
                host=HOST,
                port=443,
                warehouse="COMPUTE_WH",
                role="ACCOUNTADMIN",
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn
            snowpark_session = Session.builder.configs({
                "connection": conn
            }).create()
            st.session_state.snowpark_session = snowpark_session
            with conn.cursor() as cur:
                cur.execute(f"USE DATABASE {DATABASE}")
                cur.execute(f"USE SCHEMA {SCHEMA}")
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = TRUE")
            st.session_state.authenticated = True
            st.success("Authentication successful! Redirecting...")
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
else:
    # --- Main App Logic ---
    # Initialize Snowpark session and Root object for authenticated users.
    session = st.session_state.snowpark_session
    root = Root(session)

    if st.session_state.rerun_trigger:
        st.session_state.rerun_trigger = False
        st.rerun()

    # --- Run Snowflake Query ---
    # Execute a SQL query and return results as a pandas DataFrame.
    def run_snowflake_query(query):
        try:
            if not query:
                return None
            df = session.sql(query)
            data = df.collect()
            if not data:
                return None
            columns = df.schema.names
            result_df = pd.DataFrame(data, columns=columns)
            return result_df
        except Exception as e:
            st.error(f"❌ SQL Execution Error: {str(e)}")
            return None

    # --- Query Classification Functions ---
    # Classify queries as structured, complete, summarize, suggestion, or greeting using regex.
    def is_structured_query(query: str):
        structured_patterns = [
            r'\b(count|number|where|group by|order by|sum|avg|max|min|total|how many|which|show|list|names?|are there any|rejected deliveries?|least|highest|duration|approval)\b',
            r'\b(vendor|supplier|requisition|purchase order|po|organization|department|buyer|delivery|received|billed|rejected|late|on time|late deliveries?|approval duration)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in structured_patterns)

    def is_complete_query(query: str):
        complete_patterns = [r'\b(generate|write|create|describe|explain)\b']
        return any(re.search(pattern, query.lower()) for pattern in complete_patterns)

    def is_summarize_query(query: str):
        summarize_patterns = [r'\b(summarize|summary|condense)\b']
        return any(re.search(pattern, query.lower()) for pattern in summarize_patterns)

    def is_question_suggestion_query(query: str):
        suggestion_patterns = [
            r'\b(what|which|how)\b.*\b(questions|type of questions|queries)\b.*\b(ask|can i ask|pose)\b',
            r'\b(give me|show me|list)\b.*\b(questions|examples|sample questions)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in suggestion_patterns)

    def is_greeting_query(query: str):
        greeting_patterns = [
            r'^\b(hello|hi|hey|greet)\b$',
            r'^\b(hello|hi|hey|greet)\b\s.*$'
        ]
        return any(re.search(pattern, query.lower()) for pattern in greeting_patterns)

    # --- Cortex Complete Function ---
    # Call Cortex COMPLETE function to generate a response for a given prompt.
    def complete(model, prompt):
        try:
            prompt = prompt.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}') AS response"
            result = session.sql(query).collect()
            return result[0]["RESPONSE"]
        except Exception as e:
            st.error(f"❌ COMPLETE Function Error: {str(e)}")
            return None

    # --- Summarize Function ---
    # Call Cortex SUMMARIZE function to condense text input.
    def summarize(text):
        try:
            text = text.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query).collect()
            return result[0]["SUMMARY"]
        except Exception as e:
            st.error(f"❌ SUMMARIZE Function Error: {str(e)}")
            return None

    # --- Parse SSE Response ---
    # Parse Server-Sent Events (SSE) responses from Cortex API into a list of events.
    def parse_sse_response(response_text: str) -> List[Dict]:
        events = []
        lines = response_text.strip().split("\n")
        current_event = {}
        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                if data_str != "[DONE]":
                    try:
                        data_json = json.loads(data_str)
                        current_event["data"] = data_json
                        events.append(current_event)
                        current_event = {}
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Failed to parse SSE data: {str(e)} - Data: {data_str}")
        return events

    # --- Process SSE Response ---
    # Extract SQL or search results from SSE responses based on query type.
    def process_sse_response(response, is_structured):
        sql = ""
        search_results = []
        if not response:
            return sql, search_results
        try:
            for event in response:
                if event.get("event") == "message.delta" and "data" in event:
                    delta = event["data"].get("delta", {})
                    content = delta.get("content", [])
                    for item in content:
                        if item.get("type") == "tool_results":
                            tool_results = item.get("tool_results", {})
                            if "content" in tool_results:
                                for result in tool_results["content"]:
                                    if result.get("type") == "json":
                                        result_data = result.get("json", {})
                                        if is_structured and "sql" in result_data:
                                            sql = result_data.get("sql", "")
                                        elif not is_structured and "searchResults" in result_data:
                                            search_results = [sr["text"] for sr in result_data["searchResults"]]
        except Exception as e:
            st.error(f"❌ Error Processing Response: {str(e)}")
        return sql.strip(), search_results

    # --- Snowflake API Call ---
    # Make HTTP request to Cortex API for structured or unstructured queries.
    def snowflake_api_call(query: str, is_structured: bool = False):
        payload = {
            "model": st.session_state.model_name,
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "tools": []
        }
        if is_structured:
            payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
            payload["tool_resources"] = {"analyst1": {"semantic_model_file": SEMANTIC_MODEL}}
        else:
            payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
            payload["tool_resources"] = {"search1": {"name": "PROC_SERVICE", "max_results": st.session_state.num_retrieved_chunks}}
        try:
            resp = requests.post(
                url=f"https://{HOST}{API_ENDPOINT}",
                json=payload,
                headers={
                    "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
                    "Content-Type": "application/json",
                },
                timeout=API_TIMEOUT // 1000
            )
            if resp.status_code < 400:
                if not resp.text.strip():
                    st.error("❌ API returned an empty response.")
                    return None
                return parse_sse_response(resp.text)
            else:
                raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"❌ API Request Error: {str(e)}")
            return None

    # --- Summarize Unstructured Answer ---
    # Summarize unstructured responses into concise bullet points.
    def summarize_unstructured_answer(answer):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|")\s', answer)
        return "\n".join(f"- {sent.strip()}" for sent in sentences[:6])

    # --- Suggest Sample Questions ---
    # Generate sample procurement-related questions when a query fails or is ambiguous.
    def suggest_sample_questions(query: str) -> List[str]:
        try:
            prompt = (
                f"The user asked: '{query}'. This question may be ambiguous or unclear in the context of a business-facing procurement analytics assistant. "
                f"Generate 3-5 clear, concise sample questions related to purchase orders, requisitions, suppliers, or procurement```python
# Import necessary libraries for Streamlit UI, JSON handling, regex, HRTP requests,
# Snowflake connectivity, pandas for data manipulation, Plotly for visualizations, and typing.
import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from snowflake.core import Root
from typing import Any, Dict, List, Optional, Tuple
import plotly.express as px
import time

# --- Snowflake/Cortex Configuration ---
# Define constants for Snowflake connection and Cortex API settings.
# These specify the host, database, schema, API endpoint, and semantic model for procurement data.
HOST = "gbjyvct-lsb50763.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000  # in milliseconds
CORTEX_SEARCH_SERVICES = "PROC_SERVICE"
SEMANTIC_MODEL = '@"AI"."DWH_MART"."PROCUREMENT_SEARCH"/procurement.yaml'

# --- Model Options ---
# List available Cortex language models for user selection.
MODELS = [
    "mistral-large",
    "snowflake-arctic",
    "llama3-70b",
    "llama3-8b",
]

# --- Streamlit Page Config ---
# Configure Streamlit app with title, wide layout, and auto sidebar.
st.set_page_config(
    page_title="Welcome to Cortex AI Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Session State Initialization ---
# Initialize session state to manage authentication, connections, chat history, and app settings.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None
    st.session_state.chat_history = []
    st.session_state.messages = []
if "last_suggestions" not in st.session_state:
    st.session_state.last_suggestions = []
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = None
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = None
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_sql" not in st.session_state:
    st.session_state.current_sql = None
if "current_summary" not in st.session_state:
    st.session_state.current_summary = None
if "service_metadata" not in st.session_state:
    st.session_state.service_metadata = [{"name": "PROC_SERVICE", "search_column": ""}]
if "selected_cortex_search_service" not in st.session_state:
    st.session_state.selected_cortex_search_service = "PROC_SERVICE"
if "model_name" not in st.session_state:
    st.session_state.model_name = "mistral-large"
if "num_retrieved_chunks" not in st.session_state:
    st.session_state.num_retrieved_chunks = 100
if "num_chat_messages" not in st.session_state:
    st.session_state.num_chat_messages = 10
if "use_chat_history" not in st.session_state:
    st.session_state.use_chat_history = True
if "clear_conversation" not in st.session_state:
    st.session_state.clear_conversation = False
if "rerun_trigger" not in st.session_state:
    st.session_state.rerun_trigger = False

# --- CSS Styling ---
# Apply custom CSS to hide Streamlit branding, prevent chat message shading, disable copy buttons,
# and fix the header (title + semantic model text) at the top of the page.
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    background-color: transparent !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow: hidden !important;
}
[data-testid="stChatMessageContent"] {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow: hidden !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
.copy-button, [data-testid="copy-button"], [title="Copy to clipboard"], [data-testid="stTextArea"] {
    display: none !important;
}
/* Fix the header (title + semantic model text) at the top */
.fixed-header {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    background-color: #ffffff !important; /* White background, adjust for dark theme if needed */
    z-index: 9999 !important; /* Ensure header stays on top */
    padding: 0.8rem 1.5rem !important; /* Responsive padding */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important; /* Subtle shadow */
    box-sizing: border-box !important;
}
/* Adjust width to account for sidebar */
@media (min-width: 992px) {
    .fixed-header {
        left: 250px !important; /* Match sidebar width */
        width: calc(100% - 250px) !important; /* Adjust for sidebar */
    }
}
/* On mobile, when sidebar collapses, use full width */
@media (max-width: 991px) {
    .fixed-header {
        left: 0 !important;
        width: 100% !important;
    }
}
/* Responsive font size for the title */
.fixed-header h1 {
    font-size: clamp(1.5rem, 2.5vw, 2rem) !important; /* Responsive font size */
    margin: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important; /* Handle long titles */
}
/* Ensure semantic model text doesn't break layout */
.fixed-header div:nth-child(2) {
    font-size: 0.9rem !important;
    color: #666 !important; /* Match the gray color from the screenshot */
    margin-top: 0.2rem !important;
}
/* Add padding to main content to prevent overlap with fixed header */
main {
    padding-top: 80px !important; /* Adjust based on header height */
}
</style>
""", unsafe_allow_html=True)

# --- Stream Text Function ---
# Stream text output in chunks with a delay for a typewriter effect.
def stream_text(text: str, chunk_size: int = 1, delay: float = 0.02):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(delay)

# --- Start New Conversation ---
# Reset session state to clear chat history, query results, and chart settings for a new conversation.
def start_new_conversation():
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.current_query = None
    st.session_state.current_results = None
    st.session_state.current_sql = None
    st.session_state.current_summary = None
    st.session_state.chart_x_axis = None
    st.session_state.chart_y_axis = None
    st.session_state.chart_type = "Bar Chart"
    st.session_state.last_suggestions = []
    st.session_state.clear_conversation = False
    st.session_state.rerun_trigger = True

# --- Initialize Service Metadata ---
# Fetch and store metadata for the Cortex search service, including the search column.
def init_service_metadata():
    st.session_state.service_metadata = [{"name": "PROC_SERVICE", "search_column": ""}]
    st.session_state.selected_cortex_search_service = "PROC_SERVICE"
    try:
        svc_search_col = session.sql("DESC CORTEX SEARCH SERVICE PROC_SERVICE;").collect()[0]["search_column"]
        st.session_state.service_metadata = [{"name": "PROC_SERVICE", "search_column": svc_search_col}]
    except Exception as e:
        st.error(f"❌ Failed to verify PROC_SERVICE: {str(e)}. Using default configuration.")

# --- Initialize Config Options ---
# Set up sidebar controls for clearing conversations and configuring model and context settings.
def init_config_options():
    st.sidebar.button("Clear conversation", on_click=start_new_conversation)
    st.sidebar.toggle("Use chat history", key="use_chat_history", value=True)
    with st.sidebar.expander("Advanced options"):
        st.selectbox("Select model:", MODELS, key="model_name")
        st.number_input(
            "Select number of context chunks",
            value=100,
            key="num_retrieved_chunks",
            min_value=1,
            max_value=400
        )
        st.number_input(
            "Select number of messages to use in chat history",
            value=10,
            key="num_chat_messages",
            min_value=1,
            max_value=100
        )

# --- Query Cortex Search Service ---
# Query the Cortex search service to retrieve relevant procurement data context for a given query.
def query_cortex_search_service(query):
    try:
        db, schema = session.get_current_database(), session.get_current_schema()
        root = Root(session)
        cortex_search_service = (
            root.databases[db]
            .schemas[schema]
            .cortex_search_services["PROC_SERVICE"]
        )
        context_documents = cortex_search_service.search(
            query, columns=[], limit=st.session_state.num_retrieved_chunks
        )
        results = context_documents.results
        service_metadata = st.session_state.service_metadata
        search_col = service_metadata[0]["search_column"]
        context_str = ""
        for i, r in enumerate(results):
            context_str += f"Context document {i+1}: {r[search_col]} \n" + "\n"
        return context_str
    except Exception as e:
        st.error(f"❌ Error querying Cortex Search service: {str(e)}")
        return ""

# --- Get Chat History ---
# Retrieve recent chat history based on user-specified message limits.
def get_chat_history():
    start_index = max(
        0, len(st.session_state.chat_history) - st.session_state.num_chat_messages
    )
    return st.session_state.chat_history[start_index : len(st.session_state.chat_history) - 1]

# --- Make Chat History Summary ---
# Summarize chat history and current question into a single query using Cortex.
def make_chat_history_summary(chat_history, question):
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extends the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history_str}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """
    summary = complete(st.session_state.model_name, prompt)
    return summary

# --- Create Prompt ---
# Construct a prompt for Cortex, combining chat history and search service context if applicable.
def create_prompt(user_question):
    chat_history_str = ""
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context = query_cortex_search_service(question_summary)
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        else:
            prompt_context = query_cortex_search_service(user_question)
    else:
        prompt_context = query_cortex_search_service(user_question)
        chat_history = []
    
    if not prompt_context.strip():
        return complete(st.session_state.model_name, user_question)
    
    prompt = f"""
        [INST]
        You are a helpful AI chat assistant with RAG capabilities. When a user asks you a question,
        you will also be given context provided between <context> and </context> tags. Use that context
        with the user's chat history provided in the between <chat_history> and </chat_history> tags
        to provide a summary that addresses the user's question. Ensure the answer is coherent, concise,
        and directly relevant to the user's question.

        If the user asks a generic question which cannot be answered with the given context or chat_history,
        just respond directly and concisely to the user's question using the LLM.

        <chat_history>
        {chat_history_str}
        </chat_history>
        <context>
        {prompt_context}
        </context>
        <question>
        {user_question}
        </question>
        [/INST]
        Answer:
    """
    return complete(st.session_state.model_name, prompt)

# --- Authentication Logic ---
# Handle user authentication with Snowflake and set up Snowpark session on success.
if not st.session_state.authenticated:
    st.title("Welcome to Snowflake Cortex AI")
    st.write("Please login to interact with your data")
    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")
    if st.button("Login"):
        try:
            conn = snowflake.connector.connect(
                user=st.session_state.username,
                password=st.session_state.password,
                account="gbjyvct-lsb50763",
                host=HOST,
                port=443,
                warehouse="COMPUTE_WH",
                role="ACCOUNTADMIN",
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn
            snowpark_session = Session.builder.configs({
                "connection": conn
            }).create()
            st.session_state.snowpark_session = snowpark_session
            with conn.cursor() as cur:
                cur.execute(f"USE DATABASE {DATABASE}")
                cur.execute(f"USE SCHEMA {SCHEMA}")
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = TRUE")
            st.session_state.authenticated = True
            st.success("Authentication successful! Redirecting...")
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
else:
    # --- Main App Logic ---
    # Initialize Snowpark session and Root object for authenticated users.
    session = st.session_state.snowpark_session
    root = Root(session)

    if st.session_state.rerun_trigger:
        st.session_state.rerun_trigger = False
        st.rerun()

    # --- Run Snowflake Query ---
    # Execute a SQL query and return results as a pandas DataFrame.
    def run_snowflake_query(query):
        try:
            if not query:
                return None
            df = session.sql(query)
            data = df.collect()
            if not data:
                return None
            columns = df.schema.names
            result_df = pd.DataFrame(data, columns=columns)
            return result_df
        except Exception as e:
            st.error(f"❌ SQL Execution Error: {str(e)}")
            return None

    # --- Query Classification Functions ---
    # Classify queries as structured, complete, summarize, suggestion, or greeting using regex.
    def is_structured_query(query: str):
        structured_patterns = [
            r'\b(count|number|where|group by|order by|sum|avg|max|min|total|how many|which|show|list|names?|are there any|rejected deliveries?|least|highest|duration|approval)\b',
            r'\b(vendor|supplier|requisition|purchase order|po|organization|department|buyer|delivery|received|billed|rejected|late|on time|late deliveries?|approval duration)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in structured_patterns)

    def is_complete_query(query: str):
        complete_patterns = [r'\b(generate|write|create|describe|explain)\b']
        return any(re.search(pattern, query.lower()) for pattern in complete_patterns)

    def is_summarize_query(query: str):
        summarize_patterns = [r'\b(summarize|summary|condense)\b']
        return any(re.search(pattern, query.lower()) for pattern in summarize_patterns)

    def is_question_suggestion_query(query: str):
        suggestion_patterns = [
            r'\b(what|which|how)\b.*\b(questions|type of questions|queries)\b.*\b(ask|can i ask|pose)\b',
            r'\b(give me|show me|list)\b.*\b(questions|examples|sample questions)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in suggestion_patterns)

    def is_greeting_query(query: str):
        greeting_patterns = [
            r'^\b(hello|hi|hey|greet)\b$',
            r'^\b(hello|hi|hey|greet)\b\s.*$'
        ]
        return any(re.search(pattern, query.lower()) for pattern in greeting_patterns)

    # --- Cortex Complete Function ---
    # Call Cortex COMPLETE function to generate a response for a given prompt.
    def complete(model, prompt):
        try:
            prompt = prompt.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}') AS response"
            result = session.sql(query).collect()
            return result[0]["RESPONSE"]
        except Exception as e:
            st.error(f"❌ COMPLETE Function Error: {str(e)}")
            return None

    # --- Summarize Function ---
    # Call Cortex SUMMARIZE function to condense text input.
    def summarize(text):
        try:
            text = text.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query).collect()
            return result[0]["SUMMARY"]
        except Exception as e:
            st.error(f"❌ SUMMARIZE Function Error: {str(e)}")
            return None

    # --- Parse SSE Response ---
    # Parse Server-Sent Events (SSE) responses from Cortex API into a list of events.
    def parse_sse_response(response_text: str) -> List[Dict]:
        events = []
        lines = response_text.strip().split("\n")
        current_event = {}
        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                if data_str != "[DONE]":
                    try:
                        data_json = json.loads(data_str)
                        current_event["data"] = data_json
                        events.append(current_event)
                        current_event = {}
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Failed to parse SSE data: {str(e)} - Data: {data_str}")
        return events

    # --- Process SSE Response ---
    # Extract SQL or search results from SSE responses based on query type.
    def process_sse_response(response, is_structured):
        sql = ""
        search_results = []
        if not response:
            return sql, search_results
        try:
            for event in response:
                if event.get("event") == "message.delta" and "data" in event:
                    delta = event["data"].get("delta", {})
                    content = delta.get("content", [])
                    for item in content:
                        if item.get("type") == "tool_results":
                            tool_results = item.get("tool_results", {})
                            if "content" in tool_results:
                                for result in tool_results["content"]:
                                    if result.get("type") == "json":
                                        result_data = result.get("json", {})
                                        if is_structured and "sql" in result_data:
                                            sql = result_data.get("sql", "")
                                        elif not is_structured and "searchResults" in result_data:
                                            search_results = [sr["text"] for sr in result_data["searchResults"]]
        except Exception as e:
            st.error(f"❌ Error Processing Response: {str(e)}")
        return sql.strip(), search_results

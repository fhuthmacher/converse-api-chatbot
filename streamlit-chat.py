import streamlit as st
import boto3
from datetime import datetime
import pandas as pd
import json
from pyathena import connect
import os
from dotenv import load_dotenv, find_dotenv
from typing import Any, Dict, List
import inspect
from pydantic import BaseModel, Field, create_model
import pytz
import re

local_env_filename = 'config.env'
load_dotenv(find_dotenv(local_env_filename),override=True)
os.environ['REGION'] = os.getenv('REGION')
REGION = os.environ['REGION']
os.environ['S3_BUCKET_NAME'] = os.getenv('S3_BUCKET_NAME')
S3_BUCKET_NAME = os.environ['S3_BUCKET_NAME']
os.environ['GUARDRAIL_IDENTIFIER'] = os.getenv('GUARDRAIL_IDENTIFIER')
GUARDRAIL_IDENTIFIER = os.environ['GUARDRAIL_IDENTIFIER']
os.environ['GUARDRAIL_VERSION'] = os.getenv('GUARDRAIL_VERSION')
GUARDRAIL_VERSION = os.environ['GUARDRAIL_VERSION']
os.environ['KB_ID'] = os.getenv('KB_ID')
KB_ID = os.environ['KB_ID']

bedrock_runtime_client = boto3.client('bedrock-runtime')

bedrock_agent_client = boto3.client("bedrock-agent-runtime")

MODEL_IDS = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]

CHAT_HISTORY_PATTERN = r'<CHAT_HISTORY>(.*?)</CHAT_HISTORY>'

# Strip out the portion of the response with regex.
def extract_with_regex(response, regex):
    matches = re.search(regex, response, re.DOTALL)
    # Extract the matched content, if any
    return matches.group(1).strip() if matches else None

# 4. Define decorator class for our tool definitions
def bedrock_tool(name, description):
    def decorator(func):
        input_model = create_model(
            func.__name__ + "_input",
            **{
                name: (param.annotation, param.default)
                for name, param in inspect.signature(func).parameters.items()
                if param.default is not inspect.Parameter.empty
            },
        )

        func.bedrock_schema = {
            'toolSpec': {
                'name': name,
                'description': description,
                'inputSchema': {
                    'json': input_model.schema()
                }
            }
        }
        return func

    return decorator

# 5. Define tools 
class ToolsList:

    @bedrock_tool(
        name="search_knowledge_database",
        description="search knowledge database to find answers to frequently asked customer questions (FAQ)"
    )
    def search_knowledge_database(self, user_question: str = Field(..., description="customer question")):
        numberOfResults=3
        response = bedrock_agent_client.retrieve(
            retrievalQuery= {
                'text': user_question
            },
            knowledgeBaseId=KB_ID,
            retrievalConfiguration= {
                'vectorSearchConfiguration': {
                    'numberOfResults': numberOfResults,
                    'overrideSearchType': "HYBRID"
                }
            }
        )
        
        contexts = []
        retrievalResults = response.get('retrievalResults')
        #print(retrievalResults)
        for retrievedResult in retrievalResults:
            print(type(retrievedResult))
            print(str(retrievedResult))
            
            text = retrievedResult.get('content').get('text')
            # Remove the "Document 1: " prefix if it exists
            if text.startswith("Document 1: "):
                text = text[len("Document 1: "):]
            contexts.append(text)
        contexts_string = ', '.join(contexts)
        return contexts_string


    @bedrock_tool(
        name="query_database",
        description="Get auto parts for a given car model year, make, engine and model"
    )
    def query_database(self, 
                        car_model_year_ref: int = Field(..., description="car model year"),
                        car_model_make_ref: str = Field(..., description="car model make"),
                        car_model_ref: str = Field(..., description="car model"),
                        car_model_engine: str = Field(..., description="car engine")):
        print(f"{datetime.now():%H:%M:%S} - query_database for {car_model_year_ref}, {car_model_make_ref}, {car_model_ref}, {car_model_engine}")
        
        try:
            # still search database even if we are still missing some information
            if car_model_year_ref == '?' or car_model_year_ref == 'What is the model year of your Suzuki?':
                print(f"{datetime.now():%H:%M:%S} - car_model_year_ref had value {car_model_year_ref}")
                car_model_year_ref = '%'
            if car_model_make_ref == '?' or car_model_make_ref == 'What is the car make?': 
                car_model_make_ref = '%'
                print(f"{datetime.now():%H:%M:%S} - car_model_make_ref had value {car_model_make_ref}")
            if car_model_ref == '?' or car_model_ref == 'What is the specific model name/number (e.g. Swift, Grand Vitara)?': 
                car_model_ref = '%'
                print(f"{datetime.now():%H:%M:%S} - car_model_ref had value {car_model_ref}")
            if car_model_engine == '?' or car_model_engine == 'What is the engine size/specification?': 
                car_model_engine = '%'
                print(f"{datetime.now():%H:%M:%S} - car_model_engine had value {car_model_engine}")

            query = f'SELECT parts FROM "demo-catalog"."data" WHERE CAST(year AS VARCHAR) LIKE \'%{car_model_year_ref}%\' AND UPPER(make) LIKE UPPER(\'%{car_model_make_ref}%\') AND UPPER(model) LIKE UPPER(\'%{car_model_ref}%\') AND UPPER(engine) LIKE UPPER(\'%{car_model_engine}%\')'
            print(f"{datetime.now():%H:%M:%S} - query: {query}")
            cursor = connect(s3_staging_dir=f"s3://{S3_BUCKET_NAME}/athena/",
                                region_name=REGION).cursor()
            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall()).to_string(index=False)
            print(f"{datetime.now().strftime('%H:%M:%S')} - Tool result: {df}\n")
        except Exception as e:
            print(f"{datetime.now().strftime('%H:%M:%S')} - Error: {e}")
            raise
        return df


# 6. Define helper functions for converse API calls with function calling
toolConfig = {
    'tools': [tool.bedrock_schema for tool in ToolsList.__dict__.values() if hasattr(tool, 'bedrock_schema')],
    'toolChoice': {'auto': {}}
}

def converse_with_tools(modelId, messages, system='', toolConfig=None):
    print(f'toolConfig: {toolConfig}')
    return bedrock_runtime_client.converse(
        modelId=modelId,
        system=system,
        messages=messages,
        toolConfig=toolConfig,
        guardrailConfig={
            'guardrailIdentifier': GUARDRAIL_IDENTIFIER,
            'guardrailVersion': GUARDRAIL_VERSION,
            # 'trace': 'enabled'|'disabled'
        },
    )

def converse(tool_class, modelId, prompt, system='', toolConfig=None):
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    st.session_state.history.append(messages)
    print(f"{datetime.now():%H:%M:%S} - Invoking model...")
    MAX_LOOPS = 6
    loop_count = 0
    continue_loop = True

    while continue_loop:
        loop_count = loop_count + 1
        # print(f"Loop count: {loop_count}")
        if loop_count >= MAX_LOOPS:
            # print(f"Hit loop limit: {loop_count}")
            break

        output = converse_with_tools(modelId, messages, system, toolConfig)
        messages.append(output['output']['message'])
        # print(f"{datetime.now():%H:%M:%S} - Got output from model...")

        function_calling = [c['toolUse'] for c in output['output']['message']['content'] if 'toolUse' in c]
        # print(f'length of function_calling list: {len(function_calling)}')
        if function_calling:
            tool_result_message = {"role": "user", "content": []}
            for function in function_calling:
                # print(f"{datetime.now():%H:%M:%S} - Function calling - Calling tool...")
                tool_name = function['name']
                tool_args = function['input'] or {}
                tool_response = getattr(tool_class, tool_name)(**tool_args)
                # print(f"{datetime.now():%H:%M:%S} - Function calling - Got tool response...")
                tool_result_message['content'].append({
                    'toolResult': {
                        'toolUseId': function['toolUseId'],
                        'content': [{"text": tool_response}]
                    }
                })
            messages.append(tool_result_message)
            # print(f"{datetime.now():%H:%M:%S} - Function calling - Calling model with result...")
                        
        else:

            # check if further messages are required by going through content
            response_content_blocks = output['output']['message'].get('content')
            for content_block in response_content_blocks:
                text = content_block.get('text')
                if text is not None:
                    continue_loop = False
            # print(f"{datetime.now():%H:%M:%S} - Function calling - Got final answer.")


    return messages, output

### Streamlit setup
st.set_page_config(layout="wide")
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit UI
st.markdown("### Converse API for Amazon Bedrock - Function Calling Demo")
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
tabs = st.tabs(["Conversation", "Message Details"])

with tabs[0]:
    st.sidebar.image('./images/bedrock.png', width=60)
    st.sidebar.divider()
    modelId = st.sidebar.selectbox("Model ID", MODEL_IDS, index=0)
    st.sidebar.divider()
    flow = st.sidebar.container(border=True)
    flow.markdown(f"**Flow status:**")
    st.sidebar.divider()
    st.sidebar.image('./images/AWS_logo_RGB.png', width=40)
    with st.expander("Examples", expanded=False):
        st.markdown("""
                    * Hi, I am looking for parts for my Suzuki?
                    * Does su-5002-replacement-air-filter work with my Suzuki?
                    * Hi, I am looking for an autopart for my Suzuki. The model year is 2015, the engine is 376. It's the LTF400F KingQuad FSi model.
                    * Is K&N parts better than Eaton?
                    """)

    prompt = st.text_input("Enter your question about auto parts", "")

    if st.button("Submit"):
        ### ADJUST YOUR SYSTEM PROMPT HERE - IF DESIRED ###
        system_prompt = [{"text": f"You are a helpful auto part shopping assistant. You're provided with 2 tools: \
                          'query_database' which searches for auto parts for a given car model year, make, car model, and engine; and \
                          'search_knowledge_database' which enables you to find answers to frequently asked customer questions (FAQ). \
                            Follow the below instructions: \
                            1) Only use the tool if required. You can call a given tool multiple times in the same response if required. \
                            2) You can call multiple tools in the same response if required. \
                            3) Don't make assumptions, if required, ask the user for more information. \
                            4) If you call a function but you don't have all a respective input parameter, then use ? as value and other values for this parameter. \
                            5) Review the chat history to determine if some of the input parameters were provided in prior responses. \
                            6) Don't make reference to the tools in your final answer. \
                            7) Think step by step before providing an answer. \
                            Chat History: \
                            {st.session_state.history}"}]

        with st.spinner("In progress..."):            
            messages, output = converse(ToolsList(), modelId, prompt, system_prompt, toolConfig)

        # final_response = output['output']['message']['content'][0]['text']
        st.divider()
        st.markdown("**Conversation**")
        for message in messages:
            role = message['role']
            content_items = message['content']
            for item in content_items:
                if 'text' in item:
                    st.markdown(f"**{role.capitalize()}:** {item['text']}")
                elif 'toolResult' in item:
                    with st.expander(f"**Tool Result**", expanded=True):
                        st.markdown(f"{item['toolResult']['content'][0]['text']}")

        with tabs[1]:
            st.markdown("**Request Messages**")
            st.json(st.session_state.history)

    if st.button("Clear Chat History"):
        st.session_state.history = []
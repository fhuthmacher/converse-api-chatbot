# Amazon Bedrock Converse API example with tool usage
With function calling, we can provide LLMs with descriptions of tools and functions it can use. An LLM is able to intelligently decide based on user query when and how to use those tools to help answer questions and complete tasks.

This repository contains two examples (one streamlit app and one notebook) to get you started with Function Calling on Amazon Bedrock.

The Converse or ConverseStream API is a unified structured text API action that allows you simplifying the invocations to Bedrock LLMs, using a universal syntax and message structured prompts for any of the supported model providers, and adding support for tool use of function calling with a unified syntax. 

Function calling tool use with Converse API in Bedrock:
- Notebook
- Streamlit demo

## Environment
1. Ensure you've enabled Claude Sonnet and Claude Haiku in the Bedrock Console
2. Ensure you have adequate permissions to call Bedrock from the Python SDK (Boto3)

## Getting Started
1. Clone the repository:
   ```
   git clone https://github.com/fhuthmacher/converse-api-chatbot.git
   cd converse-api-chatbot
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install boto3 PyAthena python-dotenv pandas numpy streamlit pydantic --force-reinstall --no-cache
   ```

4. Update config.env - set region and s3 bucket name for Athena
   ```
   REGION=us-east-1
   S3_BUCKET_NAME=XXX
   ```
5. Create your Athena table and update query_database tool.

6. Explore the notebook / run the streamlit application.
   ```
   streamlit run streamlit-chat.py
   ```

For more information check the documentation and the [API examples here](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/function-calling).
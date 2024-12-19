# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import tiktoken
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
#from langchain.chat_models import ChatAnthropic
from langchain_experimental.agents import create_csv_agent
#from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool

# Token limit for the Claude model
token_limit = 100000  # Adjust based on the specific model capabilities

# Initialize the tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Claude may use GPT-3.5/4 tokenizer for now

# Initialize the Python REPL tool
python_repl_tool = PythonAstREPLTool()
# Define the tools list
agent_tools = [python_repl_tool]

def load_llm():
 llm = CTransformers(
  model='llama-2-7b-chat.ggmlv3.q8_0.bin',
  model_type='llama',
  config={'max_new_tokens': 10000,
             'temperature': 0.4,
             'context_length': 10000}
  )
 return llm

from langchain_community.llms import OpenLLM

#gpt2_llm = CTransformers(model="marella/gpt-2-ggml")

def calculate_token_count(text):
    """Calculate the number of tokens in the given text."""
    tokens = tokenizer.encode(text)
    return len(tokens)


def truncate_csv_to_token_limit(csv_path, token_limit):
    """Truncate CSV data to fit within the token limit."""
    df = pd.read_csv(csv_path)
    data_str = df.to_csv(index=False)
    token_count = calculate_token_count(data_str)

    if token_count <= token_limit:
        return data_str

    # Truncate rows to stay within the token limit
    truncated_df = pd.DataFrame()
    for _, row in df.iterrows():
        new_df = pd.concat([truncated_df, pd.DataFrame([row])])
        if calculate_token_count(new_df.to_csv(index=False)) > token_limit:
            break
        truncated_df = new_df

    return truncated_df.to_csv(index=False)


# Truncate the CSV content to fit within the token limit
csv_file_path = "test_data.csv"
csv_content = truncate_csv_to_token_limit(csv_file_path, token_limit)

# Load environment variables for API key
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize ChatAnthropic
chat_llm = ChatAnthropic(model="claude-v1", anthropic_api_key=anthropic_api_key, verbose=True)

# Add a tool that extracts information in a structured format
from langchain.tools import Tool

def dataframe_summary(df):
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }

data_summary_tool = Tool(
    name="Data Summary",
    func=dataframe_summary,
    description="Provides summary information about the DataFrame."
)


llama_llm = load_llm()
# Create a CSV Agent
agent = create_csv_agent(llm=llama_llm, path=csv_file_path, tools=[PythonAstREPLTool()], verbose=True, allow_dangerous_code=True)

# Example user interaction
if __name__ == "__main__":
    print("CSV Agent is ready. Ask a question about your CSV data.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the agent. Goodbye!")
            break
        try:
            response = agent.invoke(user_input)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Error: {e}")


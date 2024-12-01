import os
import json
import logging
import requests
import difflib
import streamlit as st
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description
import urllib3

# Configure logging
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global variables for lazy initialization
llm = None
agent_executor = None

# NetBoxController for CRUD Operations
class NetBoxController:
    def __init__(self, netbox_url, api_token):
        self.netbox = netbox_url.rstrip('/')
        self.api_token = api_token
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f"Token {self.api_token}",
        }

    def get_api(self, api_url: str, params: dict = None):
        response = requests.get(
            f"{self.netbox}{api_url}",
            headers=self.headers,
            params=params,
            verify=False
        )
        response.raise_for_status()
        return response.json()

    def post_api(self, api_url: str, payload: dict):
        response = requests.post(
            f"{self.netbox}{api_url}",
            headers=self.headers,
            json=payload,
            verify=False
        )
        response.raise_for_status()
        return response.json()

    def delete_api(self, api_url: str):
        response = requests.delete(
            f"{self.netbox}{api_url}",
            headers=self.headers,
            verify=False
        )
        response.raise_for_status()
        return response.json()


# Function to load supported URLs with their names from a JSON file
def load_urls(file_path='netbox_apis.json'):
    if not os.path.exists(file_path):
        return {"error": f"URLs file '{file_path}' not found."}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [(entry['URL'], entry.get('Name', '')) for entry in data]
    except Exception as e:
        return {"error": f"Error loading URLs: {str(e)}"}


def check_url_support(api_url: str) -> dict:
    url_list = load_urls()
    if "error" in url_list:
        return url_list  # Return error if loading URLs failed

    urls = [entry[0] for entry in url_list]
    names = [entry[1] for entry in url_list]

    close_url_matches = difflib.get_close_matches(api_url, urls, n=1, cutoff=0.6)
    close_name_matches = difflib.get_close_matches(api_url, names, n=1, cutoff=0.6)

    if close_url_matches:
        closest_url = close_url_matches[0]
        matching_name = [entry[1] for entry in url_list if entry[0] == closest_url][0]
        return {"status": "supported", "closest_url": closest_url, "closest_name": matching_name}
    elif close_name_matches:
        closest_name = close_name_matches[0]
        closest_url = [entry[0] for entry in url_list if entry[1] == closest_name][0]
        return {"status": "supported", "closest_url": closest_url, "closest_name": closest_name}
    else:
        return {"status": "unsupported", "message": f"The input '{api_url}' is not supported."}


# Tools for interacting with NetBox
@tool
def discover_apis(dummy_input: str = None) -> dict:
    """Discover available NetBox APIs from a local JSON file."""
    try:
        if not os.path.exists("netbox_apis.json"):
            return {"error": "API JSON file not found. Please ensure 'netbox_apis.json' exists in the project directory."}
        
        with open("netbox_apis.json", "r") as f:
            data = json.load(f)
        return {"apis": data, "message": "APIs successfully loaded from JSON file"}
    except Exception as e:
        return {"error": f"An error occurred while loading the APIs: {str(e)}"}


@tool
def check_supported_url_tool(api_url: str) -> dict:
    """Check if an API URL or Name is supported by NetBox."""
    result = check_url_support(api_url)
    if result.get('status') == 'supported':
        closest_url = result['closest_url']
        closest_name = result['closest_name']
        return {
            "status": "supported",
            "message": f"The closest supported API URL is '{closest_url}' ({closest_name}).",
            "action": {
                "next_tool": "get_netbox_data_tool",
                "input": closest_url
            }
        }
    return result


@tool
def get_netbox_data_tool(api_url: str) -> dict:
    """Fetch data from NetBox."""
    try:
        netbox_controller = NetBoxController(
            netbox_url=os.getenv("NETBOX_URL"),
            api_token=os.getenv("NETBOX_TOKEN")
        )
        data = netbox_controller.get_api(api_url)
        return data
    except requests.HTTPError as e:
        return {"error": f"Failed to fetch data from NetBox: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@tool
def create_netbox_data_tool(input: str) -> dict:
    """Create new data in NetBox."""
    try:
        data = json.loads(input)
        api_url = data.get("api_url")
        payload = data.get("payload")

        if not api_url or not payload:
            raise ValueError("Both 'api_url' and 'payload' must be provided.")

        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary.")

        netbox_controller = NetBoxController(
            netbox_url=os.getenv("NETBOX_URL"),
            api_token=os.getenv("NETBOX_TOKEN")
        )
        return netbox_controller.post_api(api_url, payload)
    except Exception as e:
        return {"error": f"An error occurred in create_netbox_data_tool: {str(e)}"}


@tool
def delete_netbox_data_tool(api_url: str) -> dict:
    """Delete data from NetBox."""
    try:
        netbox_controller = NetBoxController(
            netbox_url=os.getenv("NETBOX_URL"),
            api_token=os.getenv("NETBOX_TOKEN")
        )
        return netbox_controller.delete_api(api_url)
    except requests.HTTPError as e:
        return {"error": f"Failed to delete data from NetBox: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def process_agent_response(response):
    if response and response.get("status") == "supported" and "next_tool" in response.get("action", {}):
        next_tool = response["action"]["next_tool"]
        tool_input = response["action"]["input"]

        # Automatically invoke the next tool
        return agent_executor.invoke({
            "input": tool_input,
            "chat_history": st.session_state.chat_history,
            "agent_scratchpad": "",
            "tool": next_tool
        })
    else:
        return response

# ============================================================
# Streamlit App
# ============================================================

def configure_page():
    st.title("NetBox Configuration")
    base_url = st.text_input("NetBox URL", placeholder="https://demo.netbox.dev")
    api_token = st.text_input("NetBox API Token", type="password", placeholder="Your API Token")

    if st.button("Save and Continue"):
        if not base_url or not api_token:
            st.error("All fields are required.")
        else:
            st.session_state['NETBOX_URL'] = base_url
            st.session_state['NETBOX_TOKEN'] = api_token
            os.environ['NETBOX_URL'] = base_url
            os.environ['NETBOX_TOKEN'] = api_token
            st.success("Configuration saved! Redirecting to chat...")
            st.session_state['page'] = "chat"

def initialize_agent():
    global llm, agent_executor
    if not llm:
        # Initialize the LLM with the API key from session state
        llm = Ollama(model="llama3.2", base_url="http://ollama:11434")

        # Define tools
        tools = [discover_apis, check_supported_url_tool, get_netbox_data_tool, create_netbox_data_tool, delete_netbox_data_tool]

        # Create the prompt template
        tool_descriptions = render_text_description(tools)
        # Create the PromptTemplate
        template = """
        Assistant is a network assistant capable of managing NetBox data using CRUD operations.

        TOOLS:
        - discover_apis: Discovers available NetBox APIs from a local JSON file.
        - check_supported_url_tool: Checks if an API URL or Name is supported by NetBox.
        - get_netbox_data_tool: Fetches data from NetBox using the specified API URL.
        - create_netbox_data_tool: Creates new data in NetBox using the specified API URL and payload.
        - delete_netbox_data_tool: Deletes data from NetBox using the specified API URL.

        GUIDELINES:
        1. Use 'check_supported_url_tool' to validate ambiguous or unknown URLs or Names.
        2. If certain about the URL, directly use 'get_netbox_data_tool', 'create_netbox_data_tool', or 'delete_netbox_data_tool'.
        3. Follow a structured response format to ensure consistency.

        FORMAT:
        Thought: [Your thought process]
        Action: [Tool Name]
        Action Input: [Tool Input]
        Observation: [Tool Response]
        Final Answer: [Your response to the user]

        Begin:

        Previous conversation history:
        {chat_history}

        New input: {input}

        {agent_scratchpad}
        """
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": tool_descriptions,
                "tool_names": ", ".join([t.name for t in tools])
            }
        )

        # Create the ReAct agent
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

        # Create the AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=10
        )

def chat_page():
    st.title("Chat with NetBox AI Agent")
    user_input = st.text_input("Ask NetBox a question:", key="user_input")

    initialize_agent()

    # Initialize session state variables if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Button to submit the question
    if st.button("Send"):
        if user_input:
            # Add the user input to the conversation history
            st.session_state.conversation.append({"role": "user", "content": user_input})

            # Invoke the agent with the user input and current chat history
            try:
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                    "agent_scratchpad": ""  # Initialize agent scratchpad as an empty string
                })

                # Process the agent's response
                final_response = process_agent_response(response)

                # Extract the final answer
                final_answer = final_response.get('output', 'No answer provided.')

                # Display the question and answer
                st.write(f"**Question:** {user_input}")
                st.write(f"**Answer:** {final_answer}")

                # Add the response to the conversation history
                st.session_state.conversation.append({"role": "assistant", "content": final_answer})

                # Update chat history with the new conversation
                st.session_state.chat_history = "\n".join(
                    [f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state.conversation]
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# Page Navigation
if 'page' not in st.session_state:
    st.session_state['page'] = "configure"

if st.session_state['page'] == "configure":
    configure_page()
elif st.session_state['page'] == "chat":
    chat_page()
import os
import json
import logging
import requests
import difflib
import streamlit as st
from langchain.tools import Tool  # Import Tool instead of using @tool decorator
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
        full_url = f"{self.netbox}/{api_url.lstrip('/')}"  # Fix URL construction
        response = requests.get(
            full_url,
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

def discover_apis():
    """
    Load and return the available NetBox APIs from the JSON file.
    """
    file_path = 'netbox_apis.json'
    
    if not os.path.exists(file_path):
        return {"error": f"API JSON file '{file_path}' not found."}
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return {"apis": data, "message": "APIs successfully loaded from JSON file."}
    except Exception as e:
        return {"error": f"Error loading APIs: {str(e)}"}

# Discover APIs Tool
discover_apis_tool = Tool(
    name="discover_apis",
    description="Discover available NetBox APIs from a local JSON file.",
    func=lambda _: discover_apis()
)

# Tool to check if a URL or name is valid
check_supported_url_tool = Tool(
    name="check_supported_url_tool",
    description="Check if an API URL or Name is supported by NetBox. Use this to find the correct API endpoint.",
    func=lambda query: check_url_support(query)
)

# Enhanced to ensure correct URL lookup before making API calls
get_netbox_data_tool = Tool(
    name="get_netbox_data_tool",
    description="Fetch data from NetBox using the correct API URL. Use 'check_supported_url_tool' first.",
    func=lambda api_url: fetch_with_lookup(api_url)
)

def fetch_with_lookup(api_url: str):
    """
    Check if the API URL is supported before making a request.
    """
    # Step 1: Lookup the correct API endpoint
    lookup_result = check_url_support(api_url)
    
    if lookup_result.get("status") == "supported":
        correct_url = lookup_result["closest_url"]
        try:
            netbox_controller = NetBoxController(
                netbox_url=os.getenv("NETBOX_URL"),
                api_token=os.getenv("NETBOX_TOKEN")
            )
            # Step 2: Fetch the data
            data = netbox_controller.get_api(correct_url)
            
            # Step 3: Count circuits if applicable
            if 'count' in data:
                return {"status": "success", "message": f"You have {data['count']} circuits in NetBox."}
            return {"status": "success", "message": "Data fetched successfully."}

        except Exception as e:
            return {"error": f"Failed to fetch data: {str(e)}"}
    
    return {"error": f"Unsupported API URL. Closest match: {lookup_result.get('closest_url')}"}

# ✅ Improved Create NetBox Data Tool
create_netbox_data_tool = Tool(
    name="create_netbox_data_tool",
    description="Create new data in NetBox. Requires 'api_url' and 'payload'.",
    func=lambda input_data: create_data_handler(input_data)
)

def create_data_handler(input_data):
    if not isinstance(input_data, dict):
        return {"error": "Invalid input. Expected a dictionary with 'api_url' and 'payload'."}

    api_url = input_data.get("api_url")
    payload = input_data.get("payload")

    if not api_url or not isinstance(payload, dict):
        return {"error": "Both 'api_url' and a valid 'payload' dictionary are required."}

    try:
        netbox_controller = NetBoxController(
            netbox_url=os.getenv("NETBOX_URL"),
            api_token=os.getenv("NETBOX_TOKEN")
        )
        response = netbox_controller.post_api(api_url, payload)
        return {
            "status": "success",
            "message": f"Resource created successfully at {api_url}.",
            "response": response
        }

    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        return {"error": f"Failed to create data: {str(e)}"}

# Delete NetBox Data Tool
delete_netbox_data_tool = Tool(
    name="delete_netbox_data_tool",
    description="Delete data from NetBox.",
    func=lambda api_url: NetBoxController(
        netbox_url=os.getenv("NETBOX_URL"),
        api_token=os.getenv("NETBOX_TOKEN")
    ).delete_api(api_url)
)

def process_agent_response(response):
    if not isinstance(response, dict):
        logging.error(f"Unexpected response format: {response}")
        return {"error": "Unexpected response format. Please check the input."}

    if response.get("status") == "success":
        return response

    if response.get("status") == "supported" and "next_tool" in response.get("action", {}):
        next_tool = response["action"]["next_tool"]
        tool_input = response["action"]["input"]

        return agent_executor.invoke({
            "input": tool_input,
            "chat_history": st.session_state.chat_history,
            "agent_scratchpad": "",
            "tool": next_tool
        })

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
        llm = Ollama(model="llama3.1", base_url="http://ollama:11434")

        # Define tools
        tools = [
           discover_apis_tool,
           check_supported_url_tool,
           get_netbox_data_tool,
           create_netbox_data_tool,
           delete_netbox_data_tool
        ]

        # Extract tool names for the prompt
        tool_names = ", ".join([tool.name for tool in tools])
        tool_descriptions = "\n".join([f"{tool.name}: {tool.description.split('.')[0]}" for tool in tools])
        prompt_template = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
            template="""
        You are a helpful network assistant that manages NetBox data with CRUD operations.

        **TASK FLOW:**

        1. **ALWAYS** check the correct API endpoint using 'discover_apis' or 'check_supported_url_tool'.
        2. **READ** data using 'get_netbox_data_tool'.
        3. **CREATE** data using 'create_netbox_data_tool'.
        4. If the data answers the question, **STOP** and provide the answer.  
        5. **DO NOT** perform additional actions once the task is complete.

        **IMPORTANT RULE:**  
        ⚠️ **NEVER** provide both a Final Answer **and** an Action. Choose one.

        **TOOLS:**  
        {tools}

        Available tool names: {tool_names}

        **FORMAT:**  
        Thought: [Your reasoning]  
        Action: [Tool Name]  
        Action Input: [Input to the Tool in JSON format]  
        Observation: [Result]  
        Final Answer: [Answer to the User]  

        Begin!

        Question: {input}  
        {agent_scratchpad}
        """
        )

        # Create the ReAct agent with the prompt
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt_template.partial(
                tools=tool_descriptions,
                tool_names=tool_names
            )
        )

        # Create the AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=100
        )

ollama_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'discover_apis',
            'description': 'Discover available NetBox APIs from a local JSON file.',
            'parameters': {'type': 'object', 'properties': {}}
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'check_supported_url_tool',
            'description': 'Check if an API URL or Name is supported by NetBox.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'api_url': {'type': 'string', 'description': 'API URL or Name to check'}
                },
                'required': ['api_url']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_netbox_data_tool',
            'description': 'Fetch data from NetBox using the specified API URL.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'api_url': {'type': 'string', 'description': 'API URL to fetch data from'}
                },
                'required': ['api_url']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'create_netbox_data_tool',
            'description': 'Create new data in NetBox.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'api_url': {'type': 'string', 'description': 'API URL to post data to'},
                    'payload': {'type': 'object', 'description': 'Payload to send to NetBox'}
                },
                'required': ['api_url', 'payload']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'delete_netbox_data_tool',
            'description': 'Delete data from NetBox.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'api_url': {'type': 'string', 'description': 'API URL to delete data from'}
                },
                'required': ['api_url']
            }
        }
    }
]

def chat_page():
    st.title("Chat with NetBox AI Agent")
    user_input = st.text_input("Ask NetBox a question:", key="user_input")

    initialize_agent()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Send"):
        if user_input:
            # Add user input to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                # ✅ Use agent_executor to process user input
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                    "agent_scratchpad": ""
                })

                # Extract and display the final answer
                final_answer = response.get('output', 'No answer provided.')
                st.write(f"**Answer:** {final_answer}")

                # Update chat history
                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Page Navigation
if 'page' not in st.session_state:
    st.session_state['page'] = "configure"

if st.session_state['page'] == "configure":
    configure_page()
elif st.session_state['page'] == "chat":
    chat_page()
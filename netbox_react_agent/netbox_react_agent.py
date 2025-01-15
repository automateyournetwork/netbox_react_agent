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
        full_url = f"{self.netbox}/{api_url.lstrip('/')}"
        response = requests.get(full_url, headers=self.headers, params=params, verify=False)
        response.raise_for_status()
        return response.json()

    def post_api(self, api_url: str, payload: dict):
        full_url = f"{self.netbox}{api_url}"
        logging.info(f"POST Request to URL: {full_url}")
        logging.info(f"Headers: {self.headers}")
        logging.info(f"Payload: {json.dumps(payload)}")
    
        try:
            response = requests.post(
                full_url,
                headers=self.headers,
                json=payload,
                verify=False
            )
            logging.info(f"Response Status Code: {response.status_code}")
            logging.info(f"Response Content: {response.text}")
    
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"POST request failed: {e}")
            return {"error": f"Request failed: {e}"}

    def delete_api(self, api_url: str):
        full_url = f"{self.netbox}{api_url}"
        logging.info(f"🗑️ DELETE Request to URL: {full_url}")
        logging.info(f"Headers: {self.headers}")

        try:
            response = requests.delete(
                full_url,
                headers=self.headers,
                verify=False
            )

            logging.info(f"📡 Response Status Code: {response.status_code}")
            logging.info(f"📦 Response Content: {response.text}")

            if response.status_code == 204:
                logging.info(f"✅ Deletion successful for {full_url}")
                return {"status": "success", "message": "Deletion successful."}
            else:
                logging.warning(f"⚠️ Deletion failed. Status code: {response.status_code}")
                return {"error": f"Failed to delete. Status code: {response.status_code}"}

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ DELETE request failed: {e}")
            return {"error": f"Request failed: {e}"}
        
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

    # Direct match should take priority
    if api_url in urls:
        return {"status": "supported", "closest_url": api_url}

    # Use difflib for approximate matches
    close_matches = difflib.get_close_matches(api_url, urls, n=1, cutoff=0.6)
    if close_matches:
        return {"status": "supported", "closest_url": close_matches[0]}

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

get_netbox_data_tool = Tool(
    name="get_netbox_data_tool",
    description="Fetch data from NetBox using the correct API URL.",
    func=lambda input_data: fetch_with_lookup(input_data.get("api_url"))  # No 'payload' needed for GET
)

def fetch_with_lookup(api_url: str):
    lookup_result = check_url_support(api_url)

    if lookup_result.get("status") == "supported":
        correct_url = lookup_result["closest_url"]
        try:
            netbox_controller = NetBoxController(
                netbox_url=os.getenv("NETBOX_URL"),
                api_token=os.getenv("NETBOX_TOKEN")
            )
            data = netbox_controller.get_api(correct_url)

            if isinstance(data, dict) and 'results' in data:
                return {"status": "success", "message": f"Found {len(data['results'])} items.", "data": data}

            return {"status": "success", "message": "Data fetched successfully.", "data": data}

        except Exception as e:
            return {"error": f"GET request failed: {str(e)}"}
    return {"error": f"Unsupported API URL. Closest match: {lookup_result.get('closest_url')}"}

# ✅ Improved Create NetBox Data Tool
create_netbox_data_tool = Tool(
    name="create_netbox_data_tool",
    description="Create new data in NetBox. Requires 'api_url' and 'payload'.",
    func=lambda input_data: create_data_handler(input_data)
)

def create_data_handler(input_data):
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input. Expected a JSON object."}

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
            "message": f"Successfully created resource at {api_url}.",
            "response": response
        }

    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        return {"error": f"POST request failed: {str(e)}"}

def delete_data_handler(input_data):
    logging.info("🚀 delete_data_handler was called.")
    logging.info(f"📝 Input Data: {json.dumps(input_data, indent=2)}")

    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError:
            logging.error("❌ Invalid JSON input provided to delete_data_handler.")
            return {"error": "Invalid JSON input. Expected a JSON object."}

    api_url = input_data.get("api_url")
    payload = input_data.get("payload", {})
    name = payload.get("name")

    if not api_url or not name:
        logging.error("❌ Missing 'api_url' or 'name' in payload.")
        return {"error": "Both 'api_url' and 'payload' with 'name' are required."}

    try:
        logging.info(f"🔍 Looking up provider '{name}' at {api_url}")

        netbox_controller = NetBoxController(
            netbox_url=os.getenv("NETBOX_URL"),
            api_token=os.getenv("NETBOX_TOKEN")
        )

        # Lookup entity by name to get its ID
        lookup_response = netbox_controller.get_api(api_url, params={'name': name})
        logging.info(f"📦 Lookup response: {json.dumps(lookup_response, indent=2)}")

        if lookup_response.get('count', 0) == 0:
            logging.warning(f"⚠️ No provider found with the name '{name}'.")
            return {"error": f"No resource found at '{api_url}' with name '{name}'."}

        entity_id = lookup_response['results'][0]['id']
        delete_url = f"{api_url.rstrip('/')}/{entity_id}/"

        logging.info(f"🗑️ Preparing to DELETE at {delete_url}")

        # Perform the deletion
        delete_response = netbox_controller.delete_api(delete_url)
        logging.info(f"📝 DELETE response: {delete_response}")

        if delete_response.get("status") == "success":
            return {
                "status": "success",
                "message": f"Successfully deleted '{name}' at {api_url}."
            }
        else:
            return {"error": delete_response.get("error", "Unknown error during deletion.")}

    except Exception as e:
        logging.error(f"❌ Error in delete_data_handler: {e}")
        return {"error": f"Error deleting data: {str(e)}"}
    
delete_netbox_data_tool = Tool(
    name="delete_netbox_data_tool",
    description="Delete data in NetBox. Requires 'api_url' and 'payload' with 'name'.",
    func=delete_data_handler
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
        llm = Ollama(model="command-r7b", base_url="http://ollama:11434")

        # ✅ Define the tools
        tools = [
            Tool(name="discover_apis", func=discover_apis, description="Discover available NetBox APIs."),
            Tool(name="check_supported_url_tool", func=check_url_support, description="Check if a NetBox API URL or Name is supported."),
            Tool(name="get_netbox_data_tool", func=fetch_with_lookup, description="Fetch data from NetBox using a valid API URL."),
            Tool(name="create_netbox_data_tool", func=create_data_handler, description="Create new data in NetBox with an API URL and payload."),
            Tool(name="delete_netbox_data_tool", func=delete_data_handler, description="Delete data in NetBox with an API URL and payload."),
        ]

        # Extract tool names and descriptions
        tool_names = ", ".join([tool.name for tool in tools])
        tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

        # ✅ Updated PromptTemplate
        prompt_template = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
            template="""
            You are a network assistant managing NetBox data using CRUD operations.

            **TOOLS:**  
            {tools}

            **Available Tool Names (use exactly as written):**  
            {tool_names}

            **FORMAT:**  
            Thought: [Your reasoning]  
            Action: [Tool Name]  
            Action Input: [Input to the Tool as JSON]  
            Observation: [Result]  
            Final Answer: [Answer to the User]  

            **Examples:**  
            - To fetch all circuits:  
              Thought: I need to retrieve all circuits from NetBox.  
              Action: get_netbox_data_tool  
              Action Input: {{ "api_url": "/api/circuits/" }}

            - To create a provider called "Bell Canada":  
              Thought: I need to create a provider named 'Bell Canada' with the slug 'bell'.  
              Action: create_netbox_data_tool  
              Action Input: {{ 
                "api_url": "/api/circuits/providers/", 
                "payload": {{ 
                  "name": "Bell Canada", 
                  "slug": "bell" 
                }} 
              }}


            - To delete a provider called "Bell Canada":  
              Thought: I need to create a provider named 'Bell Canada' with the slug 'bell'.  
              Action: delete_netbox_data_tool  
              Action Input: {{ 
                "api_url": "/api/circuits/providers/", 
                "payload": {{ 
                  "name": "Bell Canada",
                  "slug": "bell" 
                }} 
              }}

            **Begin!**

            Question: {input}  
            {agent_scratchpad}
            """
        )

        logging.info(f"🛠️ Registered tools: {[tool.name for tool in tools]}")

        # ✅ Pass 'tool_names' and 'tools' to the agent
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt_template.partial(
                tool_names=tool_names,
                tools=tool_descriptions
            )
        )

        # ✅ AgentExecutor with error handling
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=50
        )
        logging.info("🚀 AgentExecutor initialized with tools.")
        
def configure_page():
    st.title("NetBox Configuration")
    base_url = st.text_input("NetBox URL", placeholder="https://demo.netbox.dev")
    api_token = st.text_input("NetBox API Token", type="password", placeholder="Your API Token")

    if st.button("Save and Continue"):
        if not base_url or not api_token:
            st.error("All fields are required.")
        else:
            # Save the configuration to session state
            st.session_state['NETBOX_URL'] = base_url
            st.session_state['NETBOX_TOKEN'] = api_token

            # Save to environment variables
            os.environ['NETBOX_URL'] = base_url
            os.environ['NETBOX_TOKEN'] = api_token

            st.success("Configuration saved! Redirecting to chat...")
            st.session_state['page'] = "chat"

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
                logging.info(f"📝 User input: {user_input}")

                # ✅ Use agent_executor to process user input
                response = agent_executor.invoke({
                    "input": user_input,
                    "agent_scratchpad": ""
                })

                logging.info(f"🤖 Agent response: {response}")
                
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
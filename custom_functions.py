import json
import requests
import os
from groq import Groq
from dotenv import load_dotenv
import assistant_instructions as assistant_instructions

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")

# Init Groq Client
client = Groq(api_key=GROQ_API_KEY)


# Add lead to Airtable
def create_lead(name="", company_name="", phone="", email=""):
    url = "https://api.airtable.com/v0/appyrh5KOoEhrytFB/Leads"
    headers = {
        "Authorization": 'Bearer ' + AIRTABLE_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "records": [{
            "fields": {
                "Name": name,
                "Phone": phone,
                "Email": email,
                "CompanyName": company_name,
            }
        }]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("Lead created successfully.")
        return response.json()
    else:
        print(f"Failed to create lead: {response.text}")
        return {"error": f"Failed to create lead: {response.text}"}


# Create or load assistant configuration for Groq
def create_assistant(client):
    assistant_file_path = 'assistant.json'

    # Groq için assistant konfigürasyonu
    assistant_config = {
        "model": "llama-3.3-70b-versatile",
        "system_prompt": assistant_instructions.assistant_instructions,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "create_lead",
                    "description": "Capture lead details and save to Airtable.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the lead."
                            },
                            "phone": {
                                "type": "string",
                                "description": "Phone number of the lead."
                            },
                            "email": {
                                "type": "string",
                                "description": "Email of the lead."
                            },
                            "company_name": {
                                "type": "string",
                                "description": "CompanyName of the lead."
                            }
                        },
                        "required": ["name", "email", "company_name"]
                    }
                }
            }
        ],
        "knowledge_base": None  # Groq'da file_search yok, bu yüzden None
    }

    # If there is an assistant.json file already, then load that assistant
    if os.path.exists(assistant_file_path):
        with open(assistant_file_path, 'r') as file:
            assistant_data = json.load(file)
            print("Loaded existing assistant configuration.")
            return assistant_data
    else:
        # Knowledge base handling for Groq
        knowledge_content = ""
        if os.path.exists("knowledge.docx"):
            try:
                # docx dosyasını okumak için python-docx gerekli
                from docx import Document
                doc = Document("knowledge.docx")
                knowledge_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                print("Knowledge base loaded from knowledge.docx")
            except ImportError:
                print("python-docx library not found. Install with: pip install python-docx")
                print("Knowledge base not loaded.")
            except Exception as e:
                print(f"Error loading knowledge base: {e}")

        # System prompt'a knowledge base'i ekle
        if knowledge_content:
            enhanced_prompt = f"""{assistant_instructions.assistant_instructions}

KNOWLEDGE BASE:
{knowledge_content}

Use the above knowledge base to answer questions about Academy Club's services and activities. Always refer to this information when answering specific questions about the company."""
            assistant_config["system_prompt"] = enhanced_prompt
            assistant_config["knowledge_base"] = knowledge_content

        # Create a new assistant.json file to load on future runs
        with open(assistant_file_path, 'w', encoding='utf-8') as file:
            json.dump(assistant_config, file, ensure_ascii=False, indent=2)
            print("Created a new assistant configuration and saved it.")

    return assistant_config


# Groq ile chat completion yapma fonksiyonu
def get_groq_response(messages, tools=None, model="llama-3.3-70b-versatile"):
    """
    Groq API kullanarak chat completion yapar
    """
    try:
        if tools:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1500
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )
        return response
    except Exception as e:
        print(f"Error during Groq API call: {e}")
        return None


# Knowledge base'den bilgi çekme fonksiyonu (Groq için basit arama)
def search_knowledge_base(query, knowledge_content):
    """
    Basit keyword-based arama. Gelişmiş RAG için vector search kullanılabilir.
    """
    if not knowledge_content:
        return ""

    query_words = query.lower().split()
    lines = knowledge_content.split('\n')
    relevant_lines = []

    for line in lines:
        if any(word in line.lower() for word in query_words):
            relevant_lines.append(line)

    return '\n'.join(relevant_lines[:5])  # İlk 5 ilgili satırı döndür


# Assistant helper fonksiyonu
def process_message_with_assistant(message, conversation_history=None, assistant_config=None):
    """
    Mesajı assistant konfigürasyonu ile işler
    """
    if not assistant_config:
        assistant_config = create_assistant(client)

    if not conversation_history:
        conversation_history = []

    # System message'ı ekle
    messages = [{"role": "system", "content": assistant_config["system_prompt"]}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})

    # Knowledge base arama (basit)
    if assistant_config.get("knowledge_base"):
        relevant_info = search_knowledge_base(message, assistant_config["knowledge_base"])
        if relevant_info:
            enhanced_message = f"{message}\n\nRelevant information from knowledge base:\n{relevant_info}"
            messages[-1]["content"] = enhanced_message

    # Groq API çağrısı
    response = get_groq_response(messages, assistant_config["tools"])

    return response, assistant_config
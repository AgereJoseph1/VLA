import openai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for PDF text extraction
import tempfile
import os
import json
import time
import requests
from requests.exceptions import RequestException, Timeout
import backoff  # pip install backoff
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# FastAPI instance
app = FastAPI(
    title="Document Entity Extraction API",
    description="Extract specific entities from legal documents with customizable fields",
    version="1.0.0",
)

# Update CORS middleware for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Signatory(BaseModel):
    name: str
    title: str

class DateInfo(BaseModel):
    date: str
    purpose: str

class ExtractedEntities(BaseModel):
    document_type: str
    signatories: List[Signatory]
    dates: List[DateInfo]
    organizations: List[str]

# Functions for text extraction
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file using a file path"""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return text
    except Exception as e:
        print(f"PDF extraction error: {str(e)}")
        return f"Error extracting PDF text: {str(e)}"

def extract_text_from_image(file_path: str) -> str:
    """Extract text from image file using a file path"""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Image extraction error: {str(e)}")
        return f"Error extracting image text: {str(e)}"

# Normalize response to ensure consistent formatting and N/A values
def normalize_response(data):
    """
    Post-process API response to ensure consistent formatting:
    - Convert organizations to a simple list if it's an object
    - Replace null values with "N/A"
    """
    if isinstance(data, dict):
        # Special case for organizations when it's incorrectly structured as an object
        if "organizations" in data and isinstance(data["organizations"], dict) and "name" in data["organizations"]:
            # Convert object with name field to a simple list
            data["organizations"] = [data["organizations"]["name"]]
        
        # Process other fields
        for key, value in data.items():
            if value is None:
                data[key] = "N/A"
            else:
                data[key] = normalize_response(value)
        return data
    elif isinstance(data, list):
        return [normalize_response(item) if item is not None else "N/A" for item in data]
    elif data is None:
        return "N/A"
    return data

# Retry decorator for OpenAI API calls - Standard extraction with fixed fields
@backoff.on_exception(
    backoff.expo,
    (RequestException, openai.error.OpenAIError, openai.error.RateLimitError, openai.error.APIError),
    max_tries=3,
    max_time=30
)
def extract_legal_entities_with_retry(text: str):
    """Extract standard legal entities with retry logic for API failures"""
    try:
        prompt = f"""
        You are a legal document analyzer specialized in extracting key information. Please analyze the following legal document and extract specific entities with high precision.

        TASK:
        Extract the following details from the legal document, following these strict guidelines:

        1. **Document Type** [SINGLE VALUE]:
           - Identify the specific type of legal document (e.g., Employment Contract, Service Agreement, NDA)
           - Include any relevant qualifiers (e.g., Full-Time, Part-Time, Fixed-Term)

        2. **Signatories** [LIST OF OBJECTS]:
           - Extract names AND titles/positions of people who signed the document
           - Format each signatory as: {{"name": "Full Name", "title": "Position/Title"}}
           - Include ONLY official titles/positions mentioned in the signature block
           - Examples:
             * {{"name": "John Smith", "title": "Chief Executive Officer"}}
             * {{"name": "Jane Doe", "title": "Employee"}}

        3. **Dates** [LIST OF OBJECTS]:
           - Extract significant dates WITH their specific purposes
           - Format each date as: {{"date": "YYYY-MM-DD", "purpose": "Specific Purpose"}}
           - Include ONLY dates with clear purposes such as:
             * Contract Start Date
             * Contract End Date
             * Signing Date
             * Probation Period Start
             * Probation Period End
             * Offer Expiry Date
           - Example:
             * {{"date": "2024-01-01", "purpose": "Contract Start Date"}}
             * {{"date": "2024-12-31", "purpose": "Contract End Date"}}

        4. **Organizations** [LIST]:
           - Include ONLY formally involved organizations:
             * Employing company
             * Parent company
             * Subsidiary mentioned as party
             * Client organization (if relevant)
           - Maintain exact spelling and formatting
           - Include any legal entity identifiers if mentioned

        Here is the legal text to analyze:
        {text[:4000]}  # Limit text length to avoid token limits

        RESPONSE FORMAT:
        Return ONLY a JSON object with these exact keys:
        {{
            "document_type": "Specific Document Type",
            "signatories": [
                {{
                    "name": "Full Name 1",
                    "title": "Position/Title 1"
                }},
                {{
                    "name": "Full Name 2",
                    "title": "Position/Title 2"
                }}
            ],
            "dates": [
                {{
                    "date": "YYYY-MM-DD",
                    "purpose": "Specific Purpose (e.g., Contract Start Date)"
                }}
            ],
            "organizations": ["Organization Name 1", "Organization Name 2"]
        }}

        IMPORTANT:
        - Include ONLY entities that you are CERTAIN about
        - Maintain exact spelling and formatting from the document
        - Do NOT make assumptions or include ambiguous entries
        - If any category has no valid entries, return an empty list
        - Ensure all dates have clear purposes
        - Include titles/positions exactly as mentioned in the document
        - For any information not found, use "N/A" instead of leaving it blank or null
        """

        # Use a timeout to prevent hanging requests
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise legal document analyzer that extracts entities with high accuracy. You only extract information that is explicitly present in the document and return it in the exact specified JSON format. Always use 'N/A' for missing information, never null or undefined."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            request_timeout=30  # 30 second timeout
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        raise

# Retry decorator for OpenAI API calls - Custom extraction with user-specified fields
@backoff.on_exception(
    backoff.expo,
    (Exception, openai.error.OpenAIError, openai.error.RateLimitError, openai.error.APIError),
    max_tries=3,
    max_time=30
)
def extract_custom_entities(text: str, fields_to_extract: List[str]):
    """Extract user-specified entities from legal document text with N/A for missing values"""
    try:
        fields_instruction = ', '.join([f'"{field}"' for field in fields_to_extract])
        
        # Special handling instructions for specific fields
        special_instructions = """
        IMPORTANT:
        - For all fields, if the information is not found, use "N/A" for string values.
        - For example, if a signatory's title is unknown, return {"name": "John Smith", "title": "N/A"}
        - Never return null values - use "N/A" instead for any information that cannot be found in the document
        - Do not add additional structure or fields that weren't requested
        - Keep the response minimal and only extract exactly what is found in the document
        """
        
        if "dates" in fields_to_extract:
            special_instructions += """
            - For "dates": Extract as objects with both the date and its purpose/context
              Format dates as: [{"date": "YYYY-MM-DD or original format", "purpose": "What this date represents"}]
              IMPORTANT: Only include dates where you are CERTAIN about both the date AND its purpose
              DO NOT include any date where either the date itself or its purpose is uncertain
            """
        
        if "signatories" in fields_to_extract:
            special_instructions += """
            - For "signatories": Include name and title/role if available
              Format signatories as: [{"name": "Person Name", "title": "Role/Title"}]
              If title is unknown, use "N/A" for the title field
            """
            
        if "organizations" in fields_to_extract:
            special_instructions += """
            - For "organizations": Return a simple list of organization names as strings
              Format organizations as: ["Organization Name 1", "Organization Name 2"]
              Do not add additional fields or structure to organizations
            """
            
        prompt = f"""
        You are a document analyzer. Extract ONLY the following fields from this document: {fields_instruction}.
        
        For each field:
        - Extract exactly what is found in the document
        - Do not add any explanations or notes
        - Use "N/A" for any information that cannot be found
        
        {special_instructions}
        
        Here is the document text:
        {text[:4000]}
        
        Return ONLY a JSON object with these fields: {fields_instruction}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using 3.5-turbo for custom extraction to save costs
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise document analyzer that extracts only the requested fields and returns them in JSON format. Always use 'N/A' for missing information, never null or undefined. Keep your extraction minimal and don't add additional structure or fields that weren't explicitly requested."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            request_timeout=30
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        raise

def format_results_as_text(data: dict) -> str:
    """Convert extracted data into a human-readable text format"""
    output = []
    
    # Handle document type if present
    if "document_type" in data:
        output.append(f"Document Type: {data['document_type']}")
    
    # Handle signatories
    if "signatories" in data:
        output.append("\nSignatories:")
        if not data["signatories"]:
            output.append("- None found")
        else:
            for sig in data["signatories"]:
                output.append(f"- {sig['name']} ({sig['title']})")
    
    # Handle dates
    if "dates" in data:
        output.append("\nImportant Dates:")
        if not data["dates"]:
            output.append("- None found")
        else:
            for date_info in data["dates"]:
                output.append(f"- {date_info['purpose']}: {date_info['date']}")
    
    # Handle organizations
    if "organizations" in data:
        output.append("\nOrganizations:")
        if not data["organizations"]:
            output.append("- None found")
        else:
            for org in data["organizations"]:
                output.append(f"- {org}")
    
    # Handle any custom fields
    for key, value in data.items():
        if key not in ["document_type", "signatories", "dates", "organizations"]:
            output.append(f"\n{key.title()}:")
            if isinstance(value, list):
                if not value:
                    output.append("- None found")
                else:
                    for item in value:
                        output.append(f"- {item}")
            else:
                output.append(f"- {value}")
    
    return "\n".join(output)

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post(
    "/upload_legal_document",
    description="""
    Extract specific fields from a document.
    
    - Upload a document (PDF, JPG, JPEG, PNG)
    - Specify which fields to extract as a comma-separated list
    - The API will extract only those fields and return N/A for missing information
    
    Example fields: signatories, dates, organizations, contract_value, payment_terms, etc.
    """
)
async def upload_legal_document(
    file: UploadFile = File(...),
    fields: str = Form(..., description="Comma-separated list of fields to extract")
):
    """Extract specific entities from a legal documents"""
    try:
        # Parse the fields to extract
        fields_to_extract = [field.strip() for field in fields.split(',')]
        
        if not fields_to_extract:
            return JSONResponse(
                status_code=400,
                content={"error": "No fields specified for extraction"}
            )
            
        # Get the file extension
        file_extension = file.filename.split('.')[-1].lower()
        file_content = await file.read()

        # Validate file size (10MB limit)
        if len(file_content) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"error": "File too large. Maximum size is 10MB."}
            )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        try:
            # Process file based on its type
            if file_extension == "pdf":
                text = extract_text_from_pdf(temp_path)
            elif file_extension in ["jpg", "jpeg", "png"]:
                text = extract_text_from_image(temp_path)
            else:
                # Clean up temp file
                os.unlink(temp_path)
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file format: {file_extension}. Please upload a PDF or image file (JPG, JPEG, PNG)."}
                )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Check if text was extracted successfully
            if not text or text.startswith("Error extracting"):
                return JSONResponse(
                    status_code=500,
                    content={"error": "Failed to extract text from the document. Please ensure the file is not corrupted."}
                )
            
            # Extract the specified fields with N/A handling
            try:
                extracted_json = extract_custom_entities(text, fields_to_extract)
                
                try:
                    response_data = json.loads(extracted_json)
                    response_data = normalize_response(response_data)
                    formatted_text = format_results_as_text(response_data)
                    # Keep the Results wrapper
                    return {"Results": formatted_text}
                    
                except json.JSONDecodeError:
                    # Try to extract just the JSON part from the text
                    import re
                    json_match = re.search(r'({[\s\S]*})', extracted_json)
                    if json_match:
                        try:
                            response_data = json.loads(json_match.group(1))
                            response_data = normalize_response(response_data)
                            formatted_text = format_results_as_text(response_data)
                            # Keep the Results wrapper
                            return {"Results": formatted_text}
                        except:
                            pass
                    
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Failed to parse the AI response into valid JSON. Please try again."}
                    )
            
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"AI processing error: {str(e)}"}
                )

        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return JSONResponse(
                status_code=500,
                content={"error": f"Document processing error: {str(e)}"}
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Upload error: {str(e)}"}
        )


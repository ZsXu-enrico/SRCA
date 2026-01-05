"""Prompt templates for SRCA semantic representation.

RPM (Representation Prompt for Mashups): Requirement-focused reconstruction
FPA (Feature Prompt for APIs): Functional-oriented reconstruction
"""

# RPM: Requirement-focused Prompt for Mashups
RPM_TEMPLATE = """# Task
You are a technical documentation expert. Your task is to reconstruct the given mashup description into a unified format that highlights key requirements. Please analyze the description and reorganize it into the following sections:

Functional Requirements: What are the main functions this mashup needs?
Target Scenario: In what scenarios would this mashup be used?
Target Users: Who are the intended users?
Expected Outcome: What results or benefits are expected?

Requirements for the response:
- Use clear, formal, and consistent language
- Focus on essential information only
- Keep each section concise but informative
- Maintain technical accuracy
- Exclude implementation details or technical specifications

# Query
Mashup Name: {name}
Original Description: {description}
Categories: {categories}

Please provide the reconstruction:"""

# FPA: Functional-oriented Prompt for APIs
FPA_TEMPLATE = """# Task
You are a technical documentation expert. Your task is to reconstruct the given API description into a unified format that emphasizes its key functional features. Please analyze the description and reorganize it into the following sections:

Core Functionality: What are the primary functions this API provides?
Key Features: What are the distinctive technical features or capabilities?
Use Cases: What are the main application scenarios?
Primary Benefits: What advantages does this API offer?

Requirements for the response:
- Use clear, formal, and consistent language
- Focus on functional aspects only
- Keep each section concise but informative
- Maintain technical accuracy
- Exclude non-functional information (e.g., version history, release dates)

# Query
API Name: {name}
Original Description: {description}
Categories: {categories}

Please provide the reconstruction:"""


def format_rpm_prompt(description: str, categories: list, name: str = "Unknown") -> str:
    """
    Format RPM prompt for mashup description unification.

    Args:
        description: Original mashup description
        categories: List of category names
        name: Mashup name (optional)

    Returns:
        Formatted prompt string
    """
    categories_str = ", ".join(categories) if categories else "Not specified"
    return RPM_TEMPLATE.format(
        name=name,
        description=description.strip(),
        categories=categories_str
    )


def format_fpa_prompt(name: str, description: str, categories: list) -> str:
    """
    Format FPA prompt for API feature extraction.

    Args:
        name: API name
        description: API description
        categories: List of category names

    Returns:
        Formatted prompt string
    """
    categories_str = ", ".join(categories) if categories else "Not specified"
    return FPA_TEMPLATE.format(
        name=name.strip(),
        description=description.strip(),
        categories=categories_str
    )


# System prompt for LLaMA-3 chat format
SYSTEM_PROMPT = """You are a helpful assistant that specializes in understanding and describing web services and APIs."""


def format_chat_prompt(user_message: str) -> str:
    """
    Format prompt in LLaMA-3 chat template format.

    Args:
        user_message: The user's prompt

    Returns:
        Formatted chat prompt for LLaMA-3
    """
    # LLaMA-3 uses a specific chat format with special tokens
    # Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

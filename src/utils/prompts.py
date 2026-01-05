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

# Example
Mashup Name: Padvark.nl
Original Description: Padvark.nl is a web application that provides users with information on different running and cycling events in the Netherlands, allowing them to track routes and compare with other users, by entering the location of their choice.
Categories: sports, mapping, social

Output:
Functional Requirements: Location-based event search functionality, route tracking and visualization, performance comparison features
Target Scenario: Finding local sports events, planning running or cycling routes, comparing athletic performance
Target Users: Runners and cyclists seeking organized events and route planning tools
Expected Outcome: Easy discovery of sports events with integrated route tracking and social comparison capabilities

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

# Example
API Name: Google Maps
Original Description: The Google Maps API allow for the embedding of Google Maps onto web pages of outside developers, using a simple JavaScript interface or Flash interface. It is designed to work on mobile devices and desktop browsers.
Categories: mapping, location

Output:
Core Functionality: Provides comprehensive mapping and location-based services through embeddable map interfaces
Key Features: Cross-platform compatibility (mobile and desktop), simple JavaScript and Flash interfaces, interactive map visualization
Use Cases: Embedding maps in web applications, location visualization, route display, geographic data presentation
Primary Benefits: Easy integration with web applications, reliable global mapping data, responsive across devices

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

"""
Security and compliance module for Enterprise PDF AI Assistant.
Implements guardrails, credential redaction, and data sanitization.
"""

import re


def sanitize_context(context):
    """
    Sanitizes retrieved document context by redacting sensitive patterns such as
    passwords, API keys, secrets, and authorization tokens.
    
    Args:
        context (str): Raw text context retrieved from vector store.
        
    Returns:
        str: Sanitized text with sensitive data masked as [REDACTED].
    """
    patterns = [
        r"password\s*[:=]\s*\S+",
        r"api\s*key\s*[:=]\s*\S+",
        r"secret\s*[:=]\s*\S+",
        r"token\s*[:=]\s*\S+"
    ]

    for pattern in patterns:
        context = re.sub(pattern, "[REDACTED]", context, flags=re.IGNORECASE)

    return context

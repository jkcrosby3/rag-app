"""
Notification handler for document changes in the RAG system.
"""
import logging
from typing import Callable, Any
import json

logger = logging.getLogger(__name__)

class DocumentNotificationHandler:
    """Handles notifications about document changes."""
    
    def __init__(self):
        """Initialize the notification handler."""
        self._handlers = {}
        
    def register_handler(self, event_type: str, handler: Callable[[str], None]) -> None:
        """
        Register a notification handler for a specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Function that takes a message as argument
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def notify(self, event_type: str, message: str) -> None:
        """
        Notify handlers about an event.
        
        Args:
            event_type: Type of event
            message: Event message
        """
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error in notification handler: {str(e)}")
    
    def create_log_handler(self) -> Callable[[str, str], None]:
        """
        Create a handler that logs notifications.
        
        Returns:
            A notification handler function
        """
        def log_handler(event_type: str, message: str) -> None:
            logger.info(f"[NOTIFICATION] {event_type}: {message}")
        return log_handler
    
    def create_webhook_handler(self, webhook_url: str) -> Callable[[str, str], None]:
        """
        Create a handler that sends notifications to a webhook.
        
        Args:
            webhook_url: URL to send notifications to
            
        Returns:
            A notification handler function
        """
        import requests
        
        def webhook_handler(event_type: str, message: str) -> None:
            try:
                payload = {
                    "event_type": event_type,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                }
                response = requests.post(webhook_url, json=payload, timeout=5)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Error sending webhook notification: {str(e)}")
        
        return webhook_handler
    
    def create_email_handler(self, to_address: str) -> Callable[[str, str], None]:
        """
        Create a handler that sends email notifications.
        
        Args:
            to_address: Email address to send notifications to
            
        Returns:
            A notification handler function
        """
        import smtplib
        from email.message import EmailMessage
        
        def email_handler(event_type: str, message: str) -> None:
            try:
                msg = EmailMessage()
                msg.set_content(f"Document Notification\n\nType: {event_type}\nMessage: {message}")
                msg["Subject"] = f"Document Notification - {event_type}"
                msg["From"] = "document-system@example.com"
                msg["To"] = to_address
                
                # TODO: Configure SMTP server
                with smtplib.SMTP('localhost') as s:
                    s.send_message(msg)
            except Exception as e:
                logger.error(f"Error sending email notification: {str(e)}")
        
        return email_handler

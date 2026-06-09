# path: src/juniorclimbs/notifications.py

"""
JuniorClimbs Notifications (Stub)

Configurable notification channel for employees (email / slack / telegram).
Currently prints for beta. Ready for real integration.
"""

from typing import Optional


def send_notification(employee_name: str, message: str, channel: str = "email"):
    """
    Send notification to employee.
    In production this would call email/Slack/Telegram APIs based on preference.
    """
    print(f"[NOTIFICATION][{channel.upper()}] To: {employee_name}")
    print(f"  Message: {message}")
    # TODO: Real integration based on employee.notification_preference
    return True

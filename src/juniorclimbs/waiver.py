# path: src/juniorclimbs/waiver.py

"""
JuniorClimbs Digital Liability Waiver System

Production-grade waiver flow:
- Rigid QR code opens clean mobile browser form
- Multiple choice Yes/No (all must be Yes for safety compliance)
- Creates or updates member profile with full history and provenance
- Designed for real gym liability protection
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from .models import Member, Waiver, MembershipStatus
from .member_manager import MemberManager


# Standard gym safety waiver questions (all must be answered Yes)
DEFAULT_WAIVER_QUESTIONS = [
    "I have read and understand the gym rules and safety guidelines.",
    "I am physically capable of participating in climbing activities.",
    "I understand the risks involved in climbing and bouldering.",
    "I will follow all staff instructions and posted safety rules.",
    "I will not climb under the influence of alcohol or drugs.",
    "I accept full responsibility for my actions and any injuries sustained.",
    "I consent to the gym using my information for membership and safety records.",
]


class WaiverSession:
    def __init__(self, member_id: Optional[str] = None, questions: Optional[List[str]] = None):
        self.session_id = str(uuid.uuid4())
        self.member_id = member_id
        self.questions = questions or DEFAULT_WAIVER_QUESTIONS
        self.answers: Dict[int, bool] = {}  # question_index -> answer
        self.created_at = datetime.utcnow()
        self.completed = False

    def submit_answer(self, question_index: int, answer: bool) -> bool:
        if 0 <= question_index < len(self.questions):
            self.answers[question_index] = answer
            return True
        return False

    def is_complete_and_valid(self) -> bool:
        if len(self.answers) != len(self.questions):
            return False
        return all(self.answers.values())  # All must be True (Yes)

    def get_missing_questions(self) -> List[int]:
        return [i for i in range(len(self.questions)) if i not in self.answers]


class WaiverManager:
    def __init__(self, member_manager: MemberManager):
        self.member_manager = member_manager
        self.active_sessions: Dict[str, WaiverSession] = {}  # session_id -> session

    def create_waiver_session(self, member_id: Optional[str] = None, questions: Optional[List[str]] = None) -> WaiverSession:
        session = WaiverSession(member_id=member_id, questions=questions)
        self.active_sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[WaiverSession]:
        return self.active_sessions.get(session_id)

    def process_waiver_submission(
        self,
        session_id: str,
        answers: Dict[int, bool],
        ip_address: Optional[str] = None,
        signature_data: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return None

        # Apply answers
        for idx, answer in answers.items():
            session.submit_answer(idx, answer)

        if not session.is_complete_and_valid():
            return {
                "success": False,
                "error": "All questions must be answered Yes",
                "missing": session.get_missing_questions(),
            }

        # Create or get member
        member = None
        if session.member_id:
            member = self.member_manager.get_member(session.member_id)

        if not member:
            # Create new member profile from waiver flow
            member = self.member_manager.create_member(
                full_name="New Member (via Waiver)",  # Can be updated later
                email=None,
            )

        # Create waiver record with provenance
        waiver = self.member_manager.sign_waiver(
            member_id=member.id,
            signature_data=signature_data,
            ip_address=ip_address,
        )

        if waiver:
            waiver.provenance = {
                "session_id": session.session_id,
                "ip_address": ip_address,
                "submitted_via": "mobile_qr",
                "all_answers_yes": True,
            }

        # Mark session complete
        session.completed = True

        return {
            "success": True,
            "member_id": member.id,
            "waiver_id": waiver.id if waiver else None,
            "message": "Waiver completed successfully. Profile created/updated.",
        }

    def generate_qr_payload(self, member_id: Optional[str] = None) -> Dict[str, str]:
        """Generate data for a rigid QR code that opens the mobile waiver form."""
        session = self.create_waiver_session(member_id=member_id)
        return {
            "session_id": session.session_id,
            "url": f"/waiver/form/{session.session_id}",  # Frontend would handle this
            "instructions": "Scan to complete liability waiver on your phone",
        }

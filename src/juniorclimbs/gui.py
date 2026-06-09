# path: src/juniorclimbs/gui.py

"""
JuniorClimbs Desktop GUI - Local Software Suite (Production Beta)

Point-and-click desktop application for regular gym staff.
Only the member portal is web-based (can integrate with existing gym site).
"""

import customtkinter as ctk
from tkinter import messagebox
from datetime import datetime, timedelta

from .member_manager import MemberManager
from .ledger import Ledger
from .pos import POS
from .safety import SafetyManager
from .waiver import WaiverManager
from .employee import ScheduleManager
from .events import EventManager

from .models import TransactionType, PaymentMethod, MembershipStatus

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class JuniorClimbsApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("JuniorClimbs - Gym Operations")
        self.geometry("1100x750")

        self.mm = MemberManager()
        self.ledger = Ledger()
        self.pos = POS(self.mm, self.ledger)
        self.safety = SafetyManager()
        self.wm = WaiverManager(self.mm)
        self.sm = ScheduleManager()
        self.em = EventManager()

        self.current_staff_id = None  # For "My Schedule" feature

        self._create_widgets()

    def _create_widgets(self):
        # Sidebar Navigation
        self.sidebar = ctk.CTkFrame(self, width=220)
        self.sidebar.pack(side="left", fill="y", padx=8, pady=8)

        ctk.CTkLabel(self.sidebar, text="JuniorClimbs", font=("Arial", 22, "bold")).pack(pady=25)

        nav_items = [
            ("Check-in", self.show_checkin),
            ("Point of Sale", self.show_pos),
            ("Safety Zones", self.show_safety),
            ("Members", self.show_members),
            ("My Schedule", self.show_my_schedule),
            ("Events", self.show_events),
        ]

        for text, command in nav_items:
            btn = ctk.CTkButton(self.sidebar, text=text, command=command, width=200, height=40)
            btn.pack(pady=6)

        ctk.CTkLabel(self.sidebar, text="Staff Tools", font=("Arial", 12)).pack(pady=(30, 5))

        # Main content area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self.show_checkin()

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_status(self, message: str, success: bool = True):
        color = "#28a745" if success else "#dc3545"
        label = ctk.CTkLabel(self.main_frame, text=message, text_color=color, font=("Arial", 13))
        label.pack(pady=10)
        self.after(3000, label.destroy)

    # ==================== CHECK-IN ====================
    def show_checkin(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Member Check-in", font=("Arial", 20, "bold")).pack(pady=15)

        self.checkin_search = ctk.CTkEntry(self.main_frame, placeholder_text="Search by ID or Name", width=350, height=40)
        self.checkin_search.pack(pady=10)

        search_btn = ctk.CTkButton(self.main_frame, text="Search Member", command=self._search_member_for_checkin, width=200)
        search_btn.pack(pady=5)

        self.checkin_result_frame = ctk.CTkFrame(self.main_frame)
        self.checkin_result_frame.pack(pady=15, fill="x", padx=50)

        checkin_btn = ctk.CTkButton(self.main_frame, text="Check In", command=self._do_checkin, width=220, height=45, fg_color="#28a745")
        checkin_btn.pack(pady=20)

    def _search_member_for_checkin(self):
        for widget in self.checkin_result_frame.winfo_children():
            widget.destroy()

        query = self.checkin_search.get().strip().lower()
        if not query:
            return

        found = None
        for m in self.mm.members.values():
            if query in m.id.lower() or query in m.full_name.lower():
                found = m
                break

        if found:
            self.current_checkin_member = found
            info = f"{found.full_name} | Balance: ${found.current_balance} | Status: {found.status}"
            ctk.CTkLabel(self.checkin_result_frame, text=info, font=("Arial", 14)).pack()

            if found.status != MembershipStatus.ACTIVE or not self.wm.waivers:
                ctk.CTkLabel(self.checkin_result_frame, text="⚠ New member - Waiver required", text_color="#dc3545").pack()
        else:
            ctk.CTkLabel(self.checkin_result_frame, text="Member not found", text_color="#dc3545").pack()

    def _do_checkin(self):
        if not hasattr(self, 'current_checkin_member'):
            messagebox.showwarning("No Member", "Please search for a member first.")
            return

        member = self.current_checkin_member

        # Simple waiver enforcement for demo
        has_waiver = any(w.member_id == member.id for w in self.wm.waivers.values())
        if not has_waiver:
            messagebox.showwarning("Waiver Required", "This member must complete the liability waiver first.")
            self.show_waiver_flow(member)
            return

        self.mm.check_in(member.id)
        self.show_status(f"Checked in: {member.full_name}", success=True)

    # ==================== WAIVER FLOW ====================
    def show_waiver_flow(self, member):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Complete Liability Waiver", font=("Arial", 18, "bold")).pack(pady=15)

        ctk.CTkLabel(self.main_frame, text=f"For: {member.full_name}", font=("Arial", 14)).pack()

        self.waiver_answers = {}
        questions = self.wm.DEFAULT_WAIVER_QUESTIONS if hasattr(self.wm, 'DEFAULT_WAIVER_QUESTIONS') else [
            "I have read and understand the gym safety rules.",
            "I am physically capable of climbing activities.",
            "I accept responsibility for my own safety."
        ]

        for i, q in enumerate(questions):
            frame = ctk.CTkFrame(self.main_frame)
            frame.pack(fill="x", pady=4, padx=30)
            ctk.CTkLabel(frame, text=q, wraplength=500).pack(side="left", padx=10)

            var = ctk.StringVar(value="No")
            self.waiver_answers[i] = var
            ctk.CTkOptionMenu(frame, values=["Yes", "No"], variable=var, width=80).pack(side="right", padx=10)

        submit_btn = ctk.CTkButton(self.main_frame, text="Submit Waiver (All Yes Required)", command=lambda: self._submit_waiver(member), width=280, fg_color="#28a745")
        submit_btn.pack(pady=20)

    def _submit_waiver(self, member):
        answers = {i: v.get() == "Yes" for i, v in self.waiver_answers.items()}
        if not all(answers.values()):
            messagebox.showerror("Invalid", "All answers must be Yes to complete the waiver.")
            return

        self.wm.process_waiver_submission("gui", answers, ip_address="local")
        member.status = MembershipStatus.ACTIVE
        self.show_status("Waiver completed successfully. Member activated.", success=True)
        self.show_checkin()

    # ==================== POS ====================
    def show_pos(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Point of Sale", font=("Arial", 20, "bold")).pack(pady=15)

        self.pos_member_entry = ctk.CTkEntry(self.main_frame, placeholder_text="Member ID", width=300)
        self.pos_member_entry.pack(pady=8)

        ctk.CTkLabel(self.main_frame, text="Quick Items").pack()
        quick_frame = ctk.CTkFrame(self.main_frame)
        quick_frame.pack(pady=10)

        presets = [("Day Pass - $25", 25, TransactionType.DAY_PASS),
                   ("Merch - $15", 15, TransactionType.MERCH),
                   ("Food/Drink - $8", 8, TransactionType.FOOD_DRINK)]

        for label, amount, tx_type in presets:
            btn = ctk.CTkButton(quick_frame, text=label, command=lambda a=amount, t=tx_type: self._quick_sale(a, t), width=160)
            btn.pack(side="left", padx=8)

        self.pos_amount = ctk.CTkEntry(self.main_frame, placeholder_text="Custom Amount", width=200)
        self.pos_amount.pack(pady=10)

        sale_btn = ctk.CTkButton(self.main_frame, text="Complete Custom Sale", command=self._do_custom_pos, width=220)
        sale_btn.pack(pady=10)

    def _quick_sale(self, amount, tx_type):
        member_id = self.pos_member_entry.get().strip()
        if not member_id:
            messagebox.showwarning("Missing", "Enter Member ID first.")
            return
        self.pos.sell_to_member(member_id, tx_type, amount, PaymentMethod.CASH)
        self.show_status(f"Sale of ${amount} completed.", success=True)

    def _do_custom_pos(self):
        member_id = self.pos_member_entry.get().strip()
        try:
            amount = float(self.pos_amount.get())
        except:
            messagebox.showerror("Error", "Invalid amount")
            return
        self.pos.sell_to_member(member_id, TransactionType.OTHER, amount, PaymentMethod.CASH)
        self.show_status(f"Custom sale of ${amount} completed.", success=True)

    # ==================== SAFETY ====================
    def show_safety(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Safety Zones", font=("Arial", 20, "bold")).pack(pady=15)

        for area in self.safety.get_all_areas():
            frame = ctk.CTkFrame(self.main_frame)
            frame.pack(fill="x", pady=6, padx=20)

            ctk.CTkLabel(frame, text=f"{area.name} ({area.area_type}) - {area.status}").pack(side="left", padx=10)

            for status in ["open", "restricted", "closed"]:
                btn = ctk.CTkButton(frame, text=status.title(), width=90,
                                  command=lambda a=area.id, s=status: self._update_area(a, s))
                btn.pack(side="right", padx=4)

    def _update_area(self, area_id, status):
        self.safety.update_area_status(area_id, status, override=True)
        self.show_safety()

    # ==================== MEMBERS ====================
    def show_members(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Active Members", font=("Arial", 20, "bold")).pack(pady=15)

        for m in self.mm.get_active_members():
            exp = m.membership_expires.strftime("%Y-%m-%d") if m.membership_expires else "N/A"
            text = f"{m.full_name} | Bal: ${m.current_balance} | Expires: {exp}"
            ctk.CTkLabel(self.main_frame, text=text).pack(anchor="w", pady=3, padx=30)

            if m.membership_expires and m.membership_expires < datetime.utcnow() + timedelta(days=14):
                ctk.CTkLabel(self.main_frame, text="⚠ Expiring soon", text_color="#ffc107").pack(anchor="w", padx=40)

    # ==================== MY SCHEDULE ====================
    def show_my_schedule(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="My Schedule (Next 14 Days)", font=("Arial", 20, "bold")).pack(pady=15)

        if not self.current_staff_id:
            ctk.CTkLabel(self.main_frame, text="(Staff ID not set - demo mode)").pack()
            upcoming = self.sm.get_upcoming_shifts(14)[:5]
        else:
            upcoming = self.sm.get_employee_shifts(self.current_staff_id, datetime.utcnow(), datetime.utcnow() + timedelta(days=14))

        for shift in upcoming:
            emp = self.sm.employees.get(shift.employee_id)
            name = emp.full_name if emp else "You"
            text = f"{name} | {shift.start_time.strftime('%b %d %H:%M')} → {shift.end_time.strftime('%H:%M')} | {shift.get_break_info()}"
            ctk.CTkLabel(self.main_frame, text=text).pack(anchor="w", pady=4, padx=30)

    # ==================== EVENTS ====================
    def show_events(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Events & Sponsorships", font=("Arial", 20, "bold")).pack(pady=15)

        for event in self.em.get_upcoming_events(30):
            text = f"{event.title} ({event.event_type})"
            if event.partner_brand:
                text += f" | {event.partner_brand}"
            ctk.CTkLabel(self.main_frame, text=text).pack(anchor="w", pady=3, padx=30)


if __name__ == "__main__":
    app = JuniorClimbsApp()
    app.mainloop()

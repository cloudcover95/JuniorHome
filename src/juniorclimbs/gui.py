# path: src/juniorclimbs/gui.py

"""
JuniorClimbs Desktop GUI

Local software suite for gym staff. Point-and-click + keyboard input.
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

        self.current_staff_id = None

        self._create_widgets()

    def _create_widgets(self):
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

        ctk.CTkLabel(self.sidebar, text="System", font=("Arial", 12)).pack(pady=(30, 5))
        ctk.CTkButton(self.sidebar, text="Check for Updates", command=self._check_for_updates, width=200).pack(pady=4)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self.show_checkin()

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_status(self, message: str, success: bool = True):
        color = "#28a745" if success else "#dc3545"
        label = ctk.CTkLabel(self.main_frame, text=message, text_color=color)
        label.pack(pady=8)
        self.after(2800, label.destroy)

    # ==================== CHECK-IN + WAIVER ====================
    def show_checkin(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Member Check-in", font=("Arial", 20, "bold")).pack(pady=12)

        self.checkin_search = ctk.CTkEntry(self.main_frame, placeholder_text="Search by ID or Name", width=380, height=38)
        self.checkin_search.pack(pady=8)

        ctk.CTkButton(self.main_frame, text="Search", command=self._search_member_for_checkin, width=160).pack(pady=4)

        self.checkin_result = ctk.CTkFrame(self.main_frame)
        self.checkin_result.pack(pady=12, fill="x", padx=40)

        ctk.CTkButton(self.main_frame, text="Check In", command=self._do_checkin, width=220, height=42, fg_color="#28a745").pack(pady=15)

    def _search_member_for_checkin(self):
        for w in self.checkin_result.winfo_children():
            w.destroy()

        query = self.checkin_search.get().strip().lower()
        if not query: return

        found = None
        for m in self.mm.members.values():
            if query in m.id.lower() or query in m.full_name.lower():
                found = m
                break

        if found:
            self.current_checkin_member = found
            has_waiver = any(w.member_id == found.id for w in self.wm.waivers.values())
            status_text = f"{found.full_name} | Balance: ${found.current_balance} | {found.status}"
            ctk.CTkLabel(self.checkin_result, text=status_text).pack()

            if not has_waiver:
                ctk.CTkLabel(self.checkin_result, text="New member — Waiver required", text_color="#dc3545").pack()
                ctk.CTkButton(self.checkin_result, text="Complete Waiver Now", command=lambda: self.show_waiver_flow(found), fg_color="#ffc107", text_color="black").pack(pady=6)
        else:
            ctk.CTkLabel(self.checkin_result, text="Member not found", text_color="#dc3545").pack()

    def _do_checkin(self):
        if not hasattr(self, "current_checkin_member"):
            messagebox.showwarning("Error", "Search for a member first.")
            return

        member = self.current_checkin_member
        has_waiver = any(w.member_id == member.id for w in self.wm.waivers.values())

        if not has_waiver:
            messagebox.showwarning("Waiver Required", "Please complete the waiver first.")
            self.show_waiver_flow(member)
            return

        self.mm.check_in(member.id)
        self.show_status(f"Checked in: {member.full_name}")

    def show_waiver_flow(self, member):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Liability Waiver", font=("Arial", 18, "bold")).pack(pady=10)
        ctk.CTkLabel(self.main_frame, text=f"Member: {member.full_name}").pack()

        questions = [
            "I have read and understand all gym safety rules and policies.",
            "I am physically capable of participating in climbing activities.",
            "I understand the inherent risks involved in climbing and bouldering.",
            "I will follow all staff instructions and posted safety guidelines.",
            "I accept full responsibility for my own safety and any injuries."
        ]

        self.waiver_vars = {}
        for i, q in enumerate(questions):
            frame = ctk.CTkFrame(self.main_frame)
            frame.pack(fill="x", pady=3, padx=40)
            ctk.CTkLabel(frame, text=q, wraplength=520, anchor="w").pack(side="left", padx=8)
            var = ctk.StringVar(value="No")
            self.waiver_vars[i] = var
            ctk.CTkOptionMenu(frame, values=["Yes", "No"], variable=var, width=70).pack(side="right", padx=8)

        ctk.CTkLabel(self.main_frame, text="Digital Signature (type your full name)", font=("Arial", 12)).pack(pady=(15, 4))
        self.signature_entry = ctk.CTkEntry(self.main_frame, placeholder_text="Full Legal Name", width=350)
        self.signature_entry.pack()

        ctk.CTkButton(self.main_frame, text="Submit Waiver (All answers must be Yes)", 
                        command=lambda: self._submit_waiver(member), width=320, fg_color="#28a745").pack(pady=20)

    def _submit_waiver(self, member):
        answers = {i: v.get() == "Yes" for i, v in self.waiver_vars.items()}
        if not all(answers.values()):
            messagebox.showerror("Invalid", "All questions must be answered Yes.")
            return

        signature = self.signature_entry.get().strip()
        if not signature:
            messagebox.showwarning("Signature Required", "Please type your full name as signature.")
            return

        self.wm.sign_waiver(member.id, signature_data=signature)
        member.status = MembershipStatus.ACTIVE
        self.show_status("Waiver completed and member activated.")
        self.show_checkin()

    # ==================== POS ====================
    def show_pos(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Point of Sale", font=("Arial", 20, "bold")).pack(pady=12)

        self.pos_member = ctk.CTkEntry(self.main_frame, placeholder_text="Member ID", width=320)
        self.pos_member.pack(pady=6)

        ctk.CTkLabel(self.main_frame, text="Quick Presets").pack(pady=(10, 4))
        preset_frame = ctk.CTkFrame(self.main_frame)
        preset_frame.pack()

        presets = [("Day Pass $25", 25, TransactionType.DAY_PASS),
                   ("Merch $15", 15, TransactionType.MERCH),
                   ("Food/Drink $8", 8, TransactionType.FOOD_DRINK)]

        for label, amt, ttype in presets:
            ctk.CTkButton(preset_frame, text=label, width=140,
                          command=lambda a=amt, t=ttype: self._quick_pos(a, t)).pack(side="left", padx=6)

        self.pos_amount = ctk.CTkEntry(self.main_frame, placeholder_text="Custom Amount", width=200)
        self.pos_amount.pack(pady=10)

        ctk.CTkButton(self.main_frame, text="Complete Sale", command=self._custom_pos, width=200).pack(pady=8)

        # Payment method
        self.payment_method = ctk.CTkComboBox(self.main_frame, values=["Cash", "Card", "Account Balance"], width=200)
        self.payment_method.pack(pady=6)
        self.payment_method.set("Cash")

    def _quick_pos(self, amount, tx_type):
        mid = self.pos_member.get().strip()
        if not mid: 
            messagebox.showwarning("Missing", "Enter Member ID")
            return
        self.pos.sell_to_member(mid, tx_type, amount, PaymentMethod.CASH)
        self.show_status(f"Sale completed: ${amount}")

    def _custom_pos(self):
        mid = self.pos_member.get().strip()
        try:
            amt = float(self.pos_amount.get())
        except:
            messagebox.showerror("Error", "Invalid amount")
            return
        method = self.payment_method.get()
        pm = PaymentMethod.CASH if method == "Cash" else PaymentMethod.CARD
        self.pos.sell_to_member(mid, TransactionType.OTHER, amt, pm)
        self.show_status(f"Sale of ${amt} completed via {method}")

    # ==================== MEMBERS + TOP UP + RENEWAL ====================
    def show_members(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Members & Balances", font=("Arial", 20, "bold")).pack(pady=12)

        for m in self.mm.get_active_members():
            frame = ctk.CTkFrame(self.main_frame)
            frame.pack(fill="x", pady=4, padx=20)

            exp_text = m.membership_expires.strftime("%Y-%m-%d") if m.membership_expires else "N/A"
            text = f"{m.full_name} | Bal: ${m.current_balance} | Expires: {exp_text}"
            ctk.CTkLabel(frame, text=text).pack(side="left", padx=10)

            ctk.CTkButton(frame, text="Top Up", width=80, command=lambda mid=m.id: self._top_up_balance(mid)).pack(side="right", padx=4)

            if m.membership_expires and m.membership_expires < datetime.utcnow() + timedelta(days=14):
                ctk.CTkButton(frame, text="Renew Now", width=90, fg_color="#ffc107", text_color="black",
                              command=lambda mid=m.id: self._renew_membership(mid)).pack(side="right", padx=4)

    def _top_up_balance(self, member_id):
        member = self.mm.get_member(member_id)
        if not member: return

        dialog = ctk.CTkInputDialog(text="Enter amount to top up:", title="Top Up Balance")
        try:
            amount = float(dialog.get_input())
            member.current_balance += amount
            self.show_status(f"Added ${amount} to {member.full_name}")
            self.show_members()
        except:
            pass

    def _renew_membership(self, member_id):
        if messagebox.askyesno("Confirm Renewal", "Renew membership for 1 month?"):
            self.mm.auto_renew_membership(member_id, months=1)
            self.show_status("Membership renewed.")
            self.show_members()

    # ==================== SAFETY ====================
    def show_safety(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Safety Zones", font=("Arial", 20, "bold")).pack(pady=12)

        for area in self.safety.get_all_areas():
            frame = ctk.CTkFrame(self.main_frame)
            frame.pack(fill="x", pady=5, padx=20)
            ctk.CTkLabel(frame, text=f"{area.name} ({area.area_type}) - {area.status}").pack(side="left", padx=10)

            for s in ["open", "restricted", "closed"]:
                ctk.CTkButton(frame, text=s.title(), width=85, command=lambda a=area.id, st=s: self._update_safety(a, st)).pack(side="right", padx=3)

    def _update_safety(self, area_id, status):
        self.safety.update_area_status(area_id, status, override=True)
        self.show_safety()

    # ==================== MY SCHEDULE ====================
    def show_my_schedule(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="My Schedule", font=("Arial", 20, "bold")).pack(pady=12)

        upcoming = self.sm.get_upcoming_shifts(14) if not self.current_staff_id else \
                   self.sm.get_employee_shifts(self.current_staff_id)

        for shift in upcoming[:8]:
            emp = self.sm.employees.get(shift.employee_id)
            name = emp.full_name if emp else "You"
            text = f"{name} | {shift.start_time.strftime('%b %d %H:%M')} - {shift.end_time.strftime('%H:%M')} | {shift.get_break_info()}"
            ctk.CTkLabel(self.main_frame, text=text).pack(anchor="w", pady=2, padx=30)

    # ==================== EVENTS ====================
    def show_events(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Events & Sponsorships", font=("Arial", 20, "bold")).pack(pady=12)

        for e in self.em.get_upcoming_events(30):
            txt = f"{e.title} ({e.event_type})"
            if e.partner_brand: txt += f" | {e.partner_brand}"
            if e.incentive: txt += f" | {e.incentive}"
            ctk.CTkLabel(self.main_frame, text=txt).pack(anchor="w", pady=2, padx=30)

    # ==================== UPDATE MECHANISM ====================
    def _check_for_updates(self):
        messagebox.showinfo("Updates", "Checking for updates...\n\nThis feature is owner-controlled and will be enabled in a future update.")


if __name__ == "__main__":
    app = JuniorClimbsApp()
    app.mainloop()

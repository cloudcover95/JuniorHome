# path: src/juniorclimbs/gui.py

"""
JuniorClimbs Desktop GUI

Local desktop software for gym staff operations.
- Staff sign in for shifts
- Member profile appears in corner on check-in / search
- Point & click focused
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
        self.geometry("1200x780")

        self.mm = MemberManager()
        self.ledger = Ledger()
        self.pos = POS(self.mm, self.ledger)
        self.safety = SafetyManager()
        self.wm = WaiverManager(self.mm)
        self.sm = ScheduleManager()
        self.em = EventManager()

        self.current_staff_id = None
        self.current_member = None          # For member profile corner

        self._create_widgets()

    def _create_widgets(self):
        # Top bar with staff sign-in
        self.top_bar = ctk.CTkFrame(self, height=50)
        self.top_bar.pack(fill="x", padx=8, pady=(8, 0))

        self.staff_label = ctk.CTkLabel(self.top_bar, text="Not Signed In", font=("Arial", 14))
        self.staff_label.pack(side="left", padx=15)

        ctk.CTkButton(self.top_bar, text="Sign In for Shift", command=self._sign_in_staff, width=160).pack(side="right", padx=10)

        # Main content + right sidebar for profiles
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        # Sidebar navigation (left)
        self.sidebar = ctk.CTkFrame(self.content_frame, width=200)
        self.sidebar.pack(side="left", fill="y", padx=(0, 8))

        ctk.CTkLabel(self.sidebar, text="JuniorClimbs", font=("Arial", 20, "bold")).pack(pady=20)

        nav = [
            ("Check-in", self.show_checkin),
            ("Point of Sale", self.show_pos),
            ("Safety Zones", self.show_safety),
            ("Members", self.show_members),
            ("My Schedule", self.show_my_schedule),
            ("Events", self.show_events),
        ]
        for text, cmd in nav:
            ctk.CTkButton(self.sidebar, text=text, command=cmd, width=180).pack(pady=5)

        # Main working area
        self.main_area = ctk.CTkFrame(self.content_frame)
        self.main_area.pack(side="left", fill="both", expand=True)

        # Right panel - Context Profile (Member or Staff)
        self.profile_panel = ctk.CTkFrame(self, width=260)
        self.profile_panel.pack(side="right", fill="y", padx=8, pady=8)

        ctk.CTkLabel(self.profile_panel, text="Context Profile", font=("Arial", 14, "bold")).pack(pady=10)
        self.profile_content = ctk.CTkFrame(self.profile_panel)
        self.profile_content.pack(fill="both", expand=True, padx=8, pady=8)

        self.show_checkin()
        self._update_profile_panel()   # Initial empty state

    def _update_profile_panel(self, member=None, staff=None):
        for w in self.profile_content.winfo_children():
            w.destroy()

        if member:
            ctk.CTkLabel(self.profile_content, text="Current Member", font=("Arial", 13, "bold")).pack(pady=6)
            ctk.CTkLabel(self.profile_content, text=member.full_name, font=("Arial", 16)).pack()
            ctk.CTkLabel(self.profile_content, text=f"Balance: ${member.current_balance}").pack(pady=4)
            if member.membership_expires:
                ctk.CTkLabel(self.profile_content, text=f"Expires: {member.membership_expires.strftime('%Y-%m-%d')}").pack()
            ctk.CTkLabel(self.profile_content, text="Recent Activity", font=("Arial", 11)).pack(pady=(12, 4))
            history = self.mm.get_member_history(member.id)[:3]
            for h in history:
                ctk.CTkLabel(self.profile_content, text=f"• {h['event']}", font=("Arial", 10)).pack(anchor="w")

        elif staff:
            ctk.CTkLabel(self.profile_content, text="Signed In Staff", font=("Arial", 13, "bold")).pack(pady=6)
            ctk.CTkLabel(self.profile_content, text=staff.full_name, font=("Arial", 16)).pack()
            ctk.CTkLabel(self.profile_content, text=f"Role: {staff.role}").pack()
            ctk.CTkLabel(self.profile_content, text="Notifications", font=("Arial", 11)).pack(pady=(12, 4))
            ctk.CTkLabel(self.profile_content, text="(No new notifications)", text_color="gray").pack()
        else:
            ctk.CTkLabel(self.profile_content, text="No active context", text_color="gray").pack(pady=20)

    def _sign_in_staff(self):
        # Simple staff sign-in dialog for beta
        dialog = ctk.CTkInputDialog(text="Enter your Employee ID or Name:", title="Sign In for Shift")
        value = dialog.get_input()
        if not value:
            return

        # Find or create simple staff record
        found = None
        for emp in self.sm.employees.values():
            if value.lower() in emp.full_name.lower() or value == emp.id:
                found = emp
                break

        if not found:
            found = self.sm.add_employee(value, role="staff")

        self.current_staff_id = found.id
        self.staff_label.configure(text=f"Signed in: {found.full_name}")
        self._update_profile_panel(staff=found)
        self.show_status(f"Shift started for {found.full_name}")

    def clear_main_frame(self):
        for widget in self.main_area.winfo_children():
            widget.destroy()

    def show_status(self, message: str, success: bool = True):
        color = "#28a745" if success else "#dc3545"
        label = ctk.CTkLabel(self.main_area, text=message, text_color=color)
        label.pack(pady=6)
        self.after(2500, label.destroy)

    # ==================== CHECK-IN ====================
    def show_checkin(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_area, text="Member Check-in (Seamless Scan)", font=("Arial", 18, "bold")).pack(pady=10)

        self.checkin_entry = ctk.CTkEntry(self.main_area, placeholder_text="Scan or type Member ID / Name", width=400, height=40)
        self.checkin_entry.pack(pady=8)
        self.checkin_entry.bind("<Return>", lambda e: self._search_and_checkin())

        ctk.CTkButton(self.main_area, text="Check In", command=self._search_and_checkin, width=200, height=42, fg_color="#28a745").pack(pady=10)

    def _search_and_checkin(self):
        query = self.checkin_entry.get().strip()
        if not query:
            return

        member = None
        for m in self.mm.members.values():
            if query.lower() in m.id.lower() or query.lower() in m.full_name.lower():
                member = m
                break

        if not member:
            messagebox.showinfo("New Member", "Member not found. Please complete waiver.")
            # Could open new member creation here in future
            return

        # Check waiver
        has_waiver = any(w.member_id == member.id for w in self.wm.waivers.values())
        if not has_waiver:
            messagebox.showwarning("Waiver Required", "This member must complete the liability waiver first.")
            self.show_waiver_flow(member)
            return

        # Perform check-in with rich logging
        self.mm.check_in(member.id, method="desktop_scan")
        self.current_member = member

        self.show_status(f"Checked in: {member.full_name}")
        self.checkin_entry.delete(0, "end")

        # Show member profile in corner panel
        self._update_profile_panel(member=member)

    # ==================== WAIVER ====================
    def show_waiver_flow(self, member):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_area, text="Complete Liability Waiver", font=("Arial", 18, "bold")).pack(pady=10)

        questions = [
            "I have read and understand all gym safety rules.",
            "I am physically capable of climbing activities.",
            "I understand the risks involved.",
            "I will follow staff instructions and safety guidelines.",
            "I accept responsibility for my own safety."
        ]

        self.waiver_vars = {}
        for i, q in enumerate(questions):
            f = ctk.CTkFrame(self.main_area)
            f.pack(fill="x", pady=3, padx=30)
            ctk.CTkLabel(f, text=q, wraplength=480, anchor="w").pack(side="left", padx=6)
            var = ctk.StringVar(value="No")
            self.waiver_vars[i] = var
            ctk.CTkOptionMenu(f, values=["Yes", "No"], variable=var, width=70).pack(side="right", padx=6)

        ctk.CTkLabel(self.main_area, text="Type your full name as digital signature", font=("Arial", 12)).pack(pady=(12, 4))
        self.sig_entry = ctk.CTkEntry(self.main_area, placeholder_text="Full Legal Name", width=320)
        self.sig_entry.pack()

        ctk.CTkButton(self.main_area, text="Submit Waiver", command=lambda: self._submit_waiver(member),
                        width=280, fg_color="#28a745").pack(pady=15)

    def _submit_waiver(self, member):
        if not all(v.get() == "Yes" for v in self.waiver_vars.values()):
            messagebox.showerror("Error", "All answers must be Yes.")
            return

        sig = self.sig_entry.get().strip()
        if not sig:
            messagebox.showwarning("Signature", "Please enter your full name.")
            return

        self.wm.sign_waiver(member.id, signature_data=sig)
        member.status = MembershipStatus.ACTIVE
        self.show_status("Waiver completed. Member activated.")
        self.show_checkin()

    # ==================== POS ====================
    def show_pos(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_area, text="Point of Sale", font=("Arial", 18, "bold")).pack(pady=10)

        self.pos_member = ctk.CTkEntry(self.main_area, placeholder_text="Member ID", width=300)
        self.pos_member.pack(pady=6)

        ctk.CTkLabel(self.main_area, text="Quick Actions").pack(pady=6)
        for label, amt, ttype in [("Day Pass $25", 25, TransactionType.DAY_PASS),
                                   ("Merch $15", 15, TransactionType.MERCH),
                                   ("Food $8", 8, TransactionType.FOOD_DRINK)]:
            ctk.CTkButton(self.main_area, text=label, width=160,
                          command=lambda a=amt, t=ttype: self._quick_sale(a, t)).pack(pady=3)

        self.pos_amount = ctk.CTkEntry(self.main_area, placeholder_text="Custom Amount", width=200)
        self.pos_amount.pack(pady=8)

        ctk.CTkButton(self.main_area, text="Complete Sale", command=self._do_sale, width=200).pack(pady=8)

        self.pay_method = ctk.CTkComboBox(self.main_area, values=["Cash", "Card", "Account Balance"], width=200)
        self.pay_method.pack(pady=6)
        self.pay_method.set("Cash")

    def _quick_sale(self, amount, tx_type):
        mid = self.pos_member.get().strip()
        if not mid:
            messagebox.showwarning("Missing", "Enter Member ID")
            return
        self.pos.sell_to_member(mid, tx_type, amount, PaymentMethod.CASH)
        self.show_status(f"Sale of ${amount} completed")

    def _do_sale(self):
        mid = self.pos_member.get().strip()
        try:
            amt = float(self.pos_amount.get())
        except:
            messagebox.showerror("Error", "Invalid amount")
            return
        method_str = self.pay_method.get()
        pm = PaymentMethod.ACCOUNT_BALANCE if method_str == "Account Balance" else PaymentMethod.CASH
        self.pos.sell_to_member(mid, TransactionType.OTHER, amt, pm)
        self.show_status(f"Sale completed via {method_str}")

    # ==================== SAFETY ====================
    def show_safety(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_area, text="Safety Zones", font=("Arial", 18, "bold")).pack(pady=10)

        for area in self.safety.get_all_areas():
            f = ctk.CTkFrame(self.main_area)
            f.pack(fill="x", pady=4, padx=20)
            ctk.CTkLabel(f, text=f"{area.name} - {area.status}").pack(side="left", padx=10)

            for st in ["open", "restricted", "closed"]:
                ctk.CTkButton(f, text=st.title(), width=80, command=lambda a=area.id, s=st: self._update_area(a, s)).pack(side="right", padx=3)

    def _update_area(self, area_id, status):
        self.safety.update_area_status(area_id, status, override=True)
        self.show_safety()

    # ==================== MEMBERS ====================
    def show_members(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_area, text="Members", font=("Arial", 18, "bold")).pack(pady=10)

        for m in self.mm.get_active_members():
            f = ctk.CTkFrame(self.main_area)
            f.pack(fill="x", pady=3, padx=20)
            ctk.CTkLabel(f, text=f"{m.full_name} | Bal: ${m.current_balance}").pack(side="left", padx=10)
            ctk.CTkButton(f, text="Top Up", width=70, command=lambda mid=m.id: self._top_up(mid)).pack(side="right", padx=4)

    def _top_up(self, member_id):
        member = self.mm.get_member(member_id)
        if not member: return
        d = ctk.CTkInputDialog(text="Amount to add:", title="Top Up Balance")
        try:
            amt = float(d.get_input())
            member.current_balance += amt
            self.show_status(f"Added ${amt}")
            self.show_members()
        except:
            pass

    # ==================== MY SCHEDULE ====================
    def show_my_schedule(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_area, text="My Schedule", font=("Arial", 18, "bold")).pack(pady=10)

        shifts = self.sm.get_upcoming_shifts(14)
        for s in shifts[:6]:
            emp = self.sm.employees.get(s.employee_id)
            name = emp.full_name if emp else "You"
            ctk.CTkLabel(self.main_area, text=f"{name} | {s.start_time.strftime('%b %d %H:%M')} - {s.end_time.strftime('%H:%M')} | {s.get_break_info()}").pack(anchor="w", pady=2, padx=30)

    # ==================== EVENTS ====================
    def show_events(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_area, text="Events & Sponsorships", font=("Arial", 18, "bold")).pack(pady=10)

        for e in self.em.get_upcoming_events(30):
            txt = f"{e.title} ({e.event_type})"
            if e.partner_brand: txt += f" | {e.partner_brand}"
            ctk.CTkLabel(self.main_area, text=txt).pack(anchor="w", pady=2, padx=30)


if __name__ == "__main__":
    app = JuniorClimbsApp()
    app.mainloop()

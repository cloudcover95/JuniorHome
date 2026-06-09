# path: src/juniorclimbs/gui.py

"""
JuniorClimbs Desktop GUI

Local desktop software suite for gym staff operations.
Point-and-click focused. Member portal is separate (web).
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
        self.current_member = None

        self._create_widgets()

    def _create_widgets(self):
        self.top_bar = ctk.CTkFrame(self, height=48)
        self.top_bar.pack(fill="x", padx=8, pady=(8, 0))

        self.staff_status = ctk.CTkLabel(self.top_bar, text="Not Signed In", font=("Arial", 13))
        self.staff_status.pack(side="left", padx=15)

        ctk.CTkButton(self.top_bar, text="Sign In for Shift", command=self._sign_in_for_shift, width=150).pack(side="right", padx=10)

        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=8, pady=8)

        self.sidebar = ctk.CTkFrame(self.main_container, width=200)
        self.sidebar.pack(side="left", fill="y", padx=(0, 8))

        ctk.CTkLabel(self.sidebar, text="JuniorClimbs", font=("Arial", 20, "bold")).pack(pady=18)

        for text, cmd in [
            ("Check-in", self.show_checkin),
            ("Point of Sale", self.show_pos),
            ("Safety Zones", self.show_safety),
            ("Members", self.show_members),
            ("My Schedule", self.show_my_schedule),
            ("Events", self.show_events),
            ("Reports", self.show_reports),
        ]:
            ctk.CTkButton(self.sidebar, text=text, command=cmd, width=180).pack(pady=4)

        self.main_area = ctk.CTkFrame(self.main_container)
        self.main_area.pack(side="left", fill="both", expand=True)

        self.context_panel = ctk.CTkFrame(self.main_container, width=280)
        self.context_panel.pack(side="right", fill="y", padx=8)

        ctk.CTkLabel(self.context_panel, text="Active Context", font=("Arial", 13, "bold")).pack(pady=10)
        self.context_content = ctk.CTkFrame(self.context_panel)
        self.context_content.pack(fill="both", expand=True, padx=8, pady=8)

        self.show_checkin()
        self._update_context_panel()

    def _update_context_panel(self, member=None, staff=None):
        for w in self.context_content.winfo_children():
            w.destroy()

        if member:
            ctk.CTkLabel(self.context_content, text="Member Profile", font=("Arial", 14, "bold")).pack(pady=6)
            ctk.CTkLabel(self.context_content, text=member.full_name, font=("Arial", 16)).pack()
            ctk.CTkLabel(self.context_content, text=f"Balance: ${member.current_balance}").pack(pady=4)

            if member.membership_expires:
                days_left = (member.membership_expires - datetime.utcnow()).days
                color = "#dc3545" if days_left < 7 else "#28a745"
                ctk.CTkLabel(self.context_content, text=f"Expires in {days_left} days", text_color=color).pack()

            ctk.CTkButton(self.context_content, text="Top Up Balance", width=160,
                          command=lambda: self._top_up_from_panel(member.id)).pack(pady=8)

            ctk.CTkLabel(self.context_content, text="Recent Activity", font=("Arial", 11)).pack(pady=(10, 4))
            for entry in self.mm.get_member_history(member.id)[:4]:
                ctk.CTkLabel(self.context_content, text=f"• {entry['event']}", font=("Arial", 10)).pack(anchor="w")

        elif staff:
            ctk.CTkLabel(self.context_content, text="Staff Profile", font=("Arial", 14, "bold")).pack(pady=6)
            ctk.CTkLabel(self.context_content, text=staff.full_name, font=("Arial", 16)).pack()
            ctk.CTkLabel(self.context_content, text=f"Role: {staff.role}").pack()

            # Notification preference
            ctk.CTkLabel(self.context_content, text="Notification Preference", font=("Arial", 11)).pack(pady=(10, 4))
            self.notif_pref = ctk.CTkComboBox(self.context_content, values=["email", "slack", "telegram"], width=160)
            self.notif_pref.pack()
            self.notif_pref.set(staff.notification_preference)
            ctk.CTkButton(self.context_content, text="Save Preference", width=140,
                          command=lambda: self._save_notification_pref(staff)).pack(pady=6)

        else:
            ctk.CTkLabel(self.context_content, text="No active context", text_color="gray").pack(pady=30)

    def _save_notification_pref(self, staff):
        staff.notification_preference = self.notif_pref.get()
        self.show_status("Preference saved")

    def _top_up_from_panel(self, member_id):
        member = self.mm.get_member(member_id)
        if not member: return
        d = ctk.CTkInputDialog(text="Amount to add:", title="Top Up Balance")
        try:
            amt = float(d.get_input())
            member.current_balance += amt
            self.show_status(f"Added ${amt}")
            self._update_context_panel(member=member)
        except:
            pass

    def _sign_in_for_shift(self):
        dialog = ctk.CTkInputDialog(text="Enter Employee ID or Name", title="Shift Sign-In")
        value = dialog.get_input()
        if not value: return

        staff = None
        for emp in self.sm.employees.values():
            if value.lower() in emp.full_name.lower() or value == emp.id:
                staff = emp
                break

        if not staff:
            staff = self.sm.add_employee(value, role="staff")

        self.current_staff_id = staff.id
        self.staff_status.configure(text=f"Signed in: {staff.full_name}")
        self._update_context_panel(staff=staff)
        self.show_status("Shift started")

    def clear_main_area(self):
        for w in self.main_area.winfo_children():
            w.destroy()

    def show_status(self, msg, success=True):
        color = "#28a745" if success else "#dc3545"
        lbl = ctk.CTkLabel(self.main_area, text=msg, text_color=color)
        lbl.pack(pady=5)
        self.after(2200, lbl.destroy)

    # ==================== CHECK-IN + WAIVER ====================
    def show_checkin(self):
        self.clear_main_area()
        ctk.CTkLabel(self.main_area, text="Member Check-in (Seamless)", font=("Arial", 18, "bold")).pack(pady=10)

        self.checkin_entry = ctk.CTkEntry(self.main_area, placeholder_text="Scan or type Member ID / Name", width=420, height=40)
        self.checkin_entry.pack(pady=8)
        self.checkin_entry.bind("<Return>", lambda e: self._do_checkin())

        ctk.CTkButton(self.main_area, text="Check In", command=self._do_checkin,
                      width=200, height=42, fg_color="#28a745").pack(pady=10)

    def _do_checkin(self):
        query = self.checkin_entry.get().strip()
        if not query: return

        member = None
        for m in self.mm.members.values():
            if query.lower() in m.id.lower() or query.lower() in m.full_name.lower():
                member = m
                break

        if not member:
            messagebox.showinfo("Not Found", "Member not found")
            return

        has_waiver = any(w.member_id == member.id for w in self.wm.waivers.values())
        if not has_waiver:
            messagebox.showwarning("Waiver Required", "Complete waiver first.")
            self.show_waiver_flow(member)
            return

        self.mm.check_in(member.id, method="desktop")
        self.current_member = member
        self.checkin_entry.delete(0, "end")
        self.show_status(f"Checked in: {member.full_name}")
        self._update_context_panel(member=member)

    def show_waiver_flow(self, member):
        self.clear_main_area()
        ctk.CTkLabel(self.main_area, text="Complete Liability Waiver", font=("Arial", 18, "bold")).pack(pady=8)

        questions = [
            "I have read and understand all gym safety rules and policies.",
            "I am physically capable of participating in climbing activities.",
            "I understand the inherent risks involved in climbing and bouldering.",
            "I will follow all staff instructions and posted safety guidelines.",
            "I accept full responsibility for my own safety and any injuries."
        ]

        self.waiver_vars = {}
        for i, q in enumerate(questions):
            f = ctk.CTkFrame(self.main_area)
            f.pack(fill="x", pady=3, padx=20)
            ctk.CTkLabel(f, text=q, wraplength=520, anchor="w").pack(side="left", padx=6)
            var = ctk.StringVar(value="No")
            self.waiver_vars[i] = var
            ctk.CTkOptionMenu(f, values=["Yes", "No"], variable=var, width=70).pack(side="right", padx=6)

        ctk.CTkLabel(self.main_area, text="Digital Signature (type your full name)", font=("Arial", 12)).pack(pady=(12, 4))
        self.signature_entry = ctk.CTkEntry(self.main_area, placeholder_text="Full Legal Name", width=320)
        self.signature_entry.pack()

        ctk.CTkButton(self.main_area, text="Submit Waiver (All answers must be Yes)",
                      command=lambda: self._submit_waiver(member), width=300, fg_color="#28a745").pack(pady=15)

    def _submit_waiver(self, member):
        if not all(v.get() == "Yes" for v in self.waiver_vars.values()):
            messagebox.showerror("Error", "All answers must be Yes.")
            return

        sig = self.signature_entry.get().strip()
        if not sig:
            messagebox.showwarning("Signature", "Please type your full name as signature.")
            return

        self.wm.sign_waiver(member.id, signature_data=sig)
        member.status = MembershipStatus.ACTIVE
        self.show_status("Waiver completed. Member activated.")
        self.show_checkin()

    # ==================== POS ====================
    def show_pos(self):
        self.clear_main_area()
        ctk.CTkLabel(self.main_area, text="Point of Sale", font=("Arial", 18, "bold")).pack(pady=10)

        self.pos_member = ctk.CTkEntry(self.main_area, placeholder_text="Member ID", width=300)
        self.pos_member.pack(pady=6)

        ctk.CTkLabel(self.main_area, text="Quick Presets").pack(pady=6)
        for label, amt, ttype in [("Day Pass $25", 25, TransactionType.DAY_PASS),
                                   ("Merch $15", 15, TransactionType.MERCH),
                                   ("Food $8", 8, TransactionType.FOOD_DRINK)]:
            ctk.CTkButton(self.main_area, text=label, width=150,
                          command=lambda a=amt, t=ttype: self._quick_pos(a, t)).pack(pady=3)

        self.pos_amount = ctk.CTkEntry(self.main_area, placeholder_text="Custom Amount", width=200)
        self.pos_amount.pack(pady=8)

        ctk.CTkButton(self.main_area, text="Complete Sale", command=self._do_pos_sale, width=200).pack(pady=8)

        self.pay_method = ctk.CTkComboBox(self.main_area, values=["Cash", "Card", "Account Balance"], width=200)
        self.pay_method.pack(pady=6)
        self.pay_method.set("Cash")

    def _quick_pos(self, amount, tx_type):
        mid = self.pos_member.get().strip()
        if not mid:
            messagebox.showwarning("Missing", "Enter Member ID first")
            return
        self.pos.sell_to_member(mid, tx_type, amount, PaymentMethod.CASH)
        self.show_status(f"Sale of ${amount} completed")

    def _do_pos_sale(self):
        mid = self.pos_member.get().strip()
        try:
            amt = float(self.pos_amount.get())
        except:
            messagebox.showerror("Error", "Invalid amount")
            return

        method = self.pay_method.get()
        pm = PaymentMethod.ACCOUNT_BALANCE if method == "Account Balance" else PaymentMethod.CASH
        result = self.pos.sell_to_member(mid, TransactionType.OTHER, amt, pm)

        if result is None:
            # Hard decline simulation
            messagebox.showerror("Declined", "Transaction declined (insufficient balance or account issue)")
            return

        self.show_status(f"Sale completed via {method}")

    # ==================== MEMBERS + EXPIRY + RENEWAL ====================
    def show_members(self):
        self.clear_main_area()
        ctk.CTkLabel(self.main_area, text="Members", font=("Arial", 18, "bold")).pack(pady=10)

        for m in self.mm.get_active_members():
            f = ctk.CTkFrame(self.main_area)
            f.pack(fill="x", pady=3, padx=15)

            exp_text = m.membership_expires.strftime("%Y-%m-%d") if m.membership_expires else "N/A"
            ctk.CTkLabel(f, text=f"{m.full_name} | Bal: ${m.current_balance} | Expires: {exp_text}").pack(side="left", padx=8)

            ctk.CTkButton(f, text="Top Up", width=70, command=lambda mid=m.id: self._top_up_balance(mid)).pack(side="right", padx=3)

            if m.membership_expires and m.membership_expires < datetime.utcnow() + timedelta(days=14):
                ctk.CTkButton(f, text="Renew Now", width=85, fg_color="#ffc107", text_color="black",
                              command=lambda mid=m.id: self._renew_membership(mid)).pack(side="right", padx=3)

    def _top_up_balance(self, member_id):
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

    def _renew_membership(self, member_id):
        if messagebox.askyesno("Confirm", "Renew membership for 1 month?"):
            self.mm.auto_renew_membership(member_id, months=1)
            self.show_status("Membership renewed for 1 month")
            self.show_members()

    # ==================== SAFETY ====================
    def show_safety(self):
        self.clear_main_area()
        ctk.CTkLabel(self.main_area, text="Safety Zones", font=("Arial", 18, "bold")).pack(pady=10)

        for area in self.safety.get_all_areas():
            f = ctk.CTkFrame(self.main_area)
            f.pack(fill="x", pady=4, padx=15)
            ctk.CTkLabel(f, text=f"{area.name} ({area.area_type}) - {area.status}").pack(side="left", padx=8)

            for st in ["open", "restricted", "closed"]:
                ctk.CTkButton(f, text=st.title(), width=80, command=lambda a=area.id, s=st: self._update_safety(a, s)).pack(side="right", padx=3)

    def _update_safety(self, area_id, status):
        self.safety.update_area_status(area_id, status, override=True)
        self.show_safety()

    # ==================== MY SCHEDULE + ADD SHIFT ====================
    def show_my_schedule(self):
        self.clear_main_area()
        ctk.CTkLabel(self.main_area, text="My Schedule", font=("Arial", 18, "bold")).pack(pady=10)

        shifts = self.sm.get_upcoming_shifts(14)
        for s in shifts[:8]:
            emp = self.sm.employees.get(s.employee_id)
            name = emp.full_name if emp else "You"
            ctk.CTkLabel(self.main_area, text=f"{name} | {s.start_time.strftime('%b %d %H:%M')} → {s.end_time.strftime('%H:%M')} | {s.get_break_info()}").pack(anchor="w", pady=2, padx=20)

        ctk.CTkButton(self.main_area, text="Add New Shift", command=self._add_shift_dialog, width=160).pack(pady=15)

    def _add_shift_dialog(self):
        if not self.current_staff_id:
            messagebox.showwarning("Sign In", "Please sign in first.")
            return

        start_str = ctk.CTkInputDialog(text="Start (YYYY-MM-DD HH:MM):", title="Start Time").get_input()
        end_str = ctk.CTkInputDialog(text="End (YYYY-MM-DD HH:MM):", title="End Time").get_input()

        try:
            start = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
            end = datetime.strptime(end_str, "%Y-%m-%d %H:%M")
        except:
            messagebox.showerror("Error", "Invalid date format")
            return

        shift_type = ShiftType.EIGHT_HOUR if (end - start).total_seconds() > 6*3600 else ShiftType.FOUR_HOUR
        self.sm.create_shift(self.current_staff_id, start, end, shift_type)
        self.show_status("Shift added")
        self.show_my_schedule()

    # ==================== EVENTS ====================
    def show_events(self):
        self.clear_main_area()
        ctk.CTkLabel(self.main_area, text="Events & Sponsorships", font=("Arial", 18, "bold")).pack(pady=10)

        for e in self.em.get_upcoming_events(30):
            txt = f"{e.title} ({e.event_type})"
            if e.partner_brand: txt += f" | {e.partner_brand}"
            if e.incentive: txt += f" | {e.incentive}"
            ctk.CTkLabel(self.main_area, text=txt).pack(anchor="w", pady=2, padx=20)

        ctk.CTkButton(self.main_area, text="Create Event", command=self._create_event_dialog, width=160).pack(pady=12)

    def _create_event_dialog(self):
        title = ctk.CTkInputDialog(text="Event Title:", title="Create Event").get_input()
        if not title: return

        etype = ctk.CTkInputDialog(text="Type (booth, sponsorship, meeting...):", title="Type").get_input() or "event"
        start_str = ctk.CTkInputDialog(text="Start (YYYY-MM-DD HH:MM):", title="Start").get_input()

        try:
            start = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
        except:
            messagebox.showerror("Error", "Invalid date")
            return

        self.em.create_event(title, etype, start)
        self.show_status("Event created")
        self.show_events()

    # ==================== REPORTS ====================
    def show_reports(self):
        self.clear_main_area()
        ctk.CTkLabel(self.main_area, text="Reports", font=("Arial", 18, "bold")).pack(pady=10)

        daily = self.ledger.get_daily_revenue()
        ctk.CTkLabel(self.main_area, text=f"Today's Revenue: ${daily:.2f}", font=("Arial", 14)).pack(pady=8)

        expiring = len(self.mm.get_members_with_expiring_membership(7))
        ctk.CTkLabel(self.main_area, text=f"Members expiring in 7 days: {expiring}", font=("Arial", 14)).pack(pady=4)

    # ==================== UPDATE MECHANISM ====================
    def _check_for_updates(self):
        messagebox.showinfo("Updates", "Checking for updates...\nThis is owner-controlled and will be enabled in a future release.")


if __name__ == "__main__":
    app = JuniorClimbsApp()
    app.mainloop()

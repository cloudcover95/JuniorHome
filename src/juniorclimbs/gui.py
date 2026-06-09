# path: src/juniorclimbs/gui.py

"""
JuniorClimbs Desktop GUI (Local Software Suite)

Point-and-click + keyboard input interface for regular gym staff.
Fully local. Optional network/internet updates controlled by gym owner.
Lightning fast, aligned with BitNet-efficient architecture.
"""

import customtkinter as ctk
from tkinter import messagebox
from datetime import datetime

from .member_manager import MemberManager
from .ledger import Ledger
from .pos import POS
from .safety import SafetyManager
from .waiver import WaiverManager
from .employee import ScheduleManager
from .events import EventManager

from .models import TransactionType, PaymentMethod

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class JuniorClimbsApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("JuniorClimbs - Gym Operations")
        self.geometry("1000x700")

        # Core systems
        self.mm = MemberManager()
        self.ledger = Ledger()
        self.pos = POS(self.mm, self.ledger)
        self.safety = SafetyManager()
        self.wm = WaiverManager(self.mm)
        self.sm = ScheduleManager()
        self.em = EventManager()

        self._create_widgets()

    def _create_widgets(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="JuniorClimbs", font=("Arial", 20, "bold")).pack(pady=20)

        buttons = [
            ("Check-in", self.show_checkin),
            ("Point of Sale", self.show_pos),
            ("Safety Zones", self.show_safety),
            ("Members", self.show_members),
            ("Employee Schedule", self.show_schedule),
            ("Events", self.show_events),
        ]

        for text, command in buttons:
            btn = ctk.CTkButton(self.sidebar, text=text, command=command, width=180)
            btn.pack(pady=8)

        # Main content area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.current_view = None
        self.show_checkin()  # Default view

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    # ==================== VIEWS ====================

    def show_checkin(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Member Check-in", font=("Arial", 18, "bold")).pack(pady=20)

        self.checkin_entry = ctk.CTkEntry(self.main_frame, placeholder_text="Member ID or Name", width=300)
        self.checkin_entry.pack(pady=10)

        btn = ctk.CTkButton(self.main_frame, text="Check In", command=self._do_checkin, width=200)
        btn.pack(pady=10)

    def _do_checkin(self):
        identifier = self.checkin_entry.get().strip()
        if not identifier:
            messagebox.showwarning("Input Error", "Please enter member ID or name.")
            return

        # In real version: check waiver status first
        member = self.mm.get_member(identifier) or self._find_member_by_name(identifier)
        if member:
            self.mm.check_in(member.id)
            messagebox.showinfo("Success", f"Checked in: {member.full_name}")
            self.checkin_entry.delete(0, "end")
        else:
            messagebox.showerror("Not Found", "Member not found. New members must complete waiver first.")

    def _find_member_by_name(self, name: str):
        for m in self.mm.members.values():
            if name.lower() in m.full_name.lower():
                return m
        return None

    def show_pos(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Point of Sale", font=("Arial", 18, "bold")).pack(pady=20)

        self.pos_member = ctk.CTkEntry(self.main_frame, placeholder_text="Member ID", width=300)
        self.pos_member.pack(pady=8)

        self.pos_item = ctk.CTkComboBox(self.main_frame, values=["Day Pass", "Merch", "Food/Drink"], width=300)
        self.pos_item.pack(pady=8)
        self.pos_item.set("Day Pass")

        self.pos_amount = ctk.CTkEntry(self.main_frame, placeholder_text="Amount", width=300)
        self.pos_amount.pack(pady=8)

        btn = ctk.CTkButton(self.main_frame, text="Complete Sale", command=self._do_pos, width=200)
        btn.pack(pady=15)

    def _do_pos(self):
        member_id = self.pos_member.get().strip()
        item = self.pos_item.get()
        try:
            amount = float(self.pos_amount.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid amount")
            return

        item_map = {
            "Day Pass": TransactionType.DAY_PASS,
            "Merch": TransactionType.MERCH,
            "Food/Drink": TransactionType.FOOD_DRINK,
        }
        tx_type = item_map.get(item, TransactionType.OTHER)

        result = self.pos.sell_to_member(member_id, tx_type, amount, PaymentMethod.CASH)
        if result:
            messagebox.showinfo("Success", f"Sale completed. New balance: {result.get('member_balance_after', 'N/A')}")
        else:
            messagebox.showerror("Error", "Member not found")

    def show_safety(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Safety Zones", font=("Arial", 18, "bold")).pack(pady=20)

        for area in self.safety.get_all_areas():
            frame = ctk.CTkFrame(self.main_frame)
            frame.pack(fill="x", pady=6)

            ctk.CTkLabel(frame, text=f"{area.name} ({area.area_type}) - Status: {area.status}").pack(side="left", padx=10)

            btn_open = ctk.CTkButton(frame, text="Open", width=80, command=lambda a=area.id: self._update_safety(a, "open"))
            btn_open.pack(side="right", padx=5)

            btn_restricted = ctk.CTkButton(frame, text="Restricted", width=80, command=lambda a=area.id: self._update_safety(a, "restricted"))
            btn_restricted.pack(side="right", padx=5)

    def _update_safety(self, area_id: str, status: str):
        self.safety.update_area_status(area_id, status, override=True)
        self.show_safety()  # Refresh view

    def show_members(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Active Members", font=("Arial", 18, "bold")).pack(pady=20)

        for m in self.mm.get_active_members():
            ctk.CTkLabel(self.main_frame, text=f"{m.full_name} | Balance: ${m.current_balance} | Status: {m.status}").pack(anchor="w", pady=4)

    def show_schedule(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Employee Schedule (Next 14 days)", font=("Arial", 18, "bold")).pack(pady=20)

        upcoming = self.sm.get_upcoming_shifts(14)
        if not upcoming:
            ctk.CTkLabel(self.main_frame, text="No upcoming shifts.").pack()
            return

        for shift in upcoming:
            emp = self.sm.employees.get(shift.employee_id)
            name = emp.full_name if emp else "Unknown"
            ctk.CTkLabel(self.main_frame, text=f"{name} | {shift.start_time.strftime('%Y-%m-%d %H:%M')} - {shift.end_time.strftime('%H:%M')} | {shift.get_break_info()}").pack(anchor="w", pady=3)

    def show_events(self):
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Upcoming Events & Sponsorships", font=("Arial", 18, "bold")).pack(pady=20)

        events = self.em.get_upcoming_events(30)
        if not events:
            ctk.CTkLabel(self.main_frame, text="No upcoming events.").pack()
            return

        for event in events:
            text = f"{event.title} ({event.event_type})"
            if event.partner_brand:
                text += f" | {event.partner_brand}"
            if event.incentive:
                text += f" | {event.incentive}"
            ctk.CTkLabel(self.main_frame, text=text).pack(anchor="w", pady=4)


if __name__ == "__main__":
    app = JuniorClimbsApp()
    app.mainloop()

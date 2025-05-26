import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os

SAVE_FILE = "run_save.json"

def load_sim_state(file_path):
    if not os.path.exists(file_path):
        messagebox.showerror("File Not Found", f"Save file not found: {file_path}")
        return None
    with open(file_path, "r") as f:
        return json.load(f)

class MindscapeDashboard(tk.Tk):
    def __init__(self, file_path=SAVE_FILE):
        super().__init__()
        self.title("Mindscape Quantum Simulation Dashboard")
        self.geometry("950x720")
        self.resizable(True, True)
        self.file_path = file_path
        self.sim_data = None
        self.entity_var = tk.StringVar()
        
        self.create_widgets()
        self.load_state()

    def create_widgets(self):
        stats_frame = ttk.LabelFrame(self, text="Universe Stats")
        stats_frame.pack(fill='x', padx=8, pady=4)
        self.universe_name = ttk.Label(stats_frame, text="Universe Name: -")
        self.universe_name.pack(side="left", padx=12)
        self.tick_label = ttk.Label(stats_frame, text="Tick: -")
        self.tick_label.pack(side="left", padx=12)
        self.qbits_label = ttk.Label(stats_frame, text="Qubits: -")
        self.qbits_label.pack(side="left", padx=12)

        # Quantum State
        self.quantum_frame = ttk.LabelFrame(self, text="Quantum State (Summary)")
        self.quantum_frame.pack(fill='x', padx=8, pady=4)
        self.qstate_text = tk.Text(self.quantum_frame, height=2, width=80, font=('Courier', 10))
        self.qstate_text.pack(side="left", padx=8, pady=2)
        self.qnorm_label = ttk.Label(self.quantum_frame, text="Norm: -")
        self.qnorm_label.pack(side="left", padx=8)
        
        # Entities Table
        entity_frame = ttk.LabelFrame(self, text="Entities")
        entity_frame.pack(fill='x', padx=8, pady=4)
        self.entity_tree = ttk.Treeview(entity_frame, columns=('Energy', 'Alive', 'Happiness', 'Fear', 'Curiosity', 'Goals', 'Knowledge'), show='headings', height=4)
        for col in self.entity_tree['columns']:
            self.entity_tree.heading(col, text=col)
        self.entity_tree.pack(fill='x', padx=4, pady=2)
        self.entity_tree.bind("<<TreeviewSelect>>", self.on_entity_select)

        # Entity Details
        self.details_frame = ttk.LabelFrame(self, text="Entity Details")
        self.details_frame.pack(fill='x', padx=8, pady=4)
        self.details_text = scrolledtext.ScrolledText(self.details_frame, height=8, width=110, font=('Courier', 10))
        self.details_text.pack(fill='x', padx=4, pady=2)

        # Universe History/Logs
        logs_frame = ttk.LabelFrame(self, text="Universe History / Logs")
        logs_frame.pack(fill='both', padx=8, pady=4, expand=True)
        self.logs_text = scrolledtext.ScrolledText(logs_frame, height=8, width=110, font=('Courier', 10))
        self.logs_text.pack(fill='both', padx=4, pady=2, expand=True)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=8, pady=4)
        ttk.Button(btn_frame, text="Reload", command=self.load_state).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Export JSON", command=self.export_state).pack(side="left", padx=6)

    def load_state(self):
        self.sim_data = load_sim_state(self.file_path)
        if not self.sim_data:
            return
        self.update_dashboard()

    def update_dashboard(self):
        # Universe stats
        self.universe_name.config(text=f"Universe: {self.sim_data['name']}")
        self.tick_label.config(text=f"Tick: {self.sim_data['time']}")
        self.qbits_label.config(text=f"Qubits: {self.sim_data['quantum_size']}")

        # Quantum state
        qstate = self.sim_data.get("quantum_state_vector", [])
        show_vec = ", ".join([f"{(c[0]**2 + c[1]**2) ** 0.5:.3f}" for c in qstate[:8]])
        self.qstate_text.delete("1.0", tk.END)
        self.qstate_text.insert("end", f"First 8 amplitudes (magnitude): {show_vec}")

        norm = sum((c[0]**2 + c[1]**2) for c in qstate) ** 0.5
        self.qnorm_label.config(text=f"Norm: {norm:.4f}")


        # Entities
        entities = self.sim_data.get("entities", [])
        for row in self.entity_tree.get_children():
            self.entity_tree.delete(row)
        for e in entities:
            self.entity_tree.insert("", "end", iid=e["name"], values=(
                round(e["energy"], 2),
                e["alive"],
                round(e["emotions"]["happiness"], 2),
                round(e["emotions"]["fear"], 2),
                round(e["emotions"]["curiosity"], 2),
                ", ".join(e.get("goals", [])),
                len(e.get("knowledge", []))
            ))
        if entities:
            self.entity_tree.selection_set(entities[0]["name"])
            self.show_entity_details(entities[0])

        # Logs / History
        logs = self.sim_data.get("history", [])
        self.logs_text.delete("1.0", tk.END)
        self.logs_text.insert("end", "\n".join(logs[-12:]))

    def on_entity_select(self, event):
        sel = self.entity_tree.selection()
        if not sel:
            return
        name = sel[0]
        for e in self.sim_data.get("entities", []):
            if e["name"] == name:
                self.show_entity_details(e)
                break

    def show_entity_details(self, e):
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert("end", f"Name: {e['name']}\n")
        self.details_text.insert("end", f"Alive: {e['alive']}\n")
        self.details_text.insert("end", f"Energy: {e['energy']:.2f}\n")
        self.details_text.insert("end", f"Goals: {e['goals']}\n")
        self.details_text.insert("end", f"Emotions: {e['emotions']}\n")
        self.details_text.insert("end", f"Knowledge items: {len(e['knowledge'])}\n")
        if e['knowledge']:
            self.details_text.insert("end", f"Sample knowledge: {list(e['knowledge'])[:3]}\n")
        self.details_text.insert("end", "\nRecent Thoughts:\n")
        for t in e.get('thoughts', [])[-4:]:
            self.details_text.insert("end", f"- {t['text']}\n")
        self.details_text.insert("end", f"\nAssociative Memory (sample): {dict(list(e.get('associative_memory', {}).items())[:2])}\n")

    def export_state(self):
        if not self.sim_data:
            messagebox.showerror("No Data", "No simulation data loaded.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.sim_data, f, indent=2)
            messagebox.showinfo("Exported", f"Simulation state exported to {file_path}")

if __name__ == "__main__":
    app = MindscapeDashboard()
    app.mainloop()

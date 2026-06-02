import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from oscclas import simular as sim_clasico
from osccuan import simular as sim_cuantico

FIG_W, FIG_H = 5, 3.8
DPI = 100


class OsciladorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Harmonic oscillator — Numerical simulation")
        self.geometry("1500x900")

        # Default parameters.
        self.par = {
            "m": tk.DoubleVar(value=1.0),
            "omega": tk.DoubleVar(value=2.0),
            "hbar": tk.DoubleVar(value=1.0),
            "periodos": tk.IntVar(value=10),
            "q0": tk.DoubleVar(value=1.0),
            "p0": tk.DoubleVar(value=0.0),
            "dt_clas": tk.DoubleVar(value=0.05),
            "q0_q": tk.DoubleVar(value=1.5),
            "p0_q": tk.DoubleVar(value=1.0),
            "dt_q": tk.DoubleVar(value=0.05),
            "L": tk.DoubleVar(value=16.0),
            "Nx": tk.IntVar(value=1024),
        }

        self.res = None
        self._crear_widgets()

    def _crear_widgets(self):
        # Horizontal split: parameters (left) + plots (right).
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        frame_params = ttk.Frame(paned, width=300)
        paned.add(frame_params, weight=0)

        frame_plots = ttk.Frame(paned)
        paned.add(frame_plots, weight=1)

        # Scrollable parameter panel.
        canvas = tk.Canvas(frame_params, width=300)
        scrollbar = ttk.Scrollbar(frame_params, orient=tk.VERTICAL,
                                  command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(
                              scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        pf = scroll_frame
        row = 0

        def etiq(text, **kw):
            nonlocal row
            bold = kw.pop("bold", False)
            top = kw.pop("top", False)
            ttk.Label(pf, text=text,
                      font=("", 10, "bold") if bold else ("", 9)
                      ).grid(row=row, column=0, columnspan=2,
                             sticky="w", pady=(10 if top else 4, 0))
            row += 1

        def entry(var, fmt="%.3f"):
            nonlocal row
            ttk.Entry(pf, textvariable=var, width=10
                      ).grid(row=row, column=1, sticky="e", padx=5, pady=2)
            row += 1

        etiq("COMMON", bold=True, top=True)
        ttk.Label(pf, text="Mass (m)").grid(row=row, column=0, sticky="w",
                                            padx=5)
        entry(self.par["m"])
        ttk.Label(pf, text="Freq. (omega)").grid(row=row, column=0,
                                                  sticky="w", padx=5)
        entry(self.par["omega"])
        ttk.Label(pf, text="hbar (quantum)").grid(row=row, column=0,
                                                  sticky="w", padx=5)
        entry(self.par["hbar"])
        ttk.Label(pf, text="Periods").grid(row=row, column=0, sticky="w",
                                           padx=5)
        entry(self.par["periodos"], "%d")

        etiq("CLASSICAL", bold=True, top=True)
        ttk.Label(pf, text="q0").grid(row=row, column=0, sticky="w", padx=5)
        entry(self.par["q0"])
        ttk.Label(pf, text="p0").grid(row=row, column=0, sticky="w", padx=5)
        entry(self.par["p0"])
        ttk.Label(pf, text="dt").grid(row=row, column=0, sticky="w", padx=5)
        entry(self.par["dt_clas"])

        etiq("QUANTUM", bold=True, top=True)
        ttk.Label(pf, text="<q>0").grid(row=row, column=0, sticky="w",
                                        padx=5)
        entry(self.par["q0_q"])
        ttk.Label(pf, text="<p>0").grid(row=row, column=0, sticky="w",
                                        padx=5)
        entry(self.par["p0_q"])
        ttk.Label(pf, text="dt").grid(row=row, column=0, sticky="w", padx=5)
        entry(self.par["dt_q"])
        ttk.Label(pf, text="Domain L").grid(row=row, column=0, sticky="w",
                                            padx=5)
        entry(self.par["L"])
        ttk.Label(pf, text="Grid Nx").grid(row=row, column=0, sticky="w",
                                           padx=5)
        entry(self.par["Nx"], "%d")

        ttk.Button(pf, text="SIMULATE", command=self._simular,
                   style="Accent.TButton"
                   ).grid(row=row, column=0, columnspan=2,
                          pady=(20, 10), padx=5, sticky="ew")
        self.status_label = ttk.Label(pf, text="Ready", foreground="gray",
                                      font=("", 8))
        self.status_label.grid(row=row+1, column=0, columnspan=2,
                               pady=(0, 20))

        # Plot area with notebook tabs.
        self.notebook = ttk.Notebook(frame_plots)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_clas = ttk.Frame(self.notebook)
        self.tab_cuan = ttk.Frame(self.notebook)
        self.tab_comp = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_clas, text="Classical")
        self.notebook.add(self.tab_cuan, text="Quantum")
        self.notebook.add(self.tab_comp, text="Comparison")

        # Classical sub-tabs.
        self.clas_notebook = ttk.Notebook(self.tab_clas)
        self.clas_notebook.pack(fill=tk.BOTH, expand=True)
        self.clas_tabs = {}
        for name, key in [("Phase space", "fases"),
                          ("Energy error", "energia"),
                          ("Position q(t)", "posicion"),
                          ("Convergence", "convergencia")]:
            f = ttk.Frame(self.clas_notebook)
            self.clas_notebook.add(f, text=name)
            self.clas_tabs[key] = f

        # Quantum sub-tabs.
        self.cuan_notebook = ttk.Notebook(self.tab_cuan)
        self.cuan_notebook.pack(fill=tk.BOTH, expand=True)
        self.cuan_tabs = {}
        for name, key in [("Density", "densidad"),
                          ("Ehrenfest", "ehrenfest"),
                          ("Energy error", "energia"),
                          ("Norm", "norma"),
                          ("Convergence", "convergencia")]:
            f = ttk.Frame(self.cuan_notebook)
            self.cuan_notebook.add(f, text=name)
            self.cuan_tabs[key] = f

        self.comp_frame = ttk.Frame(self.tab_comp)
        self.comp_frame.pack(fill=tk.BOTH, expand=True)

        self._placeholders_clas()
        self._placeholders_cuan()
        self._placeholder_comp()

    def _placeholder_fig(self, parent, text):
        fig = Figure(figsize=(FIG_W, FIG_H), dpi=DPI)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center",
                fontsize=14, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return fig, ax, canvas

    def _placeholders_clas(self):
        self.clas_figs = {}
        for key, frame in self.clas_tabs.items():
            self.clas_figs[key] = self._placeholder_fig(frame,
                                                        "Run the simulation")

    def _placeholders_cuan(self):
        self.cuan_figs = {}
        for key, frame in self.cuan_tabs.items():
            self.cuan_figs[key] = self._placeholder_fig(frame,
                                                        "Run the simulation")

    def _placeholder_comp(self):
        self.comp_fig = self._placeholder_fig(self.comp_frame,
                                              "Run the simulation")

    def _empaquetar_fig(self, parent, fig):
        for w in parent.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _simular(self):
        self.status_label.configure(text="Simulating…", foreground="orange")
        self.update_idletasks()

        p = {k: v.get() for k, v in self.par.items()}
        try:
            rc = sim_clasico(m=p["m"], omega=p["omega"], q0=p["q0"],
                             p0=p["p0"], dt=p["dt_clas"],
                             periodos=p["periodos"])
            rq = sim_cuantico(m=p["m"], omega=p["omega"], hbar=p["hbar"],
                              q0=p["q0_q"], p0=p["p0_q"], dt=p["dt_q"],
                              periodos=p["periodos"], L=p["L"], Nx=p["Nx"])
            self.res = (rc, rq)
            self._plot_clasico(rc)
            self._plot_cuantico(rq)
            self._plot_comparacion(rc, rq)
            self.status_label.configure(
                text=f"OK  (classical: dt={p['dt_clas']}, quantum: dt={p['dt_q']})",
                foreground="green")
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", foreground="red")
            import traceback
            traceback.print_exc()

    # ─────────── CLASSICAL PLOTS ───────────
    def _plot_clasico(self, r):
        dt = self.par["dt_clas"].get()
        omega = self.par["omega"].get()
        n_c = int(4 * 2 * np.pi / omega / dt)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.plot(r["q_ex"][:n_c], r["p_ex"][:n_c], "k-", lw=1.5,
                label="Exact")
        ax.plot(r["q_eu"][:n_c], r["p_eu"][:n_c], "b--", lw=1.2,
                label="Symplectic Euler")
        ax.plot(r["q_vl"][:n_c], r["p_vl"][:n_c], "r-.", lw=1.2,
                label="Verlet")
        ax.plot(r["q_ns"][:n_c], r["p_ns"][:n_c], "g:", lw=2,
                label="Explicit Euler")
        ax.set_xlabel("$q$"); ax.set_ylabel("$p$")
        ax.set_title("Phase space (4 periods)")
        ax.legend(fontsize=8); ax.axis("equal")
        fig.tight_layout()
        self._empaquetar_fig(self.clas_tabs["fases"], fig)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        H_ex, H0 = r["H_ex"], r["H0"]
        ax.plot(r["t"], (r["H_eu"] - H_ex) / H0, "b", lw=0.8,
                label="Symplectic Euler")
        ax.plot(r["t"], (r["H_vl"] - H_ex) / H0, "r", lw=0.8,
                label="Verlet")
        ax.plot(r["t"], (r["H_ns"] - H_ex) / H0, "g", lw=0.8,
                label="Explicit Euler")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("$t$"); ax.set_ylabel(r"$(H - H_0)/H_0$")
        ax.set_title("Relative energy error")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._empaquetar_fig(self.clas_tabs["energia"], fig)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.plot(r["t"], r["q_ex"], "k-", lw=1.5, label="Exact")
        ax.plot(r["t"], r["q_eu"], "b--", lw=1, label="Symplectic Euler")
        ax.plot(r["t"], r["q_vl"], "r-.", lw=1, label="Verlet")
        ax.set_xlabel("$t$"); ax.set_ylabel("$q(t)$")
        ax.set_title("Position evolution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._empaquetar_fig(self.clas_tabs["posicion"], fig)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        dts = r["dts"]
        ax.loglog(dts, r["err_eu"], "bo-",
                  label=f"Symplectic Euler (order ~{r['c_eu'][0]:.2f})")
        ax.loglog(dts, r["err_vl"], "rs-",
                  label=f"Verlet (order ~{r['c_vl'][0]:.2f})")
        ax.loglog(dts, dts**1, "k:", label="O(h)")
        ax.loglog(dts, dts**2, "k--", label="O(h²)")
        ax.set_xlabel(r"$\Delta t$"); ax.set_ylabel("Max error in $q$")
        ax.set_title("Convergence — classical")
        ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        self._empaquetar_fig(self.clas_tabs["convergencia"], fig)

    # ─────────── QUANTUM PLOTS ───────────
    def _plot_cuantico(self, r):
        T = self.par["periodos"].get() * 2 * np.pi / self.par["omega"].get()

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.plot(r["x"], np.abs(r["psi0"])**2, "k-", lw=1.5, label="$t=0$")
        ax.plot(r["x"], np.abs(r["psi_f_lt"])**2, "b--", lw=1,
                label=f"$t={T:.1f}$ (Lie)")
        ax.plot(r["x"], np.abs(r["psi_f_st"])**2, "r-.", lw=1,
                label=f"$t={T:.1f}$ (Strang)")
        ax.set_xlabel("$q$"); ax.set_ylabel(r"$|\psi(q)|^2$")
        ax.set_title("Probability density")
        ax.legend(fontsize=8); ax.set_xlim(-6, 6)
        fig.tight_layout()
        self._empaquetar_fig(self.cuan_tabs["densidad"], fig)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.plot(r["t"], r["q_lt"], "b--", lw=1,
                label=r"$\langle q \rangle$ Lie (ord. 1)")
        ax.plot(r["t"], r["q_st"], "r-.", lw=1,
                label=r"$\langle q \rangle$ Strang (ord. 2)")
        ax.plot(r["t"], r["q_cl"], "k-", lw=1, label="Classical (Ehrenfest)")
        ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle \hat{q} \rangle$")
        ax.set_title("Expectation value (Ehrenfest)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._empaquetar_fig(self.cuan_tabs["ehrenfest"], fig)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.plot(r["t"], (r["H_lt"] - r["H_lt"][0]) / r["H_lt"][0],
                "b", lw=0.8, label="Lie–Trotter (ord. 1)")
        ax.plot(r["t"], (r["H_st"] - r["H_st"][0]) / r["H_st"][0],
                "r", lw=0.8, label="Strang (ord. 2)")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\Delta\langle\hat{H}\rangle/\langle\hat{H}\rangle_0$")
        ax.set_title("Relative energy error")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._empaquetar_fig(self.cuan_tabs["energia"], fig)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.plot(r["t"], r["norm_lt"] - 1, "b", lw=0.8,
                label="Lie–Trotter")
        ax.plot(r["t"], r["norm_st"] - 1, "r--", lw=0.8, label="Strang")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("$t$"); ax.set_ylabel(r"$\|\psi\|^2 - 1$")
        ax.set_title("Norm error (unitarity)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._empaquetar_fig(self.cuan_tabs["norma"], fig)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        dts_c = r["dts_c"]
        ax.loglog(dts_c, r["err_lt"], "bo-",
                  label=f"Lie–Trotter (order ~{r['c_lt'][0]:.2f})")
        ax.loglog(dts_c, r["err_st"], "rs-",
                  label=f"Strang (order ~{r['c_st'][0]:.2f})")
        ax.loglog(dts_c, dts_c**1, "k:", label="O(h)")
        ax.loglog(dts_c, dts_c**2, "k--", label="O(h²)")
        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel("Max error in $\langle q \\rangle$")
        ax.set_title("Convergence — quantum")
        ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        self._empaquetar_fig(self.cuan_tabs["convergencia"], fig)

    # ─────────── COMPARISON ───────────
    def _plot_comparacion(self, rc, rq):
        dts_clas = rc["dts"]
        dts_cuan = rq["dts_c"]
        f_lt = interp1d(np.log(dts_cuan), np.log(rq["err_lt"]),
                        kind="linear", fill_value="extrapolate")
        f_st = interp1d(np.log(dts_cuan), np.log(rq["err_st"]),
                        kind="linear", fill_value="extrapolate")
        err_lt_i = np.exp(f_lt(np.log(dts_clas)))
        err_st_i = np.exp(f_st(np.log(dts_clas)))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(dts_clas, rc["err_eu"], "bo-",
                  label="Symplectic Euler (classical, ord. 1)")
        ax.loglog(dts_clas, rc["err_vl"], "rs-",
                  label="Verlet (classical, ord. 2)")
        ax.loglog(dts_clas, err_lt_i, "D--", color="orange",
                  label="Lie–Trotter (quantum, ord. 1)")
        ax.loglog(dts_clas, err_st_i, "v--", color="purple",
                  label="Strang (quantum, ord. 2)")
        ax.loglog(dts_clas, dts_clas**1, "k:", label="O(h)")
        ax.loglog(dts_clas, dts_clas**2, "k--", label="O(h²)")
        ax.set_xlabel(r"$\Delta t$"); ax.set_ylabel("Max error")
        ax.set_title("Convergence: classical vs. quantum")
        ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()

        for w in self.comp_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.comp_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    app = OsciladorApp()
    app.mainloop()

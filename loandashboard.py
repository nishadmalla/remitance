"""
=============================================================
  AI Remittance Loan Viability Dashboard  v2
  Redesigned UI — Tkinter + Scikit-Learn
=============================================================
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  TRAIN MODELS
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
N = 1000

def generate_data(n):
    ya  = np.round(np.random.uniform(0.5, 12, n), 1)
    rem = np.random.randint(12000, 65000, n)
    fs  = np.random.randint(2, 8, n)
    exp = fs * np.random.randint(3500, 6500, n)
    el  = np.random.choice([0, 1, 2], n, p=[0.55, 0.35, 0.10])
    sav = np.random.randint(0, 80000, n)
    ba  = np.random.randint(1, 120, n)
    ds  = np.random.randint(1, 90, n)
    rf  = np.random.choice([15, 30, 45, 60], n, p=[0.10, 0.55, 0.25, 0.10])
    co  = np.clip(np.random.uniform(0.4, 1.0, n), 0.4, 1.0)
    sur = rem - exp
    dr  = np.round(el * 15000 / np.clip(rem, 1, None), 2)
    st  = np.clip((ds/rf)*0.4 + (exp/np.clip(rem,1,None))*0.4 + (el/3)*0.2, 0, 1)
    X   = np.column_stack([ya,rem,fs,exp,el,sav,ba,ds,rf,co,sur,dr,st])
    sc  = 0.28*co + 0.25*(1-dr/3) + 0.22*np.clip(sur/30000,-1,1) + 0.15*(ya/12) + 0.10*(sav/80000)
    y   = (sc + np.random.normal(0,0.06,n) > 0.38).astype(int)
    return X, y

X, y = generate_data(N)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)

MODELS = {
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}
ACC = {}
for nm, m in MODELS.items():
    if nm == "Logistic Regression":
        m.fit(Xtr_s, ytr); ACC[nm] = round(accuracy_score(yte, m.predict(Xte_s))*100, 1)
    else:
        m.fit(Xtr, ytr);   ACC[nm] = round(accuracy_score(yte, m.predict(Xte))*100, 1)

# ─────────────────────────────────────────────────────────────
#  PALETTE
# ─────────────────────────────────────────────────────────────
BG        = "#F4F3EF"
SIDEBAR   = "#1E1E2E"
CARD      = "#FFFFFF"
CARD2     = "#F9F8F5"
BORDER    = "#E5E3DC"
TXT1      = "#1A1A2E"
TXT2      = "#6B6A63"
TXT3      = "#9B9A93"
PURPLE    = "#7F77DD"
PURPLE_D  = "#534AB7"
TEAL      = "#1D9E75"
AMBER     = "#EF9F27"
CORAL     = "#D85A30"
BLUE      = "#378ADD"
GREEN     = "#1D9E75"
RED       = "#D85A30"
SLIDER_BG = "#E8E6F0"
SLIDER_FG = "#7F77DD"

# ─────────────────────────────────────────────────────────────
#  CUSTOM SLIDER WIDGET
# ─────────────────────────────────────────────────────────────
class SmartSlider(tk.Frame):
    """A polished slider with icon, label, live value box, and +/- buttons."""

    def __init__(self, parent, label, icon, from_, to_, default,
                 fmt="{:.0f}", unit="", color=PURPLE, step=1, **kw):
        super().__init__(parent, bg=CARD, **kw)
        self.fmt   = fmt
        self.unit  = unit
        self.step  = step
        self.from_ = from_
        self.to_   = to_
        self.var   = tk.DoubleVar(value=default)
        self.color = color

        # ── row 1: icon + label + value box ──
        top = tk.Frame(self, bg=CARD)
        top.pack(fill="x", pady=(8, 2))

        tk.Label(top, text=icon, bg=CARD, fg=color,
                 font=("Segoe UI Emoji", 12)).pack(side="left", padx=(0, 5))
        tk.Label(top, text=label, bg=CARD, fg=TXT1,
                 font=("Segoe UI", 9, "bold")).pack(side="left")

        # Value entry (typed input)
        self.entry_var = tk.StringVar(value=self._fmt(default))
        self.entry = tk.Entry(top, textvariable=self.entry_var, width=9,
                              font=("Segoe UI", 9, "bold"), fg=color,
                              bg=CARD2, relief="flat",
                              highlightthickness=1,
                              highlightbackground=BORDER,
                              highlightcolor=color,
                              justify="right")
        self.entry.pack(side="right", ipady=2)
        if unit:
            tk.Label(top, text=unit, bg=CARD, fg=TXT3,
                     font=("Segoe UI", 8)).pack(side="right", padx=(0, 3))

        # ── row 2: minus + track + plus ──
        mid = tk.Frame(self, bg=CARD)
        mid.pack(fill="x", pady=(0, 4))

        btn_style = dict(bg=SLIDER_BG, fg=color, relief="flat",
                         font=("Segoe UI", 10, "bold"),
                         width=2, cursor="hand2",
                         activebackground=color, activeforeground="white")
        tk.Button(mid, text="−", command=self._dec, **btn_style).pack(side="left")

        self.track = tk.Canvas(mid, height=22, bg=CARD, bd=0,
                               highlightthickness=0)
        self.track.pack(side="left", fill="x", expand=True, padx=4)
        self.track.bind("<Configure>",   self._redraw)
        self.track.bind("<Button-1>",    self._click)
        self.track.bind("<B1-Motion>",   self._drag)

        tk.Button(mid, text="+", command=self._inc, **btn_style).pack(side="right")

        # ── range labels ──
        bot = tk.Frame(self, bg=CARD)
        bot.pack(fill="x", pady=(0, 6))
        tk.Label(bot, text=self._fmt(from_), bg=CARD, fg=TXT3,
                 font=("Segoe UI", 7)).pack(side="left", padx=(28, 0))
        tk.Label(bot, text=self._fmt(to_), bg=CARD, fg=TXT3,
                 font=("Segoe UI", 7)).pack(side="right", padx=(0, 28))

        # ── divider ──
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        self.var.trace_add("write", self._on_var_change)
        self.entry.bind("<Return>",    self._on_entry)
        self.entry.bind("<FocusOut>",  self._on_entry)

    def _fmt(self, v):
        s = self.fmt.format(float(v))
        # add thousands separator for large numbers
        try:
            n = float(s)
            if n >= 1000 and "." not in self.fmt:
                return f"{int(n):,}"
        except:
            pass
        return s

    def get(self):
        return self.var.get()

    def _frac(self):
        r = self.to_ - self.from_
        return (self.var.get() - self.from_) / r if r else 0

    def _redraw(self, *_):
        self.track.delete("all")
        w = self.track.winfo_width()
        h = 22
        r = 6
        # track background
        self.track.create_rounded_rect = lambda x1,y1,x2,y2,r,**kw: \
            self.track.create_polygon(
                x1+r,y1, x2-r,y1, x2,y1+r, x2,y2-r, x2-r,y2, x1+r,y2, x1,y2-r, x1,y1+r,
                smooth=True, **kw)
        cy = h // 2
        self.track.create_rectangle(0, cy-3, w, cy+3, fill=SLIDER_BG, outline="", width=0)
        # filled portion
        fx = max(12, int(self._frac() * (w - 24)) + 12)
        self.track.create_rectangle(12, cy-3, fx, cy+3, fill=self.color, outline="", width=0)
        # thumb
        self.track.create_oval(fx-9, cy-9, fx+9, cy+9, fill=self.color, outline=CARD, width=2)

    def _px_to_val(self, x):
        w = self.track.winfo_width()
        frac = max(0, min(1, (x - 12) / max(w - 24, 1)))
        raw  = self.from_ + frac * (self.to_ - self.from_)
        return round(round(raw / self.step) * self.step, 10)

    def _click(self, e):
        self.var.set(self._px_to_val(e.x))
    def _drag(self, e):
        self.var.set(self._px_to_val(e.x))

    def _inc(self):
        self.var.set(min(self.to_, self.var.get() + self.step))
    def _dec(self):
        self.var.set(max(self.from_, self.var.get() - self.step))

    def _on_var_change(self, *_):
        self.entry_var.set(self._fmt(self.var.get()))
        self._redraw()

    def _on_entry(self, *_):
        try:
            raw = float(self.entry_var.get().replace(",", ""))
            self.var.set(max(self.from_, min(self.to_, raw)))
        except ValueError:
            self.entry_var.set(self._fmt(self.var.get()))


# ─────────────────────────────────────────────────────────────
#  SECTION HEADER HELPER
# ─────────────────────────────────────────────────────────────
def section_header(parent, icon, title, bg=CARD):
    f = tk.Frame(parent, bg=bg)
    f.pack(fill="x", pady=(14, 4), padx=16)
    tk.Label(f, text=icon, bg=bg, fg=PURPLE,
             font=("Segoe UI Emoji", 13)).pack(side="left", padx=(0, 6))
    tk.Label(f, text=title, bg=bg, fg=TXT1,
             font=("Segoe UI", 10, "bold")).pack(side="left")
    tk.Frame(parent, bg=PURPLE, height=2).pack(fill="x", padx=16, pady=(0, 6))


# ─────────────────────────────────────────────────────────────
#  METRIC CARD
# ─────────────────────────────────────────────────────────────
def metric_card(parent, label, var, color, width=None):
    kw = {"width": width} if width else {}
    outer = tk.Frame(parent, bg=BORDER, **kw)
    inner = tk.Frame(outer, bg=CARD, padx=14, pady=10)
    inner.pack(fill="both", padx=1, pady=1, expand=True)
    tk.Label(inner, text=label.upper(), bg=CARD, fg=TXT3,
             font=("Segoe UI", 7, "bold")).pack(anchor="w")
    tk.Label(inner, textvariable=var, bg=CARD, fg=color,
             font=("Segoe UI", 18, "bold")).pack(anchor="w", pady=(2, 0))
    return outer, var


# ─────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────
class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Loan Viability Dashboard — Remittance Intelligence System")
        self.geometry("1340x860")
        self.minsize(1100, 720)
        self.configure(bg=BG)

        self.sel_model = tk.StringVar(value="Random Forest")
        self.v_verdict = tk.StringVar(value="Awaiting Input")
        self.v_prob    = tk.StringVar(value="—")
        self.v_loan    = tk.StringVar(value="—")
        self.v_emi     = tk.StringVar(value="—")
        self.v_risk    = tk.StringVar(value="—")
        self.v_zero    = tk.StringVar(value="—")

        self._setup_styles()
        self._build()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Sidebar.TFrame", background=SIDEBAR)

    # ──────────────────────────────────────────
    def _build(self):
        # ── HEADER ────────────────────────────
        hdr = tk.Frame(self, bg=SIDEBAR, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="🏦", bg=SIDEBAR,
                 font=("Segoe UI Emoji", 18)).pack(side="left", padx=(20, 8), pady=10)
        title_f = tk.Frame(hdr, bg=SIDEBAR)
        title_f.pack(side="left", pady=10)
        tk.Label(title_f, text="AI Remittance Loan Viability Dashboard",
                 bg=SIDEBAR, fg="white",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(title_f, text="Nepal Bank — Internship Presentation Project",
                 bg=SIDEBAR, fg="#8E8CB0",
                 font=("Segoe UI", 9)).pack(anchor="w")

        # model accuracy pills in header
        pill_f = tk.Frame(hdr, bg=SIDEBAR)
        pill_f.pack(side="right", padx=20)
        for nm, ac in ACC.items():
            short = nm.split()[0]
            p = tk.Frame(pill_f, bg="#2D2D45", padx=10, pady=4)
            p.pack(side="left", padx=4)
            tk.Label(p, text=f"{short}  {ac}%", bg="#2D2D45", fg="#A8A6D0",
                     font=("Segoe UI", 8, "bold")).pack()

        # ── BODY ──────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Left panel (scrollable form)
        left_outer = tk.Frame(body, bg=CARD, width=360)
        left_outer.pack(side="left", fill="y")
        left_outer.pack_propagate(False)

        canvas_scroll = tk.Canvas(left_outer, bg=CARD, bd=0,
                                  highlightthickness=0, width=356)
        scrollbar = tk.Scrollbar(left_outer, orient="vertical",
                                 command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)

        self.form = tk.Frame(canvas_scroll, bg=CARD, padx=0)
        form_win  = canvas_scroll.create_window((0, 0), window=self.form,
                                                anchor="nw", width=356)
        self.form.bind("<Configure>",
                       lambda e: canvas_scroll.configure(
                           scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.bind("<MouseWheel>",
                           lambda e: canvas_scroll.yview_scroll(
                               int(-1*(e.delta/120)), "units"))

        self._build_form(self.form)

        # Right panel
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=16, pady=12)
        self._build_results(right)

    # ──────────────────────────────────────────
    def _build_form(self, parent):
        # ── ALGORITHM ─────────────────────────
        section_header(parent, "🤖", "Algorithm")
        algo_f = tk.Frame(parent, bg=CARD, padx=16)
        algo_f.pack(fill="x")
        for nm in MODELS:
            row = tk.Frame(algo_f, bg=CARD)
            row.pack(fill="x", pady=3)
            rb = tk.Radiobutton(row, text=nm, variable=self.sel_model, value=nm,
                                bg=CARD, fg=TXT1, selectcolor=PURPLE,
                                activebackground=CARD, font=("Segoe UI", 9),
                                command=self._on_change)
            rb.pack(side="left")
            tk.Label(row, text=f"✓ {ACC[nm]}% acc", bg=CARD, fg=TEAL,
                     font=("Segoe UI", 8, "bold")).pack(side="right")

        # ── MIGRANT PROFILE ───────────────────
        section_header(parent, "✈️", "Migrant Profile")
        sp = tk.Frame(parent, bg=CARD, padx=16)
        sp.pack(fill="x")

        self.s_years   = SmartSlider(sp, "Years Abroad",            "📅", 1,  12,    3,   fmt="{:.1f}", unit="yrs",  color=PURPLE, step=0.5)
        self.s_remit   = SmartSlider(sp, "Monthly Remit",           "💵", 5000, 70000, 28000, unit="NPR",  color=TEAL,   step=500)
        self.s_freq    = SmartSlider(sp, "Remit Frequency",         "🔄", 15, 60,   30,  unit="days", color=BLUE,   step=15)
        self.s_days    = SmartSlider(sp, "Days Since Last Transfer","⏳", 1,  90,   20,  unit="days", color=AMBER,  step=1)
        self.s_consist = SmartSlider(sp, "Consistency Score",       "📈", 0,  100,  75,  unit="%",    color=TEAL,   step=1)
        for w in [self.s_years, self.s_remit, self.s_freq, self.s_days, self.s_consist]:
            w.pack(fill="x")

        # ── FAMILY FINANCES ───────────────────
        section_header(parent, "🏠", "Family Finances")
        fp = tk.Frame(parent, bg=CARD, padx=16)
        fp.pack(fill="x")

        self.s_fsize   = SmartSlider(fp, "Family Size",             "👨‍👩‍👧", 2,  8,    4,   unit="ppl",  color=BLUE,   step=1)
        self.s_expense = SmartSlider(fp, "Monthly Expenses",        "🧾", 3000, 50000, 18000, unit="NPR",  color=CORAL,  step=500)
        self.s_savings = SmartSlider(fp, "Savings Balance",         "💰", 0, 80000, 15000, unit="NPR",  color=TEAL,   step=1000)
        self.s_loans   = SmartSlider(fp, "Existing Loans",          "🏦", 0,  2,    0,   unit="",     color=CORAL,  step=1)
        self.s_bank_age= SmartSlider(fp, "Bank Account Age",        "🗓️", 1, 120,  24,  unit="mo",   color=BLUE,   step=1)
        for w in [self.s_fsize, self.s_expense, self.s_savings, self.s_loans, self.s_bank_age]:
            w.pack(fill="x")

        # ── PREDICT BUTTON ────────────────────
        btn_f = tk.Frame(parent, bg=CARD, padx=16, pady=14)
        btn_f.pack(fill="x")
        self.btn = tk.Button(btn_f, text="⚡   RUN PREDICTION",
                             bg=PURPLE, fg="white",
                             font=("Segoe UI", 11, "bold"),
                             relief="flat", cursor="hand2",
                             pady=12, activebackground=PURPLE_D,
                             activeforeground="white",
                             command=self._predict)
        self.btn.pack(fill="x")
        tk.Label(btn_f, text="or press  Enter", bg=CARD, fg=TXT3,
                 font=("Segoe UI", 8)).pack(pady=(4, 0))
        self.bind("<Return>", lambda e: self._predict())

    # ──────────────────────────────────────────
    def _build_results(self, parent):
        # ── VERDICT BANNER ────────────────────
        self.banner = tk.Frame(parent, bg=CARD, height=90)
        self.banner.pack(fill="x", pady=(0, 12))
        self.banner.pack_propagate(False)
        banner_border = tk.Frame(self.banner, bg=BORDER)
        banner_border.place(relwidth=1, relheight=1)
        banner_inner = tk.Frame(banner_border, bg=CARD)
        banner_inner.place(x=1, y=1, relwidth=1, relheight=1, width=-2, height=-2)

        # accent bar
        self.accent = tk.Frame(banner_inner, bg=TXT3, width=6)
        self.accent.pack(side="left", fill="y")

        btext = tk.Frame(banner_inner, bg=CARD, padx=18)
        btext.pack(side="left", fill="both", expand=True, pady=16)
        self.lbl_verdict = tk.Label(btext, textvariable=self.v_verdict,
                                    bg=CARD, fg=TXT3,
                                    font=("Segoe UI", 18, "bold"))
        self.lbl_verdict.pack(anchor="w")
        self.lbl_sub = tk.Label(btext, text="Adjust inputs and click  ⚡ RUN PREDICTION",
                                bg=CARD, fg=TXT3, font=("Segoe UI", 9))
        self.lbl_sub.pack(anchor="w")

        bprob = tk.Frame(banner_inner, bg=CARD, padx=24)
        bprob.pack(side="right", pady=16)
        self.lbl_prob = tk.Label(bprob, textvariable=self.v_prob,
                                 bg=CARD, fg=TXT3,
                                 font=("Segoe UI", 32, "bold"))
        self.lbl_prob.pack()
        tk.Label(bprob, text="repayment probability", bg=CARD, fg=TXT3,
                 font=("Segoe UI", 8)).pack()

        # ── METRIC CARDS ROW ──────────────────
        mrow = tk.Frame(parent, bg=BG)
        mrow.pack(fill="x", pady=(0, 12))
        defs = [
            ("Loan Amount (NPR)", self.v_loan, PURPLE),
            ("EMI Period",        self.v_emi,  TEAL),
            ("Risk Level",        self.v_risk, AMBER),
            ("Zero Interest",     self.v_zero, BLUE),
        ]
        for i, (lbl, var, col) in enumerate(defs):
            card, _ = metric_card(mrow, lbl, var, col)
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 8 if i < 3 else 0))
            mrow.columnconfigure(i, weight=1)

        # ── CHARTS ────────────────────────────
        chart_outer = tk.Frame(parent, bg=CARD, bd=0)
        chart_outer.pack(fill="both", expand=True)
        chart_border = tk.Frame(chart_outer, bg=BORDER)
        chart_border.pack(fill="both", expand=True)
        chart_inner = tk.Frame(chart_border, bg=CARD)
        chart_inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.fig = Figure(facecolor=CARD)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_inner)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._placeholder_chart()

    # ──────────────────────────────────────────
    def _on_change(self):
        pass

    def _predict(self):
        ya   = self.s_years.get()
        rem  = self.s_remit.get()
        fs   = self.s_fsize.get()
        exp  = self.s_expense.get()
        el   = self.s_loans.get()
        sav  = self.s_savings.get()
        ba   = self.s_bank_age.get()
        ds   = self.s_days.get()
        rf   = self.s_freq.get()
        co   = self.s_consist.get() / 100.0
        sur  = rem - exp
        dr   = round(el * 15000 / max(rem, 1), 2)
        st   = min(1.0, (ds/max(rf,1))*0.4 + (exp/max(rem,1))*0.4 + (el/3)*0.2)
        Xinp = np.array([[ya, rem, fs, exp, el, sav, ba, ds, rf, co, sur, dr, st]])

        probs_all = {}
        for nm, m in MODELS.items():
            Xi = scaler.transform(Xinp) if nm == "Logistic Regression" else Xinp
            probs_all[nm] = round(m.predict_proba(Xi)[0][1], 4)

        sel  = self.sel_model.get()
        prob = probs_all[sel]
        ok   = prob >= 0.5

        # recommendations
        if ok:
            loan = int(min(rem * max(0.8, prob), 80000))
            emi  = 3 if loan > 40000 else 2 if loan > 20000 else 1
            zero = co > 0.80 and el == 0 and ba > 12
            risk = "Low ✓" if prob > 0.80 else "Medium"
        else:
            loan, emi, zero = 0, 0, False
            risk = "High ✗" if prob < 0.30 else "Medium-High"

        # update banner
        c = GREEN if ok else RED
        self.accent.config(bg=c)
        self.lbl_verdict.config(
            text="✅  LOAN APPROVED" if ok else "❌  LOAN REJECTED", fg=c)
        self.lbl_sub.config(
            text=f"{sel}  •  {prob*100:.1f}% repayment confidence  •  "
                 f"Surplus NPR {int(sur):,}/mo", fg=TXT2)
        self.lbl_prob.config(text=f"{prob*100:.1f}%", fg=c)

        self.v_prob.set(f"{prob*100:.1f}%")
        self.v_loan.set(f"{loan:,}" if loan else "N/A")
        self.v_emi.set(f"{emi} mo" if emi else "N/A")
        self.v_risk.set(risk)
        self.v_zero.set("Yes ✓" if zero else "No")

        self._draw_charts(prob, probs_all, st, sur, dr, co, ok)

    # ──────────────────────────────────────────
    def _placeholder_chart(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(CARD)
        ax.text(0.5, 0.5, "Run a prediction to see analytics",
                ha="center", va="center", color=TXT3, fontsize=13,
                transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw()

    def _draw_charts(self, prob, probs_all, stress, surplus, debt_r, consist, ok):
        self.fig.clear()
        self.fig.patch.set_facecolor(CARD)
        gs = self.fig.add_gridspec(1, 3, wspace=0.38,
                                   left=0.05, right=0.97, top=0.88, bottom=0.14)
        vc = GREEN if ok else RED

        # ── Chart 1: Donut gauge ──────────────
        ax1 = self.fig.add_subplot(gs[0])
        sizes  = [prob, 1-prob]
        colors = [vc, "#EDECE8"]
        ax1.pie(sizes, colors=colors, startangle=90,
                wedgeprops=dict(width=0.42, edgecolor=CARD, linewidth=3))
        ax1.text(0, 0.10, f"{prob*100:.1f}%", ha="center", va="center",
                 fontsize=17, fontweight="bold", color=vc)
        ax1.text(0, -0.22, "Repayment\nProbability", ha="center", va="center",
                 fontsize=8.5, color=TXT2)
        ax1.set_title("Prediction Score", fontsize=10, fontweight="bold",
                      color=TXT1, pad=10)
        ax1.set_facecolor(CARD)

        # ── Chart 2: Risk factors ─────────────
        ax2 = self.fig.add_subplot(gs[1])
        labels = ["Repayment prob", "Consistency", "Low stress", "Low debt ratio"]
        vals   = [prob, consist, 1-stress, max(0, 1-min(debt_r,1))]
        bar_c  = [GREEN if v > 0.65 else AMBER if v > 0.40 else CORAL for v in vals]
        y_pos  = np.arange(len(labels))

        # background track bars
        ax2.barh(y_pos, [1]*4, color="#EDECE8", height=0.55, zorder=1)
        bars = ax2.barh(y_pos, vals, color=bar_c, height=0.55,
                        edgecolor=CARD, linewidth=1.5, zorder=2)
        ax2.set_xlim(0, 1.15)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=8.5, color=TXT1)
        ax2.axvline(0.5, color=TXT3, linestyle="--", linewidth=0.8, zorder=3)
        ax2.set_title("Risk Factor Analysis", fontsize=10, fontweight="bold",
                      color=TXT1, pad=10)
        ax2.tick_params(axis="x", labelsize=7.5, colors=TXT2)
        ax2.set_facecolor(CARD)
        ax2.spines[["top","right","left","bottom"]].set_visible(False)
        ax2.tick_params(left=False)
        for bar, v in zip(bars, vals):
            ax2.text(v + 0.03, bar.get_y() + bar.get_height()/2,
                     f"{v:.0%}", va="center", fontsize=8,
                     fontweight="bold", color=TXT1)

        # ── Chart 3: All model comparison ─────
        ax3 = self.fig.add_subplot(gs[2])
        names   = list(probs_all.keys())
        pvals   = list(probs_all.values())
        x       = np.arange(len(names))
        bcolors = [GREEN if p >= 0.5 else RED for p in pvals]
        bars3   = ax3.bar(x, pvals, color=bcolors, width=0.5,
                          edgecolor=CARD, linewidth=2, zorder=2)

        # background
        ax3.bar(x, [1]*len(names), color="#EDECE8", width=0.5, zorder=1)
        ax3.bar(x, pvals, color=bcolors, width=0.5,
                edgecolor=CARD, linewidth=2, zorder=2)
        ax3.axhline(0.5, color=TXT3, linestyle="--", linewidth=0.9, zorder=3,
                    label="Threshold 50%")

        short = ["Random\nForest", "Gradient\nBoosting", "Logistic\nRegression"]
        ax3.set_xticks(x)
        ax3.set_xticklabels(short, fontsize=8, color=TXT1)
        ax3.set_ylim(0, 1.12)
        ax3.set_title("Algorithm Comparison", fontsize=10, fontweight="bold",
                      color=TXT1, pad=10)
        ax3.set_ylabel("Repayment probability", fontsize=8, color=TXT2)
        ax3.tick_params(axis="y", labelsize=7.5, colors=TXT2)
        ax3.set_facecolor(CARD)
        ax3.spines[["top","right","left","bottom"]].set_visible(False)
        ax3.tick_params(bottom=False)
        ax3.legend(fontsize=7, frameon=False, labelcolor=TXT2)
        for bar, v in zip(bars3, pvals):
            ax3.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                     f"{v:.0%}", ha="center", fontsize=9,
                     fontweight="bold", color=TXT1)

        self.canvas.draw()


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()
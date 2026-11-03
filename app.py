import csv, datetime, os
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from core import content as C
from core.gd import *

st.set_page_config(page_title="Gradient Descent - Virtual Lab", page_icon="📉", layout="wide")
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#171d2a",
    "axes.edgecolor": "#65708a",
    "axes.labelcolor": "#e6eaf2",
    "axes.titlecolor": "#e6eaf2",
    "text.color": "#e6eaf2",
    "xtick.color": "#c4ccda",
    "ytick.color": "#c4ccda",
    "grid.color": "#394256",
})
st.markdown("""<style>
.vl-head{background:#1b2a4a;color:#f4f6fb;padding:.7rem 1.2rem;border:1px solid #34466d;border-radius:6px;margin-bottom:1rem}
.vl-head small{opacity:.9}.vl-foot{border-top:1px solid #394256;margin-top:3rem;padding-top:.8rem;font-size:.85rem;color:#aab4c5}
</style>""", unsafe_allow_html=True)

SECTIONS = ["Aim", "Theory", "Pretest", "Procedure", "Simulation", "Posttest", "References", "Contributors", "Feedback"]
st.markdown('<div class="vl-head"><b>Machine Learning Virtual Lab</b><br><small>Computer Science and Engineering › Machine Learning Lab › Experiments › Experiment 2</small></div>', unsafe_allow_html=True)
sec = st.sidebar.radio("Gradient Descent", SECTIONS)
st.title("Gradient Descent: Convergence, Learning Rate and Optimization Performance")


@st.cache_data
def prepared(uploaded_bytes):
    import io
    df, src = load_bank(io.BytesIO(uploaded_bytes) if uploaded_bytes else None)
    Xs, mu, sd = preprocess(df)
    return df, src, Xs


def quiz(key, qs):
    with st.form(key):
        picks = [st.radio(f"Q{i+1}. {q}", o, index=None, key=f"{key}{i}") for i, (q, o, _, _) in enumerate(qs)]
        done = st.form_submit_button("Submit")
    if done:
        score = sum(p == q[1][q[2]] for p, q in zip(picks, qs))
        st.success(f"Score: {score} / {len(qs)}")
        for i, (q, o, a, e) in enumerate(qs):
            st.markdown(f"**Q{i+1}** {'✅' if picks[i] == o[a] else '❌'} Answer: *{o[a]}*. {e}")


def curve(ax, ys, labels=None, **kw):
    for i, y in enumerate(ys):
        ax.plot(y, label=None if labels is None else labels[i], **kw)
    ax.set_xlabel("Iteration"); ax.grid(True)


if sec == "Aim":
    st.header("Aim"); st.markdown(C.AIM)
    st.info("Pipeline: Dataset → Objective Function → Gradient → Parameter Update → Convergence")
    st.markdown(C.DATASET)
elif sec == "Theory":
    st.header("Theory")
    for t in (C.THEORY_GRADIENT, C.THEORY_GD, C.THEORY_CONV):
        st.markdown(t)
elif sec == "Pretest":
    st.header("Pretest"); quiz("pre", C.PRETEST)
elif sec == "Procedure":
    st.header("Procedure")
    for i, s in enumerate(C.PROCEDURE, 1):
        st.markdown(f"{i}. {s}")
elif sec == "Posttest":
    st.header("Posttest"); quiz("post", C.POSTTEST)
elif sec == "References":
    st.header("References")
    for r in C.REFERENCES:
        st.markdown(f"- {r}")
elif sec == "Contributors":
    st.header("Contributors")
    st.markdown("Apurva Gokhale | 15  \nYuvraj Kumbhavdekar | 26  \nPriyal Sawant | 56")
elif sec == "Feedback":
    st.header("Feedback")
    with st.form("fb"):
        name = st.text_input("Name"); rating = st.slider("Rating", 1, 5, 4); msg = st.text_area("Comments")
        if st.form_submit_button("Submit"):
            new = not os.path.exists("feedback.csv")
            with open("feedback.csv", "a", newline="") as f:
                w = csv.writer(f)
                if new: w.writerow(["time", "name", "rating", "comments"])
                w.writerow([datetime.datetime.now().isoformat(), name, rating, msg])
            st.success("Thank you!")

elif sec == "Simulation":
    st.header("Simulation")
    up = st.file_uploader("Optional: upload bank-full.csv (';' separated)", type="csv")
    df, src, Xs = prepared(up.getvalue() if up else None)
    st.caption(f"Data source: {src}")
    init = st.radio("Initial θ", ["All 5.0 (recommended)", "Random N(0,1)", "Zeros (as in manual)"], horizontal=True)
    t0 = {"A": np.full(7, 5.0), "R": np.random.default_rng(0).normal(size=7), "Z": np.zeros(7)}[init[0]]
    if init[0] == "Z":
        st.warning("With standardized data the minimizer is θ* = 0, so starting at zeros means the gradient is already 0: "
                   "the objective stays flat at J_min for every learning rate. Use a non-zero start to observe convergence.")
    jmin = objective_function(Xs.mean(axis=0), Xs)
    tabs = st.tabs(["1 Dataset", "2 Run GD", "3 Parameters", "4 Learning rates", "5 Iterations", "6 Convergence & results"])

    with tabs[0]:
        st.write("Shape:", df.shape); st.dataframe(df.head())
        st.write("Selected features dtypes / missing values")
        st.dataframe(pd.DataFrame({"dtype": df[FEATURES].dtypes.astype(str), "missing": df[FEATURES].isnull().sum()}))
        st.write("Standardized data (first 5 rows)"); st.dataframe(pd.DataFrame(Xs[:5], columns=FEATURES))

    with tabs[1]:
        c1, c2 = st.columns(2)
        lr = c1.slider("Learning rate", 0.001, 2.5, 0.1, 0.001, format="%.3f")
        n = c2.slider("Iterations", 1, 500, 100)
        th, cs, _ = gradient_descent(Xs, lr, n, t0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Initial objective", f"{objective_function(t0, Xs):.6f}")
        m2.metric("Final objective", f"{cs[-1]:.6f}")
        m3.metric("Minimum possible J", f"{jmin:.6f}")
        st.dataframe(pd.DataFrame({"initial θ": t0, "final θ": th}, index=FEATURES))
        fig, ax = plt.subplots(figsize=(8, 5)); curve(ax, [cs]); ax.set_ylabel("Objective Function")
        ax.set_title("Gradient Descent Convergence"); st.pyplot(fig)
        ok, msg = classify(lr, cs); (st.success if ok else st.error)(msg)

    with tabs[2]:
        lr_p = st.slider("Learning rate", 0.001, 1.9, 0.1, 0.001, key="lrp", format="%.3f")
        _, _, H = gradient_descent(Xs, lr_p, 100, t0)
        fig, axs = plt.subplots(2, 4, figsize=(14, 6)); axs = axs.ravel()
        for i, f in enumerate(FEATURES):
            axs[i].plot(H[:, i]); axs[i].set_title(f); axs[i].grid(True); axs[i].set_xlabel("Iteration")
        axs[7].axis("off"); fig.tight_layout(); st.pyplot(fig)
        st.caption("Each θ_j moves from its initial value toward the feature mean (0 after standardization).")

    with tabs[3]:
        sel = st.multiselect("Learning rates", sorted(set(LR_LIST + [1.5, 1.9, 2.1])), LR_LIST)
        res = {l: gradient_descent(Xs, l, 100, t0)[1] for l in sel}
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        for l, c in res.items():
            axs[0].plot(c, label=f"LR = {l}")
            axs[1].plot(np.maximum(np.array(c) - jmin, 1e-16), label=f"LR = {l}")
        axs[0].set_ylabel("Objective Function"); axs[1].set_ylabel("J − J_min (log scale)"); axs[1].set_yscale("log")
        for a in axs: a.set_xlabel("Iteration"); a.grid(True); a.legend()
        fig.suptitle("Effect of Learning Rate on Convergence"); st.pyplot(fig)
        st.dataframe(pd.DataFrame([{"Learning Rate": l, "Final Objective": c[-1], "Iterations to converge": iters_to_converge(c), "Behavior": classify(l, c)[1]} for l, c in res.items()]))

    with tabs[4]:
        lr_i = st.select_slider("Learning rate", LR_LIST, 0.1, key="lri")
        rows = [[k, gradient_descent(Xs, lr_i, k, t0)[1][-1]] for k in ITER_LIST]
        idf = pd.DataFrame(rows, columns=["Iterations", "Final Objective"]); st.dataframe(idf)
        fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(idf["Iterations"], idf["Final Objective"], marker="o")
        ax.set_xlabel("Number of Iterations"); ax.set_ylabel("Final Objective"); ax.set_title("Effect of Number of Iterations"); ax.grid(True)
        st.pyplot(fig)

    with tabs[5]:
        tol = st.select_slider("Tolerance ε", [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10], 1e-6, format_func=lambda v: f"{v:.0e}")
        rows = []
        for l in LR_LIST:
            _, c, _ = gradient_descent(Xs, l, 100, t0)
            ok, perf = classify(l, c, tol)
            rows.append({"Learning Rate": l, "Iterations": 100, "Initial Objective": objective_function(t0, Xs),
                         "Final Objective": c[-1], "Iterations to ε": iters_to_converge(c, tol),
                         "Converged?": "Yes" if ok else "No", "Performance": perf})
        rdf = pd.DataFrame(rows); st.dataframe(rdf)
        st.download_button("Download result table (CSV)", rdf.to_csv(index=False), "result_table.csv")
        st.text_area("Conclusion (write your own)", height=120)

st.markdown('<div class="vl-foot">Machine Learning Virtual Lab · Experiment 2 · Built with Streamlit</div>', unsafe_allow_html=True)

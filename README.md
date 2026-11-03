# Gradient Descent Virtual Lab (Streamlit)

Layout mirrors Virtual Labs experiments: Aim, Theory, Pretest, Procedure, Simulation, Posttest, References, Contributors, Feedback.
The interface uses a dark theme.

## Run
    pip install -r requirements.txt
    streamlit run app.py

## Data
Download `bank-full.csv` from the UCI Bank Marketing dataset and place it at `data/bank-full.csv` (or upload it in the Simulation page).
Without it, a synthetic dataset with the same 7 numeric columns is used.

## Structure
    app.py           UI and navigation
    core/gd.py       data loading, objective, gradient, GD, convergence checks
    core/content.py  theory, procedure, quizzes, references

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simpy
import random
from scipy import stats

st.set_page_config(page_title="Simulation Dashboard", layout="wide")
st.title("BSD 328: Simulation & Modelling Dashboard")

# Sidebar for navigation
st.sidebar.header("Dashboard Controls")
sim_type = st.sidebar.selectbox(
    "Select a Module", 
    ["1. RNG (LCG)", "2. Bank Queue Simulation", "3. Data Analysis (Regression & Skewness)"]
)

# -----------------------------------------
# 1. Random Number Generation
# -----------------------------------------
if sim_type == "1. RNG (LCG)":
    st.header("Linear Congruential Generator (LCG)")
    
    n_samples = st.slider("Number of Random Numbers", 100, 10000, 1000)
    
    def lcg(seed, a, c, m, n):
        numbers = []
        x = seed
        for _ in range(n):
            x = (a * x + c) % m
            numbers.append(x / m)
        return numbers

    if st.button("Generate Numbers"):
        nums = lcg(42, 1664525, 1013904223, 2**32, n_samples)
        
        fig, ax = plt.subplots()
        ax.hist(nums, bins=50, color='purple', edgecolor='black', alpha=0.7)
        ax.set_title("LCG Uniformity Histogram")
        st.pyplot(fig)
        st.success("RNG Generation Complete!")

# -----------------------------------------
# 2. Queue Simulation (Bank)
# -----------------------------------------
elif sim_type == "2. Bank Queue Simulation":
    st.header("Mobile-Money Desk Simulation")
    
    sim_time = st.number_input("Simulation Time (minutes)", value=300)
    
    def run_bank_sim(time):
        wait_times = []
        def bank_customer(env, agent):
            arrival_time = env.now
            with agent.request() as req:
                yield req
                wait_times.append(env.now - arrival_time)
                yield env.timeout(random.uniform(1, 4))

        def bank_setup(env, agent):
            while True:
                yield env.timeout(random.expovariate(1.0/3.0))
                env.process(bank_customer(env, agent))

        env = simpy.Environment()
        agent = simpy.Resource(env, capacity=1)
        env.process(bank_setup(env, agent))
        env.run(until=time)
        return wait_times

    if st.button("Run Queue Simulation"):
        waits = run_bank_sim(sim_time)
        st.metric("Average Waiting Time (mins)", round(np.mean(waits), 2))
        st.metric("Customers Waited > 5 mins", f"{(sum(1 for w in waits if w > 5) / len(waits)):.2%}")
        st.success("Simulation Complete!")

# -----------------------------------------
# 3. Data Analysis (Regression & Skewness)
# -----------------------------------------
elif sim_type == "3. Data Analysis (Regression & Skewness)":
    st.header("Upload Data for Regression Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV (Server Load vs Response Time)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())
        
        if st.button("Analyze Regression"):
            # Assuming first col is Load, second is Response
            load = df.iloc[:, 0].values
            response = df.iloc[:, 1].values
            
            slope, intercept, r_value, _, _ = stats.linregress(load, response)
            st.write(f"**Regression Equation:** Y = {slope:.2f}X + {intercept:.2f}")
            
            fig, ax = plt.subplots()
            ax.scatter(load, response, color='blue', label="Actual")
            ax.plot(load, slope*load + intercept, color='red', label="Predicted")
            ax.set_xlabel("Server Load (%)")
            ax.set_ylabel("Response Time (ms)")
            ax.legend()
            st.pyplot(fig)

    st.divider()
    
    st.header("Network Speeds (Skewness & Kurtosis)")
    speeds = [28, 30, 35, 37, 40, 42, 45, 52, 55, 60, 62, 90, 110]
    st.write(f"**Dataset:** {speeds}")
    
    if st.button("Compute Stats"):
        st.metric("Skewness", round(stats.skew(speeds), 4))
        st.metric("Kurtosis", round(stats.kurtosis(speeds), 4))
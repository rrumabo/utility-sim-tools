import streamlit as st
import subprocess
import yaml
import os

st.title("🧪 Utility PDE Simulator")


st.sidebar.header("Simulation Parameters")
L = st.sidebar.slider("Domain Length (L)", 5.0, 20.0, 10.0)
N = st.sidebar.slider("Grid Points (N)", 33, 257, 129, step=32)
alpha = st.sidebar.slider("Diffusion Coefficient (α)", 0.1, 5.0, 1.0)
dt = st.sidebar.number_input("Time Step (dt)", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
steps = st.sidebar.number_input("Steps", 100, 10000, 500, step=100)

st.sidebar.header("Initial Condition")
ic_type = st.sidebar.selectbox("Type", ["gaussian_bump"])  # Add more later
center = st.sidebar.slider("Center", -5.0, 5.0, 0.0)
width = st.sidebar.slider("Width", 0.1, 2.0, 0.5)
amplitude = st.sidebar.slider("Amplitude", 0.1, 2.0, 1.0)

plot_profile = st.sidebar.checkbox("Save Final Plot", True)
save_animation = st.sidebar.checkbox("Save GIF", True)
save_diagnostics = st.sidebar.checkbox("Save Diagnostics", True)

if st.button("Run Simulation"):
    cfg = {
        "simulation": {
            "L": L,
            "N": N,
            "alpha": alpha,
            "dt": dt,
            "steps": steps
        },
        "initial_condition": {
            "type": ic_type,
            "center": center,
            "width": width,
            "amplitude": amplitude
        },
        "output": {
            "folder": "figures",
            "plot_profile": plot_profile,
            "save_animation": save_animation,
            "save_diagnostics": save_diagnostics
        }
    }


    with open("config_gui.yaml", "w") as f:
        yaml.dump(cfg, f)

    result = subprocess.run(["python3", "main.py", "--config=config_gui.yaml"], capture_output=True, text=True)

    st.success("Simulation completed!")
    st.text("Output:")
    st.code(result.stdout)

    
    if plot_profile and os.path.exists("figures/final_comparison.png"):
        st.image("figures/final_comparison.png", caption="Final vs Initial Profile")

    if save_animation and os.path.exists("figures/heat_diffusion.gif"):
        st.image("figures/heat_diffusion.gif", caption="GIF Preview", use_container_width=True)
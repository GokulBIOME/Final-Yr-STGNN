"""
BIOME ST-GNN Streamlit App
Minimal Streamlit application for deployment.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="BIOME ST-GNN Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application."""
    
    # Title
    st.title("🧠 BIOME ST-GNN Dashboard")
    st.markdown("Spatio-Temporal Graph Neural Network for Dynamic Connectivity Analytics")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Models", "Data", "Monitoring", "Inference"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Models":
        show_models()
    elif page == "Data":
        show_data()
    elif page == "Monitoring":
        show_monitoring()
    elif page == "Inference":
        show_inference()

def show_overview():
    """Show overview page."""
    st.header("📊 System Overview")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Models", "3", "1")
    
    with col2:
        st.metric("Data Samples", "1,247", "23")
    
    with col3:
        st.metric("Training Accuracy", "94.2%", "2.1%")
    
    with col4:
        st.metric("System Status", "🟢 Online", "0")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    # Sample data
    activity_data = pd.DataFrame({
        'Time': pd.date_range('2024-01-01', periods=10, freq='H'),
        'Activity': ['Model Training', 'Data Processing', 'Inference', 'Model Evaluation', 
                    'Data Upload', 'Model Training', 'Inference', 'Data Processing', 
                    'Model Evaluation', 'System Update'],
        'Status': ['Completed', 'Completed', 'Completed', 'Completed', 'Completed',
                  'In Progress', 'Completed', 'Completed', 'Completed', 'Completed']
    })
    
    st.dataframe(activity_data, use_container_width=True)

def show_models():
    """Show models page."""
    st.header("🧠 Model Management")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["ST-GNN Basic", "ST-GNN Enhanced", "Dynamic ST-GNN"]
    )
    
    # Model parameters
    st.subheader("Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_dim = st.slider("Input Dimension", 50, 500, 200)
        hidden_dim = st.slider("Hidden Dimension", 64, 512, 256)
    
    with col2:
        output_dim = st.slider("Output Dimension", 50, 500, 200)
        num_layers = st.slider("Number of Layers", 1, 10, 3)
    
    # Model info
    st.subheader("Model Information")
    st.info(f"""
    **Model Type:** {model_type}
    **Input Dimension:** {input_dim}
    **Hidden Dimension:** {hidden_dim}
    **Output Dimension:** {output_dim}
    **Number of Layers:** {num_layers}
    """)
    
    # Training controls
    st.subheader("Training Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Training"):
            st.success("Training started!")
    
    with col2:
        if st.button("Stop Training"):
            st.warning("Training stopped!")
    
    with col3:
        if st.button("Save Model"):
            st.success("Model saved!")

def show_data():
    """Show data page."""
    st.header("📊 Data Explorer")
    
    # Data upload
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload fMRI data (.npz, .npy, .csv)",
        type=['npz', 'npy', 'csv']
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Sample data visualization
        st.subheader("Data Visualization")
        
        # Generate sample data
        sample_data = np.random.randn(100, 10)
        
        # Time series plot
        fig = go.Figure()
        for i in range(5):
            fig.add_trace(go.Scatter(
                y=sample_data[:, i],
                mode='lines',
                name=f'ROI {i+1}'
            ))
        
        fig.update_layout(
            title="Sample ROI Time Series",
            xaxis_title="Time Points",
            yaxis_title="Signal Intensity"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr_matrix = np.corrcoef(sample_data.T)
        
        fig_corr = px.imshow(
            corr_matrix,
            title="ROI Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)

def show_monitoring():
    """Show monitoring page."""
    st.header("📈 System Monitoring")
    
    # System metrics
    st.subheader("System Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU usage
        cpu_usage = np.random.uniform(20, 80)
        st.metric("CPU Usage", f"{cpu_usage:.1f}%")
        
        # Memory usage
        memory_usage = np.random.uniform(30, 70)
        st.metric("Memory Usage", f"{memory_usage:.1f}%")
    
    with col2:
        # GPU usage
        gpu_usage = np.random.uniform(0, 90)
        st.metric("GPU Usage", f"{gpu_usage:.1f}%")
        
        # Disk usage
        disk_usage = np.random.uniform(40, 85)
        st.metric("Disk Usage", f"{disk_usage:.1f}%")
    
    # Performance charts
    st.subheader("Performance Trends")
    
    # Generate sample performance data
    time_points = pd.date_range('2024-01-01', periods=50, freq='H')
    accuracy_data = np.cumsum(np.random.randn(50) * 0.1) + 0.9
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=time_points,
        y=accuracy_data,
        mode='lines+markers',
        name='Model Accuracy'
    ))
    
    fig_perf.update_layout(
        title="Model Accuracy Over Time",
        xaxis_title="Time",
        yaxis_title="Accuracy"
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)

def show_inference():
    """Show inference page."""
    st.header("⚡ Model Inference")
    
    # Input parameters
    st.subheader("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sequence_length = st.slider("Sequence Length", 10, 1000, 100)
        num_rois = st.slider("Number of ROIs", 10, 500, 100)
    
    with col2:
        model_type = st.selectbox(
            "Model Type",
            ["ST-GNN Basic", "ST-GNN Enhanced"]
        )
        return_edges = st.checkbox("Return Edge Predictions")
    
    # Generate sample input
    if st.button("Generate Sample Input"):
        sample_input = np.random.randn(sequence_length, num_rois)
        
        st.subheader("Sample Input Data")
        st.dataframe(pd.DataFrame(sample_input), use_container_width=True)
        
        # Run inference (simulated)
        if st.button("Run Inference"):
            with st.spinner("Running inference..."):
                # Simulate processing time
                import time
                time.sleep(2)
                
                # Generate sample predictions
                predictions = np.random.randn(sequence_length, num_rois)
                
                st.success("Inference completed!")
                
                # Show results
                st.subheader("Predictions")
                st.dataframe(pd.DataFrame(predictions), use_container_width=True)
                
                # Download results
                csv = pd.DataFrame(predictions).to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()

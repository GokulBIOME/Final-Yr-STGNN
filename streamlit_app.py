"""
BIOME ST-GNN Streamlit App - Simple Version
Minimal Streamlit application for deployment without external dependencies.
"""

import streamlit as st
import numpy as np
import pandas as pd

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
    
    # System information
    st.subheader("System Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info("""
        **BIOME ST-GNN System**
        - Spatio-Temporal Graph Neural Networks
        - Dynamic Connectivity Analysis
        - Real-time Processing
        - Multi-modal Data Support
        """)
    
    with info_col2:
        st.success("""
        **Features**
        - Model Training & Inference
        - Data Visualization
        - System Monitoring
        - Performance Analytics
        """)

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
    
    # Model performance
    st.subheader("Model Performance")
    
    perf_data = pd.DataFrame({
        'Epoch': range(1, 11),
        'Training Loss': np.random.uniform(0.1, 0.5, 10),
        'Validation Loss': np.random.uniform(0.15, 0.4, 10),
        'Accuracy': np.random.uniform(0.8, 0.95, 10)
    })
    
    st.dataframe(perf_data, use_container_width=True)

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
        
        # Show data statistics
        st.subheader("Data Statistics")
        stats_data = pd.DataFrame({
            'ROI': [f'ROI_{i+1}' for i in range(10)],
            'Mean': sample_data.mean(axis=0),
            'Std': sample_data.std(axis=0),
            'Min': sample_data.min(axis=0),
            'Max': sample_data.max(axis=0)
        })
        
        st.dataframe(stats_data, use_container_width=True)
        
        # Show sample data
        st.subheader("Sample Data (First 10 rows)")
        sample_df = pd.DataFrame(sample_data[:10])
        sample_df.columns = [f'ROI_{i+1}' for i in range(10)]
        st.dataframe(sample_df, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr_matrix = np.corrcoef(sample_data.T)
        corr_df = pd.DataFrame(corr_matrix)
        corr_df.columns = [f'ROI_{i+1}' for i in range(10)]
        corr_df.index = [f'ROI_{i+1}' for i in range(10)]
        
        st.dataframe(corr_df, use_container_width=True)

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
    
    # Performance trends
    st.subheader("Performance Trends")
    
    # Generate sample performance data
    time_points = pd.date_range('2024-01-01', periods=20, freq='H')
    performance_data = pd.DataFrame({
        'Time': time_points,
        'CPU Usage': np.random.uniform(20, 80, 20),
        'Memory Usage': np.random.uniform(30, 70, 20),
        'GPU Usage': np.random.uniform(0, 90, 20),
        'Accuracy': np.random.uniform(0.8, 0.95, 20)
    })
    
    st.dataframe(performance_data, use_container_width=True)
    
    # System logs
    st.subheader("System Logs")
    
    logs_data = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=15, freq='30min'),
        'Level': np.random.choice(['INFO', 'WARNING', 'ERROR'], 15),
        'Message': [
            'Model training started',
            'Data processing completed',
            'Inference request received',
            'Model saved successfully',
            'System backup completed',
            'High memory usage detected',
            'GPU temperature normal',
            'Training epoch completed',
            'Data validation passed',
            'Model evaluation finished',
            'System restart scheduled',
            'Performance metrics updated',
            'User login successful',
            'Database connection restored',
            'Cache cleared successfully'
        ]
    })
    
    st.dataframe(logs_data, use_container_width=True)

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
        input_df = pd.DataFrame(sample_input)
        input_df.columns = [f'ROI_{i+1}' for i in range(num_rois)]
        st.dataframe(input_df.head(10), use_container_width=True)
        
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
                pred_df = pd.DataFrame(predictions)
                pred_df.columns = [f'ROI_{i+1}' for i in range(num_rois)]
                st.dataframe(pred_df.head(10), use_container_width=True)
                
                # Show prediction statistics
                st.subheader("Prediction Statistics")
                pred_stats = pd.DataFrame({
                    'ROI': [f'ROI_{i+1}' for i in range(min(10, num_rois))],
                    'Mean': predictions[:, :min(10, num_rois)].mean(axis=0),
                    'Std': predictions[:, :min(10, num_rois)].std(axis=0),
                    'Min': predictions[:, :min(10, num_rois)].min(axis=0),
                    'Max': predictions[:, :min(10, num_rois)].max(axis=0)
                })
                st.dataframe(pred_stats, use_container_width=True)
                
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

"""
Page 2: Automatic EEG Segmentation
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from utils.segmentation import segment_edf_file, batch_segment_files
from utils.visualization import plot_segmentation_results, plot_selected_segment
import json

st.set_page_config(page_title="Auto Segmentation", page_icon="✂️", layout="wide")

st.title("✂️ Automatic EEG Segmentation")
st.markdown("Detect optimal 3-second motor imagery windows using ERD analysis")

# Tabs
tab1, tab2, tab3 = st.tabs(["⚙️ Single File", "📦 Batch Processing", "📊 Results"])

with tab1:
    st.header("Segment Single File")
    
    st.info("""
    **Segmentation Process:**
    1. Remove first 0.5s (noise settling)
    2. Apply sliding window (3s length, 0.25s step)
    3. Compute Mu/Beta ERD for each window
    4. Select window with maximum ERD
    5. Extract and save 3-second segment
    """)
    
    # Load available files
    data_dir = Path("data/raw")
    edf_files = sorted(data_dir.glob("*.edf")) if data_dir.exists() else []
    
    if edf_files:
        file_names = [f.name for f in edf_files]
        selected_file = st.selectbox("Select EDF file", file_names)
        
        # Segmentation parameters
        with st.expander("⚙️ Advanced Parameters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                window_length = st.number_input(
                    "Window Length (s)",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5
                )
            
            with col2:
                step_size = st.number_input(
                    "Step Size (s)",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.25,
                    step=0.05
                )
            
            with col3:
                remove_start = st.number_input(
                    "Remove Start (s)",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.5,
                    step=0.1
                )
        
        # Run segmentation
        if st.button("▶️ Run Segmentation", type="primary"):
            file_path = data_dir / selected_file
            
            with st.spinner("Analyzing EEG signal..."):
                segment, seg_info, seg_result = segment_edf_file(
                    str(file_path),
                    window_length=window_length,
                    step_size=step_size,
                    remove_start=remove_start,
                    preprocess=True
                )
            
            if segment is not None:
                st.success("✅ Segmentation complete!")
                
                # Save to session state
                st.session_state['last_segment'] = segment
                st.session_state['last_seg_info'] = seg_info
                st.session_state['last_seg_result'] = seg_result
                
                # Display results
                st.subheader("📊 Segmentation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best Window Start", f"{seg_info['tmin']:.2f}s")
                with col2:
                    st.metric("Best Window End", f"{seg_info['tmax']:.2f}s")
                with col3:
                    st.metric("ERD Score", f"{seg_info['erd_score']:.1f}%")
                with col4:
                    st.metric("Quality Score", f"{seg_info['quality_score']:.2f}")
                
                # Plot ERD analysis
                st.subheader("📈 ERD Analysis")
                fig = plot_segmentation_results(seg_result)
                st.pyplot(fig)
                
                # Plot selected segment
                st.subheader("✂️ Selected Segment (3 seconds)")
                fig = plot_selected_segment(
                    segment,
                    seg_info['sfreq'],
                    seg_result['motor_channels'],
                    seg_info
                )
                st.pyplot(fig)
                
                # Save option
                if st.button("💾 Save Segment"):
                    output_dir = Path("data/processed")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    base_name = Path(selected_file).stem
                    output_path = output_dir / f"{base_name}_segment.npy"
                    
                    np.save(output_path, segment)
                    
                    # Save metadata
                    meta_path = output_dir / f"{base_name}_metadata.json"
                    with open(meta_path, 'w') as f:
                        # Convert numpy types to Python types
                        seg_info_save = {k: (v.item() if hasattr(v, 'item') else v) 
                                        for k, v in seg_info.items()}
                        json.dump(seg_info_save, f, indent=2)
                    
                    st.success(f"✅ Saved to {output_path}")
            else:
                st.error("❌ Segmentation failed")
    else:
        st.warning("No EDF files found. Upload files first.")

with tab2:
    st.header("Batch Segmentation")
    
    st.info("Process all EDF files at once and save segments automatically")
    
    data_dir = Path("data/raw")
    edf_files = sorted(data_dir.glob("*.edf")) if data_dir.exists() else []
    
    if edf_files:
        st.write(f"Found **{len(edf_files)}** files to process")
        
        # Parameters
        with st.expander("⚙️ Batch Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                batch_window_length = st.number_input(
                    "Window Length (s)",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    key="batch_window"
                )
            
            with col2:
                batch_step_size = st.number_input(
                    "Step Size (s)",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.25,
                    step=0.05,
                    key="batch_step"
                )
        
        if st.button("▶️ Process All Files", type="primary"):
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, file_path in enumerate(edf_files):
                status_text.text(f"Processing {i+1}/{len(edf_files)}: {file_path.name}")
                
                try:
                    segment, seg_info, seg_result = segment_edf_file(
                        str(file_path),
                        window_length=batch_window_length,
                        step_size=batch_step_size,
                        remove_start=0.5,
                        preprocess=True
                    )
                    
                    if segment is not None:
                        # Save segment
                        base_name = file_path.stem
                        seg_path = output_dir / f"{base_name}_segment.npy"
                        np.save(seg_path, segment)
                        
                        # Save metadata
                        meta_path = output_dir / f"{base_name}_metadata.json"
                        seg_info_save = {k: (v.item() if hasattr(v, 'item') else v) 
                                        for k, v in seg_info.items()}
                        with open(meta_path, 'w') as f:
                            json.dump(seg_info_save, f, indent=2)
                        
                        seg_info['segment_path'] = str(seg_path)
                        results.append(seg_info)
                    
                except Exception as e:
                    st.warning(f"Failed to process {file_path.name}: {e}")
                
                progress_bar.progress((i + 1) / len(edf_files))
            
            status_text.text("✅ Batch processing complete!")
            
            st.success(f"✅ Processed {len(results)}/{len(edf_files)} files successfully")
            
            # Save results summary
            results_df = pd.DataFrame(results)
            results_csv_path = output_dir / "segmentation_results.csv"
            results_df.to_csv(results_csv_path, index=False)
            
            st.balloons()
            
            # Display summary
            st.subheader("📊 Batch Results Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_erd = results_df['erd_score'].mean()
                st.metric("Avg ERD Score", f"{avg_erd:.1f}%")
            
            with col2:
                avg_quality = results_df['quality_score'].mean()
                st.metric("Avg Quality", f"{avg_quality:.2f}")
            
            with col3:
                good_quality = len(results_df[results_df['quality_score'] > 0.5])
                st.metric("Good Quality", f"{good_quality}/{len(results_df)}")
    else:
        st.warning("No files to process")

with tab3:
    st.header("Segmentation Results")
    
    output_dir = Path("data/processed")
    
    if output_dir.exists():
        # Load segmentation results if available
        results_file = output_dir / "segmentation_results.csv"
        
        if results_file.exists():
            results_df = pd.read_csv(results_file)
            
            st.success(f"Loaded {len(results_df)} segmentation results")
            
            # Display table
            st.subheader("📋 All Segments")
            
            display_cols = ['filename', 'direction', 'subject', 'method', 
                          'tmin', 'tmax', 'erd_score', 'quality_score']
            available_cols = [col for col in display_cols if col in results_df.columns]
            
            st.dataframe(
                results_df[available_cols].round(2),
                use_container_width=True
            )
            
            # Statistics
            st.subheader("📊 Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Segments", len(results_df))
            
            with col2:
                avg_erd = results_df['erd_score'].mean()
                st.metric("Avg ERD", f"{avg_erd:.1f}%")
            
            with col3:
                avg_quality = results_df['quality_score'].mean()
                st.metric("Avg Quality", f"{avg_quality:.2f}")
            
            with col4:
                good_segments = len(results_df[results_df['quality_score'] > 0.5])
                st.metric("Good Segments", good_segments)
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ERD Score Distribution")
                st.bar_chart(results_df['erd_score'].value_counts().sort_index())
            
            with col2:
                st.subheader("Quality Score Distribution")
                st.bar_chart(results_df['quality_score'].value_counts().sort_index())
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="segmentation_results.csv",
                mime="text/csv"
            )
        else:
            st.info("No segmentation results found. Run batch processing first.")
    else:
        st.warning("No processed segments found")

st.markdown("---")
st.info("💡 **Next Step:** Go to '📊 Feature Extraction' to compute features from segments")
"""
Customer Retention Prediction and Analytics System
Main Streamlit Web Application

Run locally with:
    streamlit run app.py
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.data_loader import (
    compute_kpi_metrics,
    detect_id_columns,
    get_binary_candidate_columns,
    infer_target_mapping,
    inspect_dataset,
    load_sample_dataset,
    load_uploaded_csv,
    suggest_target_column,
)
from modules.eda import (
    figure_to_base64,
    find_matching_column,
    plot_categorical_retention_breakdown,
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_retention_by_contract_type,
    plot_retention_by_monthly_charges,
    plot_retention_by_tenure,
    plot_retention_distribution,
    summarize_eda_insights,
)
from modules.model_trainer import (
    list_saved_models,
    load_model_artifact,
    plot_confusion_matrix_figure,
    plot_feature_importance_figure,
    save_model_artifact,
    train_and_evaluate_models,
)
from modules.predictor import predict_batch_customers, predict_single_customer
from modules.preprocessing import clean_raw_dataframe, get_cleaned_preview_df
from modules.report_generator import generate_html_report, generate_markdown_report
from modules.ui_styles import (
    inject_custom_css,
    render_hero_banner,
    render_kpi_card,
    render_section_title,
)


st.set_page_config(
    page_title="Customer Retention Prediction & Analytics System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()


def _init_session_state() -> None:
    """Initialize Streamlit session state and auto-load the built-in sample dataset."""
    if "raw_df" not in st.session_state:
        try:
            sample_df = load_sample_dataset()
            st.session_state["raw_df"] = sample_df
            st.session_state["dataset_name"] = "customer_retention_sample.csv (Built-in Sample)"
            st.session_state["target_col"] = suggest_target_column(sample_df)
            st.session_state["id_cols"] = detect_id_columns(sample_df)
        except Exception:
            st.session_state["raw_df"] = None
            st.session_state["dataset_name"] = "None"
            st.session_state["target_col"] = None
            st.session_state["id_cols"] = []

    defaults = {
        "cleaned_df": None,
        "eda_df": None,
        "preprocessing_log": None,
        "target_mapping": None,
        "training_results": None,
        "active_artifact": None,
        "active_model_source": None,
        "batch_result": None,
        "last_single_prediction": None,
        "remove_duplicates": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Ensure target mapping and cleaned DataFrame stay synchronized on initial load
    _sync_dataset_derivatives()


def _sync_dataset_derivatives(force_reclean: bool = False) -> None:
    """Keep target mapping and preprocessed dataset synchronized with raw_df & target_col."""
    raw_df: Optional[pd.DataFrame] = st.session_state.get("raw_df")
    target_col: Optional[str] = st.session_state.get("target_col")
    if raw_df is None or target_col is None or target_col not in raw_df.columns:
        return

    try:
        mapping = infer_target_mapping(raw_df[target_col])
        st.session_state["target_mapping"] = mapping
    except ValueError:
        st.session_state["target_mapping"] = None
        return

    if st.session_state["cleaned_df"] is None or force_reclean:
        id_cols = st.session_state.get("id_cols", [])
        remove_dups = st.session_state.get("remove_duplicates", True)
        cleaned, audit = clean_raw_dataframe(
            raw_df,
            target_col=target_col,
            id_cols=id_cols,
            remove_duplicates=remove_dups,
        )
        eda_df = get_cleaned_preview_df(cleaned, target_col=target_col, id_cols=id_cols)
        st.session_state["cleaned_df"] = cleaned
        st.session_state["eda_df"] = eda_df
        st.session_state["preprocessing_log"] = audit


def _reset_downstream_state() -> None:
    """Reset preprocessed cache and model training results when dataset/target changes."""
    st.session_state["cleaned_df"] = None
    st.session_state["eda_df"] = None
    st.session_state["preprocessing_log"] = None
    st.session_state["training_results"] = None
    st.session_state["batch_result"] = None
    st.session_state["last_single_prediction"] = None


def _render_sidebar() -> str:
    """Render sidebar navigation and live system state summary."""
    with st.sidebar:
        st.markdown("### 📊 Retention Analytics")
        st.caption("Customer Retention Prediction & Analytics System")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard Overview",
                "🗂️ Dataset Analysis",
                "📊 Data Visualization (EDA)",
                "🤖 Model Training & Evaluation",
                "🎯 Customer Prediction",
                "📄 Reports & Export",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # Quick Data Source Switcher in Sidebar
        with st.expander("📁 Quick Dataset Upload", expanded=False):
            uploaded = st.file_uploader("Upload Customer CSV", type=["csv"], key="sidebar_csv_uploader")
            if uploaded is not None:
                if st.button("Load Uploaded CSV", use_container_width=True, key="sidebar_load_csv"):
                    try:
                        df_new = load_uploaded_csv(uploaded)
                        st.session_state["raw_df"] = df_new
                        st.session_state["dataset_name"] = uploaded.name
                        st.session_state["target_col"] = suggest_target_column(df_new)
                        st.session_state["id_cols"] = detect_id_columns(df_new)
                        _reset_downstream_state()
                        _sync_dataset_derivatives(force_reclean=True)
                        st.success(f"Loaded {uploaded.name} ({len(df_new):,} rows)")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            if st.button("Load Sample Dataset", use_container_width=True, key="sidebar_load_sample"):
                try:
                    sample_df = load_sample_dataset()
                    st.session_state["raw_df"] = sample_df
                    st.session_state["dataset_name"] = "customer_retention_sample.csv (Built-in Sample)"
                    st.session_state["target_col"] = suggest_target_column(sample_df)
                    st.session_state["id_cols"] = detect_id_columns(sample_df)
                    _reset_downstream_state()
                    _sync_dataset_derivatives(force_reclean=True)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        # Active Session Status Card
        raw_df = st.session_state.get("raw_df")
        target_col = st.session_state.get("target_col")
        active_artifact = st.session_state.get("active_artifact")
        row_info = f"{len(raw_df):,} rows × {len(raw_df.columns)} cols" if raw_df is not None else "Not loaded"
        model_info = active_artifact["model_name"] if active_artifact else "Not trained yet"

        st.markdown(
            f"""
            <div class="sidebar-card">
                <strong>Active Dataset:</strong><br/>{st.session_state.get('dataset_name', 'None')}<br/>
                <span style="color:#64748b;">{row_info}</span><br/><br/>
                <strong>Target Column:</strong> <code>{target_col or 'None'}</code><br/>
                <strong>Active ML Model:</strong><br/>{model_info}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("MUIT Noida • BCA Project (2026–2027)")

    return page


# ==============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ==============================================================================
def render_dashboard_overview() -> None:
    render_hero_banner(
        "Customer Retention Prediction & Analytics System",
        "Interactive Machine Learning & Exploratory Analytics Dashboard for identifying churn risk and driving customer retention.",
    )

    raw_df = st.session_state.get("raw_df")
    eda_df = st.session_state.get("eda_df")
    target_col = st.session_state.get("target_col")
    mapping = st.session_state.get("target_mapping")

    if raw_df is None:
        st.warning("No dataset loaded. Please upload a CSV file or load the sample dataset on the Dataset Analysis page.")
        return

    if mapping is None or target_col is None:
        st.warning(
            f"Target column `{target_col}` is not binary. Visit **Dataset Analysis** to select a valid binary retention/churn target column."
        )
        return

    active_df = eda_df if eda_df is not None else raw_df
    total_c, ret_c, risk_c, ret_rate, churn_rate = compute_kpi_metrics(
        active_df, target_col, mapping["retained_label"], mapping["churned_label"]
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Total Customers", f"{total_c:,}", "After cleaning & deduplication", "slate")
    with col2:
        render_kpi_card("Retained Customers", f"{ret_c:,}", f"Label: {mapping['retained_label']}", "teal")
    with col3:
        render_kpi_card("At-Risk / Churned", f"{risk_c:,}", f"Label: {mapping['churned_label']}", "rose")
    with col4:
        render_kpi_card("Retention Rate", f"{ret_rate:.1f}%", "Active retained share", "teal")
    with col5:
        render_kpi_card("Churn Rate", f"{churn_rate:.1f}%", "Observed churn share", "rose")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Row 2: Live Charts
    render_section_title("Live Retention & Churn Overview")
    c_left, c_right = st.columns(2)
    with c_left:
        fig1 = plot_retention_distribution(active_df, target_col, mapping["retained_label"], mapping["churned_label"])
        st.pyplot(fig1)
        plt.close(fig1)
    with c_right:
        fig2 = plot_churn_distribution(active_df, target_col, mapping["retained_label"], mapping["churned_label"])
        st.pyplot(fig2)
        plt.close(fig2)

    # Row 3: Contract & Tenure live chart if available
    contract_col = find_matching_column(active_df, ["contract_type", "contract", "plan"])
    tenure_col = find_matching_column(active_df, ["tenure", "months", "duration"])
    if contract_col:
        render_section_title(f"Retention Analysis by {contract_col}")
        fig_c = plot_retention_by_contract_type(
            active_df, target_col, mapping["retained_label"], mapping["churned_label"], contract_col
        )
        st.pyplot(fig_c)
        plt.close(fig_c)
    elif tenure_col and pd.api.types.is_numeric_dtype(active_df[tenure_col]):
        render_section_title(f"Retention Analysis by {tenure_col}")
        fig_t = plot_retention_by_tenure(
            active_df, target_col, mapping["retained_label"], mapping["churned_label"], tenure_col
        )
        st.pyplot(fig_t)
        plt.close(fig_t)

    # Row 4: Model Status & Analytical Insights
    render_section_title("Machine Learning Model & Key Insights")
    m_col, i_col = st.columns([1.2, 1])

    with m_col:
        artifact = st.session_state.get("active_artifact")
        if artifact:
            m = artifact["metrics"]
            st.success(f"Active Model: **{artifact['model_name']}** (Trained at {artifact['trained_at']})")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
            mc2.metric("Precision", f"{m['precision'] * 100:.2f}%")
            mc3.metric("Recall", f"{m['recall'] * 100:.2f}%")
            mc4.metric("F1-Score", f"{m['f1_score'] * 100:.2f}%")
        else:
            st.info(
                "No ML model is active in this session yet. Go to **Model Training & Evaluation** to train the Baseline & Random Forest classifiers (or load a saved `.joblib` model)."
            )
            saved = list_saved_models()
            if saved:
                if st.button(f"Load Latest Saved Model ({saved[0].name})", type="primary"):
                    loaded = load_model_artifact(saved[0])
                    st.session_state["active_artifact"] = loaded
                    st.session_state["active_model_source"] = str(saved[0])
                    st.rerun()

    with i_col:
        insights = summarize_eda_insights(active_df, target_col, mapping["retained_label"], mapping["churned_label"])
        st.markdown("**Automated Dataset Insights:**")
        if "avg_tenure_retained" in insights:
            st.write(
                f"- Average **{insights['tenure_col']}**: `{insights['avg_tenure_retained']}` (Retained) vs `{insights['avg_tenure_churned']}` (Churned)"
            )
        if "avg_charges_retained" in insights:
            st.write(
                f"- Average **{insights['charges_col']}**: `{insights['avg_charges_retained']}` (Retained) vs `{insights['avg_charges_churned']}` (Churned)"
            )
        if "contract_churn_rates" in insights:
            highest_c = max(insights["contract_churn_rates"].items(), key=lambda x: x[1])
            st.write(
                f"- Highest churn segment in **{insights['contract_col']}**: `{highest_c[0]}` (`{highest_c[1]}%` churn)"
            )
        if st.session_state.get("preprocessing_log"):
            log = st.session_state["preprocessing_log"]
            st.write(
                f"- Preprocessing cleaned `{log['total_duplicates_removed']}` duplicate rows and excluded IDs: `{log['excluded_id_columns'] or 'None'}`"
            )


# ==============================================================================
# PAGE 2: DATASET ANALYSIS & PREPROCESSING
# ==============================================================================
def render_dataset_analysis() -> None:
    render_hero_banner(
        "Dataset Analysis & Data Preprocessing",
        "Upload customer CSV records, inspect schema & missing/duplicate diagnostics, select target/ID columns, and execute leakage-safe cleaning.",
    )

    up_col, sample_col = st.columns([2, 1])
    with up_col:
        uploaded_file = st.file_uploader(
            "Upload Customer Dataset (CSV)",
            type=["csv"],
            help="Upload any customer churn/retention CSV file.",
            key="main_dataset_uploader",
        )
        if uploaded_file is not None:
            if st.button("Apply Uploaded Dataset", type="primary"):
                try:
                    df_new = load_uploaded_csv(uploaded_file)
                    st.session_state["raw_df"] = df_new
                    st.session_state["dataset_name"] = uploaded_file.name
                    st.session_state["target_col"] = suggest_target_column(df_new)
                    st.session_state["id_cols"] = detect_id_columns(df_new)
                    _reset_downstream_state()
                    _sync_dataset_derivatives(force_reclean=True)
                    st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df_new):,} rows).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Upload error: {exc}")

    with sample_col:
        st.markdown("**Quick Testing Dataset**")
        if st.button("Reset to Sample Customer Dataset", use_container_width=True):
            try:
                sample_df = load_sample_dataset()
                st.session_state["raw_df"] = sample_df
                st.session_state["dataset_name"] = "customer_retention_sample.csv (Built-in Sample)"
                st.session_state["target_col"] = suggest_target_column(sample_df)
                st.session_state["id_cols"] = detect_id_columns(sample_df)
                _reset_downstream_state()
                _sync_dataset_derivatives(force_reclean=True)
                st.success("Loaded built-in sample dataset.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    raw_df: Optional[pd.DataFrame] = st.session_state.get("raw_df")
    if raw_df is None:
        st.info("Upload a CSV or load the sample dataset to begin.")
        return

    inspection = inspect_dataset(raw_df)

    # Raw Dataset Summary Cards
    render_section_title("1. Raw Dataset Health & Diagnostics")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Total Records (Raw)", f"{inspection['row_count']:,}", "Before deduplication", "slate")
    with k2:
        render_kpi_card("Total Columns", f"{inspection['col_count']}", f"{len(inspection['numeric_columns'])} numeric / {len(inspection['categorical_columns'])} categorical", "indigo")
    with k3:
        render_kpi_card("Missing Cells", f"{inspection['total_missing']:,}", f"{inspection['missing_pct_overall']}% of all cells", "amber")
    with k4:
        render_kpi_card("Duplicate Records", f"{inspection['duplicate_count']:,}", "Exact duplicate rows detected", "rose")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    tab_preview, tab_schema = st.tabs(["📋 Dataset Records Preview", "🔍 Column Data Types & Missing Values"])
    with tab_preview:
        rows_to_show = st.slider("Number of preview rows", 5, min(100, len(raw_df)), 15)
        st.dataframe(raw_df.head(rows_to_show), use_container_width=True)
    with tab_schema:
        st.dataframe(inspection["column_summary"], use_container_width=True, hide_index=True)

    # Target & ID Configuration
    render_section_title("2. Target Column & Preprocessing Configuration")
    binary_cols = get_binary_candidate_columns(raw_df)
    all_cols = raw_df.columns.tolist()

    cfg1, cfg2, cfg3 = st.columns([1.2, 1.2, 1])
    with cfg1:
        current_target = st.session_state.get("target_col")
        default_idx = all_cols.index(current_target) if current_target in all_cols else len(all_cols) - 1
        selected_target = st.selectbox(
            "Select Target Column (Retention / Churn)",
            options=all_cols,
            index=default_idx,
            help=f"Detected binary columns: {', '.join(binary_cols) if binary_cols else 'None'}",
        )

    with cfg2:
        available_id_choices = [c for c in all_cols if c != selected_target]
        current_ids = [c for c in st.session_state.get("id_cols", []) if c in available_id_choices]
        selected_ids = st.multiselect(
            "Exclude Identifier Columns (e.g., Customer_ID)",
            options=available_id_choices,
            default=current_ids,
            help="ID columns are automatically excluded from ML model training to prevent memorization/leakage.",
        )

    with cfg3:
        remove_dups = st.checkbox(
            "Remove Duplicate Records",
            value=st.session_state.get("remove_duplicates", True),
        )

    # Validate target column & allow explicit class confirmation
    try:
        mapping = infer_target_mapping(raw_df[selected_target])
        c_a, c_b = st.columns(2)
        with c_a:
            st.info(
                f"Detected **Retained (Class 0)**: `{mapping['retained_label']}` ({mapping['retained_count']:,} rows — {mapping['retention_rate']}%)"
            )
        with c_b:
            st.warning(
                f"Detected **Churned / At-Risk (Class 1)**: `{mapping['churned_label']}` ({mapping['churned_count']:,} rows — {mapping['churn_rate']}%)"
            )
        if st.checkbox("Swap Retained / Churned class labels", value=False):
            mapping["retained_label"], mapping["churned_label"] = mapping["churned_label"], mapping["retained_label"]
            mapping["retained_count"], mapping["churned_count"] = mapping["churned_count"], mapping["retained_count"]
            mapping["retention_rate"], mapping["churn_rate"] = mapping["churn_rate"], mapping["retention_rate"]
    except ValueError as exc:
        st.error(str(exc))
        return

    if st.button("⚙️ Apply Configuration & Run Preprocessing", type="primary"):
        if selected_target != st.session_state.get("target_col") or selected_ids != st.session_state.get("id_cols"):
            _reset_downstream_state()
        st.session_state["target_col"] = selected_target
        st.session_state["id_cols"] = selected_ids
        st.session_state["remove_duplicates"] = remove_dups
        st.session_state["target_mapping"] = mapping
        cleaned, audit = clean_raw_dataframe(
            raw_df,
            target_col=selected_target,
            id_cols=selected_ids,
            remove_duplicates=remove_dups,
        )
        eda_df = get_cleaned_preview_df(cleaned, target_col=selected_target, id_cols=selected_ids)
        st.session_state["cleaned_df"] = cleaned
        st.session_state["eda_df"] = eda_df
        st.session_state["preprocessing_log"] = audit
        st.success("Preprocessing pipeline applied successfully!")

    # Display Preprocessing Audit
    if st.session_state.get("preprocessing_log"):
        render_section_title("3. Preprocessing Audit & Leakage Prevention Summary")
        log = st.session_state["preprocessing_log"]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Rows After Cleaning", f"{log['rows_after_cleaning']:,}", delta=f"-{log['initial_rows'] - log['rows_after_cleaning']} removed" if log['initial_rows'] > log['rows_after_cleaning'] else "0 removed")
        p2.metric("Duplicates Removed", f"{log['total_duplicates_removed']:,}")
        p3.metric("Excluded ID Columns", f"{len(log['excluded_id_columns'])}")
        p4.metric("Missing Cells Handled", f"{log['initial_missing_cells']:,}")

        st.markdown(
            """
            **How Data Leakage is Prevented:**
            - Customer ID columns (`{ids}`) are excluded prior to feature engineering.
            - Duplicate records and missing-target rows are removed deterministically before splitting.
            - Numerical `SimpleImputer(median)` + `StandardScaler()` and categorical `SimpleImputer(most_frequent)` + `OneHotEncoder()` are fitted **exclusively on the 80% training split** inside scikit-learn `Pipeline` during model training and cross-validation.
            """.format(ids=", ".join(log["excluded_id_columns"]) if log["excluded_id_columns"] else "None")
        )

        eda_df = st.session_state.get("eda_df")
        if eda_df is not None:
            with st.expander("Preview Cleaned Dataset (Post-Imputation & Deduplication)"):
                st.dataframe(eda_df.head(20), use_container_width=True)
                csv_bytes = eda_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Cleaned Dataset (CSV)",
                    data=csv_bytes,
                    file_name="cleaned_customer_dataset.csv",
                    mime="text/csv",
                )


# ==============================================================================
# PAGE 3: DATA VISUALIZATION (EDA)
# ==============================================================================
def render_eda_page() -> None:
    render_hero_banner(
        "Exploratory Data Analysis (EDA)",
        "Visualizing customer retention, churn behavior, tenure, billing charges, contract types, correlations, and service demographics from your live dataset.",
    )

    eda_df = st.session_state.get("eda_df")
    target_col = st.session_state.get("target_col")
    mapping = st.session_state.get("target_mapping")
    id_cols = st.session_state.get("id_cols", [])

    if eda_df is None or target_col is None or mapping is None:
        st.warning("Please load a valid dataset and select a binary target column on **Dataset Analysis**.")
        return

    ret_lbl = mapping["retained_label"]
    churn_lbl = mapping["churned_label"]

    # 1. Retention & Churn Distributions
    render_section_title("1. Retention & Churn Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_retention_distribution(eda_df, target_col, ret_lbl, churn_lbl)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig = plot_churn_distribution(eda_df, target_col, ret_lbl, churn_lbl)
        st.pyplot(fig)
        plt.close(fig)

    # 2. Tenure Analysis
    render_section_title("2. Customer Retention by Tenure")
    numeric_cols = [c for c in eda_df.select_dtypes(include=["number"]).columns if c not in id_cols and c != target_col]
    if numeric_cols:
        default_tenure = find_matching_column(eda_df, ["tenure", "months", "duration"]) or numeric_cols[0]
        sel_tenure = st.selectbox(
            "Select Tenure / Duration Column",
            options=numeric_cols,
            index=numeric_cols.index(default_tenure) if default_tenure in numeric_cols else 0,
            key="eda_tenure_col",
        )
        fig = plot_retention_by_tenure(eda_df, target_col, ret_lbl, churn_lbl, sel_tenure)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric columns available for tenure analysis.")

    # 3. Monthly Charges Analysis
    render_section_title("3. Retention by Monthly Charges")
    if numeric_cols:
        default_charge = find_matching_column(eda_df, ["monthly_charges", "monthlycharges", "charges", "amount"]) or numeric_cols[0]
        sel_charge = st.selectbox(
            "Select Monthly Charges Column",
            options=numeric_cols,
            index=numeric_cols.index(default_charge) if default_charge in numeric_cols else 0,
            key="eda_charges_col",
        )
        fig = plot_retention_by_monthly_charges(eda_df, target_col, ret_lbl, churn_lbl, sel_charge)
        st.pyplot(fig)
        plt.close(fig)

    # 4. Contract Type Analysis
    render_section_title("4. Retention by Contract Type")
    cat_cols = [
        c for c in eda_df.columns
        if c not in id_cols and c != target_col and not pd.api.types.is_numeric_dtype(eda_df[c])
    ]
    if cat_cols:
        default_contract = find_matching_column(eda_df, ["contract_type", "contract", "plan"]) or cat_cols[0]
        sel_contract = st.selectbox(
            "Select Contract / Plan Column",
            options=cat_cols,
            index=cat_cols.index(default_contract) if default_contract in cat_cols else 0,
            key="eda_contract_col",
        )
        fig = plot_retention_by_contract_type(eda_df, target_col, ret_lbl, churn_lbl, sel_contract)
        st.pyplot(fig)
        plt.close(fig)

    # 5. Correlation Heatmap
    render_section_title("5. Numerical Correlation Heatmap")
    try:
        fig_corr = plot_correlation_heatmap(eda_df, target_col, churn_lbl, exclude_cols=id_cols)
        st.pyplot(fig_corr)
        plt.close(fig_corr)
    except ValueError as exc:
        st.info(str(exc))

    # 6. Demographic & Service Attribute Analysis
    render_section_title("6. Customer Demographic & Service Analysis")
    if cat_cols:
        default_demo = [c for c in cat_cols if c != sel_contract][:2] if len(cat_cols) > 1 else cat_cols[:1]
        selected_demo_cols = st.multiselect(
            "Choose Demographic / Service Attributes to Analyze",
            options=cat_cols,
            default=default_demo,
            key="eda_demo_cols",
        )
        for dcol in selected_demo_cols:
            fig_d = plot_categorical_retention_breakdown(eda_df, dcol, target_col, ret_lbl, churn_lbl)
            st.pyplot(fig_d)
            plt.close(fig_d)
    else:
        st.info("No categorical demographic/service columns found in the active dataset.")


# ==============================================================================
# PAGE 4: MODEL TRAINING & EVALUATION
# ==============================================================================
def render_model_training_page() -> None:
    render_hero_banner(
        "Machine Learning Training & Evaluation",
        "Train Baseline (Logistic Regression) and Random Forest Classifiers using an 80:20 stratified split, cross-validation, and Joblib persistence.",
    )

    cleaned_df = st.session_state.get("cleaned_df")
    target_col = st.session_state.get("target_col")
    mapping = st.session_state.get("target_mapping")
    id_cols = st.session_state.get("id_cols", [])

    if cleaned_df is None or target_col is None or mapping is None:
        st.warning("Please load and preprocess a dataset first on **Dataset Analysis**.")
        return

    # Training Controls
    render_section_title("1. Configure Training & Cross-Validation")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.text_input("Train / Test Split Ratio", value="80% Train / 20% Test", disabled=True)
    with c2:
        cv_folds = st.slider("Stratified CV Folds", min_value=3, max_value=10, value=5)
    with c3:
        n_estimators = st.slider("Random Forest Trees", min_value=50, max_value=400, value=200, step=50)
    with c4:
        max_depth = st.selectbox("Random Forest Max Depth", options=[0, 6, 10, 14, 20], format_func=lambda x: "Unlimited (None)" if x == 0 else str(x))

    if st.button("🚀 Train Baseline & Random Forest Models", type="primary"):
        with st.spinner("Training models with 80:20 stratified split and cross-validation..."):
            try:
                results = train_and_evaluate_models(
                    df_clean=cleaned_df,
                    target_col=target_col,
                    retained_label=mapping["retained_label"],
                    churned_label=mapping["churned_label"],
                    id_cols=id_cols,
                    test_size=0.20,
                    cv_folds=cv_folds,
                    rf_n_estimators=n_estimators,
                    rf_max_depth=max_depth if max_depth > 0 else None,
                )
                st.session_state["training_results"] = results
                preferred_model = (
                    "Random Forest Classifier"
                    if "Random Forest Classifier" in results["models"]
                    else results["best_model_name"]
                )
                active_art = results["models"][preferred_model]["artifact"]
                st.session_state["active_artifact"] = active_art
                st.session_state["active_model_source"] = "Current Session (Trained)"

                # Save both trained models and highlight the active Random Forest model
                for m_res in results["models"].values():
                    save_model_artifact(m_res["artifact"])
                saved_path = save_model_artifact(active_art)
                st.success(
                    f"Training complete! Top evaluated model: **{results['best_model_name']}**. "
                    f"Active prediction model set to **{preferred_model}** (saved to `{saved_path.name}`)."
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    # Load Previously Saved Model Option
    with st.expander("💾 Load or Manage Saved Joblib Models", expanded=False):
        saved_models = list_saved_models()
        if saved_models:
            chosen_file = st.selectbox(
                "Saved Models in `models/`",
                options=saved_models,
                format_func=lambda p: p.name,
            )
            if st.button("Load Selected Model for Prediction"):
                try:
                    loaded_art = load_model_artifact(chosen_file)
                    st.session_state["active_artifact"] = loaded_art
                    st.session_state["active_model_source"] = chosen_file.name
                    st.success(f"Loaded `{loaded_art['model_name']}` from `{chosen_file.name}`.")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.caption("No `.joblib` models saved in `models/` yet. Train models above to save automatically.")

    results = st.session_state.get("training_results")
    if not results:
        st.info("Click **Train Baseline & Random Forest Models** above to run the 80:20 split, cross-validation, and evaluation.")
        return

    # 2. Split & Comparison Overview
    split_info = results["split_info"]
    render_section_title("2. Model Comparison (80:20 Split + Cross-Validation)")
    st.caption(
        f"Total Cleaned Samples: **{split_info['total_samples']:,}** | "
        f"Training Set (80%): **{split_info['train_samples']:,}** | "
        f"Testing Set (20%): **{split_info['test_samples']:,}** | "
        f"Features Used: **{split_info['feature_count']}** (Excluded IDs: {', '.join(split_info['excluded_id_columns']) or 'None'})"
    )
    st.dataframe(results["comparison_df"], use_container_width=True, hide_index=True)

    # 3. Side-by-Side Detailed Evaluation & Confusion Matrices for Both Models
    render_section_title("3. Detailed Evaluation & Confusion Matrices (Both Models)")
    model_names = list(results["models"].keys())

    eval_cols = st.columns(len(model_names))
    for col_widget, m_name in zip(eval_cols, model_names):
        mdata = results["models"][m_name]
        metrics = mdata["metrics"]
        with col_widget:
            st.markdown(f"#### {m_name}")
            r1a, r1b, r1c = st.columns(3)
            r1a.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            r1b.metric("Precision", f"{metrics['precision']*100:.2f}%")
            r1c.metric("Recall", f"{metrics['recall']*100:.2f}%")

            r2a, r2b, r2c = st.columns(3)
            r2a.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
            r2b.metric("CV Accuracy", f"{metrics['cv_accuracy_mean']*100:.2f}% ± {metrics['cv_accuracy_std']*100:.2f}%")
            r2c.metric("CV F1-Score", f"{metrics['cv_f1_mean']*100:.2f}% ± {metrics['cv_f1_std']*100:.2f}%")

            fig_cm = plot_confusion_matrix_figure(mdata["confusion_matrix"], m_name)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            fig_imp = plot_feature_importance_figure(mdata["feature_importance"], m_name)
            st.pyplot(fig_imp)
            plt.close(fig_imp)

            with st.expander(f"Classification Report — {m_name}"):
                rep_df = pd.DataFrame(mdata["classification_report"]).T.round(4)
                st.dataframe(rep_df, use_container_width=True)

    # Save / Activate selected model explicitly
    render_section_title("4. Save or Activate Trained Model")
    default_m_idx = model_names.index(results["best_model_name"])
    selected_model_name = st.selectbox("Select Model to Save / Activate for Predictions", options=model_names, index=default_m_idx)
    selected_mdata = results["models"][selected_model_name]

    act_c1, act_c2 = st.columns(2)
    with act_c1:
        if st.button(f"💾 Save & Activate {selected_model_name}", use_container_width=True):
            path = save_model_artifact(selected_mdata["artifact"])
            st.session_state["active_artifact"] = selected_mdata["artifact"]
            st.session_state["active_model_source"] = path.name
            st.success(f"Saved `{path}` and set `{selected_model_name}` as the active prediction model.")
    with act_c2:
        model_path = Path("models") / (
            selected_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        )
        if not model_path.exists():
            model_path = save_model_artifact(selected_mdata["artifact"])
        with open(model_path, "rb") as f:
            st.download_button(
                "⬇️ Download Model File (.joblib)",
                data=f.read(),
                file_name=model_path.name,
                mime="application/octet-stream",
                use_container_width=True,
            )


# ==============================================================================
# PAGE 5: CUSTOMER PREDICTION
# ==============================================================================
def render_prediction_page() -> None:
    render_hero_banner(
        "Customer Retention Prediction",
        "Predict retention vs. churn risk for an individual customer or score hundreds of records via batch CSV upload.",
    )

    artifact = st.session_state.get("active_artifact")
    if artifact is None:
        saved = list_saved_models()
        st.warning("No trained ML model is currently loaded in session.")
        c1, c2 = st.columns(2)
        with c1:
            if saved:
                chosen = st.selectbox("Select a saved model to load", options=saved, format_func=lambda p: p.name)
                if st.button("Load Saved Model", type="primary"):
                    st.session_state["active_artifact"] = load_model_artifact(chosen)
                    st.session_state["active_model_source"] = chosen.name
                    st.rerun()
        with c2:
            st.info("Or visit **Model Training & Evaluation** to train a fresh Random Forest & Baseline model on your dataset.")
        return

    training_res = st.session_state.get("training_results")
    if training_res and "models" in training_res:
        trained_names = list(training_res["models"].keys())
        curr_idx = trained_names.index(artifact["model_name"]) if artifact["model_name"] in trained_names else 0
        chosen_m = st.selectbox(
            "Prediction Model",
            options=trained_names,
            index=curr_idx,
            help="Defaults to the trained Random Forest Classifier; switch if you wish to compare with Baseline Logistic Regression.",
        )
        if chosen_m != artifact["model_name"]:
            artifact = training_res["models"][chosen_m]["artifact"]
            st.session_state["active_artifact"] = artifact
            st.session_state["last_single_prediction"] = None
            st.session_state["batch_result"] = None

    st.info(
        f"**Active Prediction Model:** {artifact['model_name']} | "
        f"**Test Accuracy:** {artifact['metrics']['accuracy']*100:.2f}% | "
        f"**F1-Score:** {artifact['metrics']['f1_score']*100:.2f}%"
    )

    tab_single, tab_batch = st.tabs(["👤 Individual Customer Prediction", "📦 Batch CSV Prediction"])

    # --- TAB 1: SINGLE CUSTOMER ---
    with tab_single:
        schema: Dict[str, Dict[str, Any]] = artifact["feature_schema"]
        feature_cols: List[str] = artifact["feature_columns"]

        with st.form("single_customer_prediction_form"):
            st.markdown("Enter customer attributes below:")
            cols_per_row = 3
            user_inputs: Dict[str, Any] = {}

            for i in range(0, len(feature_cols), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col_name in enumerate(feature_cols[i : i + cols_per_row]):
                    meta = schema[col_name]
                    with row_cols[j]:
                        if meta["type"] == "numeric":
                            if meta["is_int"]:
                                user_inputs[col_name] = st.number_input(
                                    col_name,
                                    min_value=int(meta["min"]),
                                    max_value=int(max(meta["max"] * 1.5, meta["min"] + 10)),
                                    value=int(round(meta["median"])),
                                    step=1,
                                )
                            else:
                                user_inputs[col_name] = st.number_input(
                                    col_name,
                                    min_value=float(meta["min"]),
                                    max_value=float(max(meta["max"] * 1.5, meta["min"] + 10.0)),
                                    value=round(float(meta["median"]), 2),
                                    step=1.0,
                                )
                        else:
                            cats = meta["categories"] or ["Unknown"]
                            default_idx = cats.index(meta["default"]) if meta["default"] in cats else 0
                            user_inputs[col_name] = st.selectbox(
                                col_name,
                                options=cats,
                                index=default_idx,
                            )

            submitted = st.form_submit_button("🔮 Predict Customer Retention", type="primary", use_container_width=True)

        if submitted:
            try:
                pred = predict_single_customer(artifact, user_inputs)
                st.session_state["last_single_prediction"] = {"inputs": user_inputs, "result": pred}
            except Exception as exc:
                st.error(f"Prediction error: {exc}")

        last_pred = st.session_state.get("last_single_prediction")
        if last_pred:
            res = last_pred["result"]
            render_section_title("Prediction Result & Probability Estimate")

            if res["is_at_risk"]:
                st.markdown(
                    f"""
                    <div class="pred-box-churn">
                        <h3 style="margin:0 0 6px 0;">⚠️ Prediction: {res['status_display']} ({res['risk_tier']})</h3>
                        <p style="margin:0;">Estimated Churn Risk Probability: <strong>{res['churn_probability']:.2f}%</strong> &nbsp;|&nbsp; Retention Probability: <strong>{res['retention_probability']:.2f}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="pred-box-retained">
                        <h3 style="margin:0 0 6px 0;">✅ Prediction: {res['status_display']} ({res['risk_tier']})</h3>
                        <p style="margin:0;">Estimated Retention Probability: <strong>{res['retention_probability']:.2f}%</strong> &nbsp;|&nbsp; Churn Risk Probability: <strong>{res['churn_probability']:.2f}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted Class", res["status_display"])
            p2.metric("Retention Probability", f"{res['retention_probability']:.2f}%")
            p3.metric("Churn Risk Probability", f"{res['churn_probability']:.2f}%")

            st.progress(min(max(res["churn_probability"] / 100.0, 0.0), 1.0), text=f"Churn Risk Meter: {res['churn_probability']:.1f}%")

            st.markdown("**Recommended Retention Actions:**")
            for tip in res["recommendations"]:
                st.write(f"- {tip}")

    # --- TAB 2: BATCH CSV PREDICTION ---
    with tab_batch:
        st.markdown(
            f"Upload a CSV containing customer records (expected feature columns: `{', '.join(artifact['feature_columns'])}`)."
        )

        # Provide a convenient downloadable sample batch template from the active dataset
        raw_df = st.session_state.get("raw_df")
        if raw_df is not None:
            template_cols = [c for c in raw_df.columns if c != artifact["target_col"]]
            template_bytes = raw_df[template_cols].head(50).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Sample Batch Input CSV (50 Customers)",
                data=template_bytes,
                file_name="sample_batch_customers.csv",
                mime="text/csv",
            )

        batch_file = st.file_uploader("Upload Batch Customer CSV", type=["csv"], key="batch_csv_uploader")
        if batch_file is not None and st.button("🚀 Run Batch Predictions", type="primary"):
            try:
                batch_df = load_uploaded_csv(batch_file)
                batch_res = predict_batch_customers(artifact, batch_df)
                st.session_state["batch_result"] = batch_res
                st.success(f"Batch prediction complete for {batch_res['summary']['total_customers']:,} customers!")
            except Exception as exc:
                st.error(f"Batch prediction error: {exc}")

        batch_res = st.session_state.get("batch_result")
        if batch_res:
            s = batch_res["summary"]
            if batch_res["missing_feature_columns"]:
                st.warning(
                    f"Note: Some model columns were missing in the uploaded file and imputed using training defaults: "
                    f"{', '.join(batch_res['missing_feature_columns'])}"
                )

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                render_kpi_card("Scored Customers", f"{s['total_customers']:,}", "Total rows processed", "slate")
            with b2:
                render_kpi_card("Predicted Retained", f"{s['retained_predicted']:,}", f"{s['predicted_retention_rate']}%", "teal")
            with b3:
                render_kpi_card("Predicted At-Risk", f"{s['at_risk_predicted']:,}", f"{s['predicted_churn_rate']}%", "rose")
            with b4:
                render_kpi_card("High-Risk Segment", f"{s['high_risk_count']:,}", "Churn prob >= 65%", "amber")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.dataframe(batch_res["predictions_df"], use_container_width=True)

            out_csv = batch_res["predictions_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Batch Prediction Results (CSV)",
                data=out_csv,
                file_name="customer_retention_predictions.csv",
                mime="text/csv",
                type="primary",
            )


# ==============================================================================
# PAGE 6: REPORTS & EXPORT
# ==============================================================================
def render_reports_page() -> None:
    render_hero_banner(
        "Project Reports & Export",
        "Generate comprehensive downloadable evaluation reports containing dataset diagnostics, preprocessing audit, EDA charts, model comparison, and prediction summaries.",
    )

    raw_df = st.session_state.get("raw_df")
    eda_df = st.session_state.get("eda_df")
    target_col = st.session_state.get("target_col")
    mapping = st.session_state.get("target_mapping")

    if raw_df is None or target_col is None or mapping is None:
        st.warning("Load a dataset first on **Dataset Analysis** to generate reports.")
        return

    active_df = eda_df if eda_df is not None else raw_df
    inspection = inspect_dataset(raw_df)
    total_c, ret_c, risk_c, ret_rate, churn_rate = compute_kpi_metrics(
        active_df, target_col, mapping["retained_label"], mapping["churned_label"]
    )
    kpi_stats = {
        "total_customers": total_c,
        "retained_customers": ret_c,
        "at_risk_customers": risk_c,
        "retention_rate": ret_rate,
        "churn_rate": churn_rate,
    }

    training_results = st.session_state.get("training_results")
    batch_res = st.session_state.get("batch_result")
    prep_log = st.session_state.get("preprocessing_log")
    dataset_name = st.session_state.get("dataset_name", "Customer Dataset")

    include_charts = st.checkbox("Include Live EDA & Confusion Matrix Charts in HTML Report", value=True)

    if st.button("📄 Generate Project Report", type="primary"):
        with st.spinner("Compiling live charts, metrics, and report sections..."):
            chart_images = []
            if include_charts:
                ret_lbl = mapping["retained_label"]
                churn_lbl = mapping["churned_label"]

                for title, fn in [
                    ("Customer Retention Distribution", lambda: plot_retention_distribution(active_df, target_col, ret_lbl, churn_lbl)),
                    ("Customer Churn Distribution", lambda: plot_churn_distribution(active_df, target_col, ret_lbl, churn_lbl)),
                ]:
                    fig = fn()
                    chart_images.append((title, figure_to_base64(fig)))
                    plt.close(fig)

                try:
                    fig_t = plot_retention_by_tenure(active_df, target_col, ret_lbl, churn_lbl)
                    chart_images.append(("Retention by Tenure", figure_to_base64(fig_t)))
                    plt.close(fig_t)
                except ValueError:
                    pass

                try:
                    fig_m = plot_retention_by_monthly_charges(active_df, target_col, ret_lbl, churn_lbl)
                    chart_images.append(("Retention by Monthly Charges", figure_to_base64(fig_m)))
                    plt.close(fig_m)
                except ValueError:
                    pass

                try:
                    fig_c = plot_retention_by_contract_type(active_df, target_col, ret_lbl, churn_lbl)
                    chart_images.append(("Retention by Contract Type", figure_to_base64(fig_c)))
                    plt.close(fig_c)
                except ValueError:
                    pass

                try:
                    fig_h = plot_correlation_heatmap(active_df, target_col, churn_lbl, exclude_cols=st.session_state.get("id_cols", []))
                    chart_images.append(("Correlation Heatmap", figure_to_base64(fig_h)))
                    plt.close(fig_h)
                except ValueError:
                    pass

                if training_results:
                    best_name = training_results["best_model_name"]
                    cm = training_results["models"][best_name]["confusion_matrix"]
                    fig_cm = plot_confusion_matrix_figure(cm, best_name)
                    chart_images.append((f"Confusion Matrix ({best_name})", figure_to_base64(fig_cm)))
                    plt.close(fig_cm)

                    imp_df = training_results["models"][best_name]["feature_importance"]
                    fig_imp = plot_feature_importance_figure(imp_df, best_name)
                    chart_images.append((f"Feature Importance ({best_name})", figure_to_base64(fig_imp)))
                    plt.close(fig_imp)

            html_report = generate_html_report(
                dataset_name=dataset_name,
                kpi_stats=kpi_stats,
                inspection=inspection,
                preprocessing_log=prep_log,
                chart_images=chart_images,
                training_results=training_results,
                batch_summary=batch_res["summary"] if batch_res else None,
                batch_preview_df=batch_res["predictions_df"] if batch_res else None,
            )
            md_report = generate_markdown_report(
                dataset_name=dataset_name,
                kpi_stats=kpi_stats,
                inspection=inspection,
                preprocessing_log=prep_log,
                training_results=training_results,
                batch_summary=batch_res["summary"] if batch_res else None,
            )

            st.session_state["generated_html_report"] = html_report
            st.session_state["generated_md_report"] = md_report

            # Save a copy to reports/ directory on disk as well
            reports_dir = Path("reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "latest_retention_report.html").write_text(html_report, encoding="utf-8")
            (reports_dir / "latest_retention_report.md").write_text(md_report, encoding="utf-8")

            st.success("Report generated successfully and saved to `reports/`!")

    if "generated_html_report" in st.session_state:
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Download Full HTML Report (Open in Browser / Print to PDF)",
                data=st.session_state["generated_html_report"].encode("utf-8"),
                file_name="customer_retention_analytics_report.html",
                mime="text/html",
                type="primary",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Download Markdown Report (.md)",
                data=st.session_state["generated_md_report"].encode("utf-8"),
                file_name="customer_retention_summary_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with st.expander("Preview Markdown Report Summary", expanded=True):
            st.markdown(st.session_state["generated_md_report"])


# ==============================================================================
# MAIN ROUTER
# ==============================================================================
def main() -> None:
    _init_session_state()
    selected_page = _render_sidebar()

    if selected_page == "🏠 Dashboard Overview":
        render_dashboard_overview()
    elif selected_page == "🗂️ Dataset Analysis":
        render_dataset_analysis()
    elif selected_page == "📊 Data Visualization (EDA)":
        render_eda_page()
    elif selected_page == "🤖 Model Training & Evaluation":
        render_model_training_page()
    elif selected_page == "🎯 Customer Prediction":
        render_prediction_page()
    elif selected_page == "📄 Reports & Export":
        render_reports_page()


if __name__ == "__main__":
    main()

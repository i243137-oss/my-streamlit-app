import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Interface", layout="wide")
st.title("📊 Exploratory Data Analysis Interface")

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        selected_column = st.sidebar.selectbox("Select Column for Analysis", options=df.columns)

        st.header("1. Dataset Overview & Preview")
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("Metadata Summary")
        col1, col2 = st.columns(2)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])

        meta_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Missing Values Count": df.isnull().sum(),
            "Missing Values (%)": (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(meta_df, use_container_width=True)

        st.subheader("Numerical Statistical Summary")
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            stats_df = df[num_cols].describe().T[["mean", "50%", "min", "max"]]
            stats_df.rename(columns={"50%": "median"}, inplace=True)
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No numerical attributes detected.")

        st.divider()

        st.header(f"2. Visual Analysis: `{selected_column}`")
        col_data = df[selected_column]
        is_numeric = pd.api.types.is_numeric_dtype(col_data) and col_data.nunique() > 10

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.set_theme(style="whitegrid")

        if is_numeric:
            st.markdown("**Detected Type:** `Numerical`")
            sns.histplot(col_data.dropna(), kde=True, ax=ax, color="steelblue")
            ax.set_title(f"Histogram of {selected_column}")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Frequency")
        else:
            st.markdown("**Detected Type:** `Categorical`")
            order = col_data.value_counts(dropna=False).index
            sns.countplot(data=df, x=selected_column, order=order, ax=ax, palette="viridis")
            ax.set_title(f"Frequency Count of {selected_column}")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Count")

            total = len(df)
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    percentage = f"{100 * height / total:.1f}%"
                    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                                textcoords='offset points')

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error parsing file: {e}")

else:
    st.info("Please upload a CSV file via the sidebar to start.")

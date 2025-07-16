# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.linear_model import LinearRegression
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import r2_score, mean_squared_error

# # st.set_page_config(page_title="📊 Visual EDA + ML", layout="wide")
# # st.title("📊 Visual EDA & Basic Machine Learning App")

# # # Upload file
# # uploaded_file = st.file_uploader("📤 Upload CSV or Excel", type=["csv", "xlsx"])

# # if uploaded_file:
# #     ext = uploaded_file.name.split(".")[-1]
# #     try:
# #         if ext == "csv":
# #             df = pd.read_csv(uploaded_file)
# #         else:
# #             df = pd.read_excel(uploaded_file, engine="openpyxl")
# #     except Exception as e:
# #         st.error(f"Error loading file: {e}")
# #         st.stop()

# #     st.subheader("🔍 Data Preview")
# #     st.dataframe(df.head())

# #     # ---- Data Cleaning ----
# #     st.subheader("🧹 Data Cleaning")
# #     if st.checkbox("Drop NA values"):
# #         df.dropna(inplace=True)
# #         st.success("Dropped NA rows.")

# #     if st.checkbox("Drop duplicate rows"):
# #         df.drop_duplicates(inplace=True)
# #         st.success("Dropped duplicates.")

# #     if st.checkbox("Encode categorical columns"):
# #         df = pd.get_dummies(df)
# #         st.success("Categoricals encoded.")

# #     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# #     # ---- Visual EDA ----
# #     st.subheader("📊 Visual EDA")

# #     plot_type = st.selectbox("Choose Plot Type", [
# #         "Scatter Plot", "Line Plot", "Box Plot", "Histogram", "Heatmap", "Pairplot"
# #     ])

# #     if plot_type in ["Scatter Plot", "Line Plot"]:
# #         x_col = st.selectbox("Select X-axis", numeric_cols)
# #         y_col = st.selectbox("Select Y-axis", [col for col in numeric_cols if col != x_col])

# #         fig, ax = plt.subplots(figsize=(10, 6))
# #         if plot_type == "Scatter Plot":
# #             ax.scatter(df[x_col], df[y_col])
# #             ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
# #         else:
# #             ax.plot(df[x_col], df[y_col])
# #             ax.set_title(f"Line Plot: {x_col} vs {y_col}")
# #         ax.set_xlabel(x_col)
# #         ax.set_ylabel(y_col)
# #         st.pyplot(fig)

# #     elif plot_type == "Box Plot":
# #         y_col = st.selectbox("Select column for Box Plot", numeric_cols)
# #         fig, ax = plt.subplots()
# #         sns.boxplot(y=df[y_col], ax=ax)
# #         ax.set_title(f"Box Plot: {y_col}")
# #         st.pyplot(fig)

# #     elif plot_type == "Histogram":
# #         col = st.selectbox("Select column for Histogram", numeric_cols)
# #         fig, ax = plt.subplots()
# #         ax.hist(df[col], bins=30, color='skyblue', edgecolor='black')
# #         ax.set_title(f"Histogram: {col}")
# #         st.pyplot(fig)

# #     elif plot_type == "Heatmap":
# #         st.write("🔻 Correlation Heatmap")
# #         fig, ax = plt.subplots(figsize=(10, 6))
# #         corr = df[numeric_cols].corr()
# #         sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
# #         st.pyplot(fig)

# #     elif plot_type == "Pairplot":
# #         st.write("🔻 Pairwise Relationships")
# #         selected = st.multiselect("Select up to 4 features", numeric_cols, default=numeric_cols[:3])
# #         if len(selected) >= 2:
# #             fig = sns.pairplot(df[selected])
# #             st.pyplot(fig)

# #     # ---- Linear Regression ----
# #     st.subheader("📈 Linear Regression")

# #     if len(numeric_cols) >= 2:
# #         target = st.selectbox("🎯 Target Variable", numeric_cols)
# #         features = st.multiselect("🧩 Features", [col for col in numeric_cols if col != target])

# #         if features and target:
# #             X = df[features]
# #             y = df[target]

# #             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #             model = LinearRegression()
# #             model.fit(X_train, y_train)
# #             y_pred = model.predict(X_test)

# #             st.write("**Model Metrics:**")
# #             st.write(f"✅ R² Score: {r2_score(y_test, y_pred):.4f}")
# #             st.write(f"✅ MSE: {mean_squared_error(y_test, y_pred):.4f}")

# #             st.subheader("🔢 Predictions vs Actual")
# #             pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
# #             st.dataframe(pred_df.head())

# #             st.line_chart(pred_df)
# #     else:
# #         st.warning("Need at least two numeric columns.")
# # else:
# #     st.info("👆 Upload a file to begin.")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error

# st.set_page_config(page_title="📊 Visual EDA + ML", layout="wide")
# st.title("📊 Basic Machine Learning App")

# # Upload file
# uploaded_file = st.file_uploader("📤 Upload CSV or Excel", type=["csv", "xlsx"])

# if uploaded_file:
#     ext = uploaded_file.name.split(".")[-1]
#     try:
#         if ext == "csv":
#             df = pd.read_csv(uploaded_file)
#         else:
#             df = pd.read_excel(uploaded_file, engine="openpyxl")
#     except Exception as e:
#         st.error(f"Error loading file: {e}")
#         st.stop()

#     st.subheader("🔍 Data Preview")
#     st.dataframe(df.head())

#     # ---- Data Cleaning ----
#     st.subheader("🧹 Data Cleaning")
#     if st.checkbox("Drop NA values"):
#         df.dropna(inplace=True)
#         st.success("Dropped NA rows.")

#     if st.checkbox("Drop duplicate rows"):
#         df.drop_duplicates(inplace=True)
#         st.success("Dropped duplicates.")

#     if st.checkbox("Encode categorical columns"):
#         df = pd.get_dummies(df)
#         st.success("Categoricals encoded.")

#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

#     # ---- Visual EDA ----
#     st.subheader("📊 Visual EDA")

#     plot_type = st.selectbox("Choose Plot Type", [
#         "Scatter Plot", "Line Plot", "Box Plot", "Histogram", "Heatmap", "Pairplot"
#     ])

#     if plot_type in ["Scatter Plot", "Line Plot"]:
#         x_col = st.selectbox("Select X-axis", numeric_cols)
#         y_col = st.selectbox("Select Y-axis", [col for col in numeric_cols if col != x_col])

#         fig, ax = plt.subplots(figsize=(7, 5))
#         if plot_type == "Scatter Plot":
#             ax.scatter(df[x_col], df[y_col])
#             ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
#         else:
#             ax.plot(df[x_col], df[y_col])
#             ax.set_title(f"Line Plot: {x_col} vs {y_col}")
#         ax.set_xlabel(x_col)
#         ax.set_ylabel(y_col)
#         fig.tight_layout()
#         st.pyplot(fig)

#     elif plot_type == "Box Plot":
#         y_col = st.selectbox("Select column for Box Plot", numeric_cols)
#         fig, ax = plt.subplots(figsize=(7, 5))
#         sns.boxplot(y=df[y_col], ax=ax)
#         ax.set_title(f"Box Plot: {y_col}")
#         fig.tight_layout()
#         st.pyplot(fig)

#     elif plot_type == "Histogram":
#         col = st.selectbox("Select column for Histogram", numeric_cols)
#         fig, ax = plt.subplots(figsize=(7, 5))
#         ax.hist(df[col], bins=30, color='skyblue', edgecolor='black')
#         ax.set_title(f"Histogram: {col}")
#         fig.tight_layout()
#         st.pyplot(fig)

#     elif plot_type == "Heatmap":
#         st.write("🔻 Correlation Heatmap")
#         fig, ax = plt.subplots(figsize=(8, 6))
#         corr = df[numeric_cols].corr()
#         sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
#         fig.tight_layout()
#         st.pyplot(fig)

#     elif plot_type == "Pairplot":
#         st.write("🔻 Pairwise Relationships")
#         selected = st.multiselect("Select up to 4 features", numeric_cols, default=numeric_cols[:3])
#         if len(selected) >= 2:
#             fig = sns.pairplot(df[selected], height=2.5)
#             st.pyplot(fig)

#     # ---- Linear Regression ----
#     st.subheader("📈 Linear Regression")

#     if len(numeric_cols) >= 2:
#         target = st.selectbox("🎯 Target Variable", numeric_cols)
#         features = st.multiselect("🧩 Features", [col for col in numeric_cols if col != target])

#         if features and target:
#             X = df[features]
#             y = df[target]

#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model = LinearRegression()
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             st.write("**Model Metrics:**")
#             st.write(f"✅ R² Score: {r2_score(y_test, y_pred):.4f}")
#             st.write(f"✅ MSE: {mean_squared_error(y_test, y_pred):.4f}")

#             st.subheader("🔢 Predictions vs Actual")
#             pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#             st.dataframe(pred_df.head())

#             st.line_chart(pred_df)
#     else:
#         st.warning("Need at least two numeric columns.")
# else:
#     st.info("👆 Upload a file to begin.")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="📊 Visual EDA + ML", layout="wide")
st.title("📊 Basic Machine Learning App")

# Upload file
uploaded_file = st.file_uploader("📤 Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1]
    try:
        if ext == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # ---- Data Cleaning ----
    st.subheader("🧹 Data Cleaning")
    if st.checkbox("Drop NA values"):
        df.dropna(inplace=True)
        st.success("Dropped NA rows.")

    if st.checkbox("Drop duplicate rows"):
        df.drop_duplicates(inplace=True)
        st.success("Dropped duplicates.")

    if st.checkbox("Encode categorical columns"):
        df = pd.get_dummies(df)
        st.success("Categoricals encoded.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ---- Visual EDA ----
    st.subheader("📊 Visual EDA")

    plot_type = st.selectbox("Choose Plot Type", [
        "Scatter Plot", "Line Plot", "Box Plot", "Histogram", "Heatmap", "Pairplot"
    ])

    if plot_type in ["Scatter Plot", "Line Plot"]:
        x_col = st.selectbox("Select X-axis", numeric_cols)
        y_col = st.selectbox("Select Y-axis", [col for col in numeric_cols if col != x_col])

        fig, ax = plt.subplots(figsize=(7, 5))
        if plot_type == "Scatter Plot":
            ax.scatter(df[x_col], df[y_col])
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        else:
            ax.plot(df[x_col], df[y_col])
            ax.set_title(f"Line Plot: {x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.tight_layout()
        st.pyplot(fig)

    elif plot_type == "Box Plot":
        y_col = st.selectbox("Select column for Box Plot", numeric_cols)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(y=df[y_col], ax=ax)
        ax.set_title(f"Box Plot: {y_col}")
        fig.tight_layout()
        st.pyplot(fig)

    elif plot_type == "Histogram":
        col = st.selectbox("Select column for Histogram", numeric_cols)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df[col], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Histogram: {col}")
        fig.tight_layout()
        st.pyplot(fig)

    elif plot_type == "Heatmap":
        st.write("🔻 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        fig.tight_layout()
        st.pyplot(fig)

    elif plot_type == "Pairplot":
        st.write("🔻 Pairwise Relationships")
        selected = st.multiselect("Select up to 4 features", numeric_cols, default=numeric_cols[:3])
        if len(selected) >= 2:
            fig = sns.pairplot(df[selected], height=2.5)
            st.pyplot(fig)

    # ---- Linear Regression ----
    st.subheader("📈 Linear Regression")

    trained_model = None
    selected_features = []
    if len(numeric_cols) >= 2:
        target = st.selectbox("🎯 Target Variable", numeric_cols)
        features = st.multiselect("🧩 Features", [col for col in numeric_cols if col != target])

        if features and target:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("**Model Metrics:**")
            st.write(f"✅ R² Score: {r2_score(y_test, y_pred):.4f}")
            st.write(f"✅ MSE: {mean_squared_error(y_test, y_pred):.4f}")

            st.subheader("🔢 Predictions vs Actual")
            pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.dataframe(pred_df.head())

            st.line_chart(pred_df)

            # Save model state and features
            trained_model = model
            selected_features = features
    else:
        st.warning("Need at least two numeric columns.")

    # ---- Prediction Section ----
    st.subheader("🧠 Predict Output for New Input")

    if trained_model and selected_features:
        st.info(f"Enter values for features: {', '.join(selected_features)}")

        user_input = st.text_input("Enter values (comma-separated if multiple):")

        if user_input:
            try:
                values = list(map(float, user_input.split(",")))
                if len(values) != len(selected_features):
                    st.error(f"❌ Please enter {len(selected_features)} values.")
                else:
                    input_array = np.array(values).reshape(1, -1)
                    prediction = trained_model.predict(input_array)[0]
                    st.success(f"🔮 Predicted Output: **{prediction:.4f}**")
            except ValueError:
                st.error("❌ Invalid input. Please enter numeric values only.")
    elif uploaded_file:
        st.warning("⚠️ Train a model first by selecting features and target.")
else:
    st.info("👆 Upload a file to begin.")

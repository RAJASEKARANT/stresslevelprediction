import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(
    page_title="Human Stress Level Prediction System",
    page_icon="",
    layout="wide"
)

# Title and Description
st.markdown("<h1 style='text-align: center; font-weight: bold;'>🧠 HUMAN STRESS DETECTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("""
This application predicts your stress level based on various health metrics using a **Random Forest Classifier**.
Adjust the sliders in the sidebar to input your data and see the prediction along with detailed visualizations.
""")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("stress.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset 'stress.csv' not found. Please ensure it exists in the directory.")
        return None

df = load_data()

if df is not None:
    # Model Training
    X = df.drop('stress_level', axis=1)
    y = df['stress_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Sidebar Inputs
    st.sidebar.header("📝 User Health Metrics")

    def user_input_features():
        heart_rate = st.sidebar.slider("Heart Rate (bpm)", int(df['heart_rate'].min()), int(df['heart_rate'].max()), int(df['heart_rate'].mean()))
        sleeping_hours = st.sidebar.slider("Sleeping Hours", float(df['sleeping_hours'].min()), float(df['sleeping_hours'].max()), float(df['sleeping_hours'].mean()))
        sleep_quality = st.sidebar.slider("Sleep Quality (1-10)", int(df['sleep_quality'].min()), int(df['sleep_quality'].max()), int(df['sleep_quality'].mean()))
        blood_oxygen = st.sidebar.slider("Blood Oxygen (SpO2 %)", float(df['blood_oxygen'].min()), float(df['blood_oxygen'].max()), float(df['blood_oxygen'].mean()))
        body_temperature = st.sidebar.slider("Body Temperature (°C)", float(df['body_temperature'].min()), float(df['body_temperature'].max()), float(df['body_temperature'].mean()))
        physical_activity = st.sidebar.slider("Physical Activity (score)", int(df['physical_activity'].min()), int(df['physical_activity'].max()), int(df['physical_activity'].mean()))

        data = {
            'heart_rate': heart_rate,
            'sleeping_hours': sleeping_hours,
            'sleep_quality': sleep_quality,
            'blood_oxygen': blood_oxygen,
            'body_temperature': body_temperature,
            'physical_activity': physical_activity
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # Predict Button
    if st.sidebar.button("Predict Stress Level"):
        prediction = rf_model.predict(input_df)
        prediction_proba = rf_model.predict_proba(input_df)

        stress_labels = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
        predicted_label = stress_labels[prediction[0]]

        st.subheader("🔮 Prediction Result")
        if prediction[0] == 0:
            st.success(f"Prediction: **{predicted_label}**")
        elif prediction[0] == 1:
            st.warning(f"Prediction: **{predicted_label}**")
        else:
            st.error(f"Prediction: **{predicted_label}**")

        st.write(f"Model Accuracy: {accuracy:.2f}")

        if accuracy < 0.70:
            st.warning("Note: Model accuracy is below 70%. Predictions might be less reliable.")

        # --- Visualizations ---

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gauge Chart: Stress Score")
            # Calculate a "score" for the gauge based on probability of high stress or weighted average
            # Simple approach: Weighted sum of probabilities (0*Low + 1*Med + 2*High) / 2 to get 0-1 scale?
            # Or just use the probability of the predicted class?
            # Let's use a weighted score for smooth gauge:
            # Score = (Prob(Med) * 0.5 + Prob(High) * 1.0) * 100

            probs = prediction_proba[0]
            # Assumes classes are [0, 1, 2]
            gauge_value = (probs[1] * 50 + probs[2] * 100)

            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = gauge_value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Stress Severity Score (0-100)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': gauge_value
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.subheader("Radar Chart: Input vs Average")
            # Prepare data for Radar Chart
            # Normalize data for better visualization in Radar chart
            categories = list(input_df.columns)

            # Get average values from dataset
            avg_values = df.drop('stress_level', axis=1).mean()

            # Normalize to max values in dataset for scaling (0-1)
            max_values = df.drop('stress_level', axis=1).max()

            input_norm = input_df.iloc[0] / max_values
            avg_norm = avg_values / max_values

            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(
                r=input_norm.values,
                theta=categories,
                fill='toself',
                name='Your Input'
            ))

            fig_radar.add_trace(go.Scatterpolar(
                r=avg_norm.values,
                theta=categories,
                fill='toself',
                name='Dataset Average'
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("Feature Importance")
        # Feature Importance
        importances = rf_model.feature_importances_
        feature_names = X.columns
        feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

        # Seaborn Bar Plot
        fig_imp, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis', ax=ax)
        ax.set_title('Feature Importance in Stress Prediction')
        st.pyplot(fig_imp)

        st.markdown("---")
        st.subheader("📊 Additional Analysis")

        # Input Summary Table
        st.markdown("### Input Summary vs Healthy Baseline")

        # Define baselines
        baselines = {
            'heart_rate': 72, # BPM
            'sleeping_hours': 8.0, # Hours
            'sleep_quality': 8, # Score
            'blood_oxygen': 98.0, # %
            'body_temperature': 36.6, # C
            'physical_activity': 75 # Score
        }

        comparison_data = []
        for col in input_df.columns:
            user_val = input_df[col].iloc[0]
            base_val = baselines.get(col, 0)
            diff = user_val - base_val

            # Formatting
            if col in ['sleeping_hours', 'blood_oxygen', 'body_temperature']:
                 user_val_fmt = f"{user_val:.2f}"
                 diff_fmt = f"{diff:.2f}"
            else:
                 user_val_fmt = f"{int(user_val)}"
                 diff_fmt = f"{int(diff)}"

            comparison_data.append({
                'Metric': col.replace('_', ' ').title(),
                'Your Value': user_val_fmt,
                'Healthy Baseline': base_val,
                'Difference': diff_fmt
            })

        comp_df = pd.DataFrame(comparison_data)
        st.table(comp_df)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Correlation Heatmap")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            # Calculate correlation only for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)

        with col4:
            st.subheader("Stress Level Distribution")
            # Scatter Plot
            # Determine suitable axes or let user choose? For now fixed.
            # Heart Rate vs Sleep Quality is usually a good indicator
            fig_scatter = px.scatter(
                df,
                x='heart_rate',
                y='sleep_quality',
                color='stress_level',
                title="Heart Rate vs Sleep Quality",
                labels={'stress_level': 'Stress Level'},
                color_continuous_scale='Bluered'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.info("Adjust the sliders in the sidebar and click 'Predict Stress Level' to see results.")

        # Show dataset sample
        st.subheader("Dataset Sample")
        st.write(df.head())

else:
    st.stop()

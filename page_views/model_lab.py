import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import advanced as A


def _model_label(model):
    name = type(model).__name__
    return {
        "LGBMClassifier": "LightGBM", "LGBMRegressor": "LightGBM",
        "XGBClassifier": "XGBoost", "XGBRegressor": "XGBoost",
        "RandomForestClassifier": "Random Forest", "RandomForestRegressor": "Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting", "LinearRegression": "Linear Regression",
    }.get(name, name)


def render(data):
    st.markdown("## 🧪 Model Performance")
    st.caption("Every candidate model, benchmarked head-to-head, with the winner auto-selected for the live app.")

    with st.spinner("Reconstructing the held-out evaluation split (one-time)..."):
        split = A.reconstruct_split(data)

    reg_label = _model_label(data["reg_model"])
    cls_label = _model_label(data["cls_model"])

    tab1, tab2 = st.tabs(["📈 Regression — Rating Prediction", "🧭 Classification — Visit Mode"])

    # ------------------------------------------------------------------
    with tab1:
        reg_diag = A.deep_regression_diagnostics(data, split)
        reg_df = data["lookup"]["reg_results_df"].copy()
        if reg_label not in reg_df["Model"].values:
            new_row = pd.DataFrame([{"Model": f"{reg_label} (deployed)", "R2": reg_diag["r2"],
                                      "RMSE": reg_diag["rmse"], "MAE": reg_diag["mae"]}])
            reg_df = pd.concat([reg_df, new_row], ignore_index=True)
        reg_df = reg_df.sort_values("R2", ascending=False).reset_index(drop=True)

        st.dataframe(reg_df.style.format({"R2": "{:.3f}", "RMSE": "{:.3f}", "MAE": "{:.3f}"}),
                      width="stretch", hide_index=True)
        fig = px.bar(reg_df.melt(id_vars="Model", value_vars=["R2", "RMSE", "MAE"]),
                     x="Model", y="value", color="variable", barmode="group", title="Regression Model Comparison")
        fig.update_layout(height=420, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
        st.success(f"**Deployed regressor: {reg_label}** — R²={reg_diag['r2']:.3f}, "
                   f"RMSE={reg_diag['rmse']:.3f}, MAE={reg_diag['mae']:.3f} on the held-out split.")
        st.caption(
            "R² in the 0.10–0.15 range is expected for review-style rating data: ratings cluster tightly "
            "around 4–5, leaving limited variance for any model to explain. Attraction reputation "
            "(AttractionAvgRating) is consistently the dominant signal."
        )

        col1, col2 = st.columns(2)
        with col1:
            scatter_df = pd.DataFrame({"Actual": reg_diag["y_test"].values, "Predicted": reg_diag["preds"]})
            fig = px.scatter(scatter_df.sample(min(3000, len(scatter_df)), random_state=42),
                              x="Actual", y="Predicted", opacity=0.4, title="Actual vs. Predicted Rating")
            fig.add_shape(type="line", x0=1, y0=1, x1=5, y1=5, line=dict(color="#C97A3D", dash="dash"))
            fig.update_layout(height=380, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.histogram(reg_diag["residuals"], nbins=30, title="Residuals (Actual − Predicted)")
            fig.update_layout(height=380, margin=dict(t=50), showlegend=False)
            st.plotly_chart(fig, width="stretch")

        reg_model = data["reg_model"]
        reg_features = data["feature_lists"]["reg_features"]
        if hasattr(reg_model, "feature_importances_"):
            imp = pd.Series(reg_model.feature_importances_, index=reg_features).sort_values(ascending=False)
            fig = px.bar(imp, orientation="h", title=f"Feature Importance — {reg_label} (Rating)")
            fig.update_layout(height=340, yaxis=dict(categoryorder="total ascending"), showlegend=False, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------
    with tab2:
        cls_diag = A.deep_classification_diagnostics(data, split)
        cls_df = data["lookup"]["cls_results_df"].copy()
        report = cls_diag["report"]
        deployed_f1 = report["weighted avg"]["f1-score"]
        if cls_label not in cls_df["Model"].values:
            new_row = pd.DataFrame([{
                "Model": f"{cls_label} (deployed)",
                "Accuracy": report["accuracy"],
                "Precision": report["weighted avg"]["precision"],
                "Recall": report["weighted avg"]["recall"],
                "F1": deployed_f1,
            }])
            cls_df = pd.concat([cls_df, new_row], ignore_index=True)
        cls_df = cls_df.sort_values("F1", ascending=False).reset_index(drop=True)

        st.dataframe(cls_df.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}",
                                            "Recall": "{:.3f}", "F1": "{:.3f}"}),
                      width="stretch", hide_index=True)
        fig = px.bar(cls_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1"]),
                     x="Model", y="value", color="variable", barmode="group", title="Classification Model Comparison")
        fig.update_layout(height=420, margin=dict(t=50), yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
        st.success(f"**Deployed classifier: {cls_label}** — weighted F1={deployed_f1:.3f} on the held-out split.")
        st.caption(
            "The dataset is imbalanced — Couples and Family visits vastly outnumber Business trips — "
            "so weighted F1 is the fairer scorecard, and class-balancing during training helps the "
            "model pay attention to minority segments."
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(cls_diag["confusion_matrix"], x=cls_diag["labels"], y=cls_diag["labels"],
                             text_auto=True, color_continuous_scale=["#F7F4EE", "#C97A3D", "#163A36"],
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             title=f"Confusion Matrix — {cls_label}")
            fig.update_layout(height=420, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = go.Figure()
            for label, rd in cls_diag["roc_data"].items():
                fig.add_trace(go.Scatter(x=rd["fpr"], y=rd["tpr"], mode="lines",
                                          name=f"{label} (AUC={rd['auc']:.2f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                      showlegend=False))
            fig.update_layout(title="ROC Curves (One-vs-Rest)", xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate", height=420, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="section-label">Which classes are hardest to predict?</div>', unsafe_allow_html=True)
        per_class = pd.DataFrame({
            label: {"Precision": report[label]["precision"], "Recall": report[label]["recall"],
                    "F1": report[label]["f1-score"], "Support": report[label]["support"]}
            for label in cls_diag["labels"]
        }).T.reset_index().rename(columns={"index": "Visit Mode"})
        fig = px.bar(per_class, x="Visit Mode", y="F1", color="F1", color_continuous_scale=["#A85A2A", "#EFE8DA", "#163A36"],
                     title="Per-Class F1 Score")
        fig.update_layout(height=360, margin=dict(t=50), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(per_class.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                      width="stretch", hide_index=True)

        cls_model = data["cls_model"]
        cls_features = data["feature_lists"]["cls_features"]
        if hasattr(cls_model, "feature_importances_"):
            imp = pd.Series(cls_model.feature_importances_, index=cls_features).sort_values(ascending=False)
            fig = px.bar(imp, orientation="h", title=f"Feature Importance — {cls_label} (Visit Mode)")
            fig.update_layout(height=340, yaxis=dict(categoryorder="total ascending"), showlegend=False, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")

        st.caption("For a per-prediction explanation (SHAP), see the **🧭 Predict Visit Mode** page.")
        st.download_button("⬇️ Download model comparison (CSV)", A.to_csv_bytes(cls_df),
                            file_name="classification_model_comparison.csv", mime="text/csv")

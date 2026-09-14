import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import advanced as A
from utils import predict_visit_mode, predict_rating


def _model_label(model):
    name = type(model).__name__
    return {
        "LGBMClassifier": "LightGBM", "LGBMRegressor": "LightGBM",
        "XGBClassifier": "XGBoost", "XGBRegressor": "XGBoost",
        "RandomForestClassifier": "Random Forest", "RandomForestRegressor": "Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting", "LinearRegression": "Linear Regression",
    }.get(name, name)


def _attraction_avg_for(lookup, category, attraction_choice, cat_attractions, attraction_names):
    attr_avg_train = lookup["attr_avg_train"]
    if attraction_choice != "Not sure yet":
        aid = [a for a in cat_attractions if attraction_names[a] == attraction_choice][0]
        return aid, attr_avg_train.get(aid, lookup["global_avg_rating"])
    same_cat_ids = lookup["attraction_category"][lookup["attraction_category"] == category].index
    vals = [attr_avg_train.get(a) for a in same_cat_ids if a in attr_avg_train.index]
    return None, (float(np.mean(vals)) if vals else lookup["global_avg_rating"])


def _run_prediction(data, lookup, continent, region, country, category, year, month, attraction_choice, cat_attractions, attraction_names):
    aid, attr_avg = _attraction_avg_for(lookup, category, attraction_choice, cat_attractions, attraction_names)
    pred_label, proba = predict_visit_mode(
        data, continent, region, country, category, int(year), int(month),
        attr_avg, lookup["global_avg_rating"],
    )
    est_rating = predict_rating(data, continent, region, country, category, int(year), int(month), attr_avg)
    return pred_label, proba, est_rating, attr_avg, aid


def render(data):
    lookup = data["lookup"]
    st.markdown("## 🧭 Predict Visit Mode")
    st.caption(
        "Tell us about the trip — we'll predict whether this looks like a Business, "
        f"Couples, Family, Friends, or Solo visit, using the best-performing classifier "
        f"(**{_model_label(data['cls_model'])}**)."
    )

    with st.form("predict_mode_form"):
        c1, c2 = st.columns(2)
        with c1:
            continent = st.selectbox("Continent", lookup["continent_list"], index=2)
            region_opts = lookup["region_by_continent"].get(continent, [])
            region = st.selectbox("Region", region_opts if region_opts else ["—"])
            country_opts = lookup["country_by_region"].get(region, [])
            country = st.selectbox("Country", country_opts if country_opts else ["—"])
        with c2:
            year = st.number_input("Visit Year", min_value=2018, max_value=2027, value=2025)
            month = st.slider("Visit Month", 1, 12, 6)
            category = st.selectbox("Attraction Category of Interest", lookup["category_list"])

        attraction_names = lookup["attraction_names"]
        attraction_category = lookup["attraction_category"]
        cat_attractions = [aid for aid, c in attraction_category.items() if c == category]
        attraction_choice = st.selectbox(
            "Specific attraction (optional — improves accuracy)",
            ["Not sure yet"] + [attraction_names[a] for a in cat_attractions],
        )
        submitted = st.form_submit_button("Predict Visit Mode", width="stretch")

    if submitted:
        st.session_state["predict_inputs"] = dict(
            continent=continent, region=region, country=country, category=category,
            year=int(year), month=int(month), attraction_choice=attraction_choice,
        )

    if "predict_inputs" in st.session_state:
        pi = st.session_state["predict_inputs"]
        cat_attractions_pi = [aid for aid, c in attraction_category.items() if c == pi["category"]]
        pred_label, proba, est_rating, attr_avg, aid = _run_prediction(
            data, lookup, pi["continent"], pi["region"], pi["country"], pi["category"],
            pi["year"], pi["month"], pi["attraction_choice"], cat_attractions_pi, attraction_names,
        )

        st.markdown(f"""<div class="predict-result"><div class="kpi-label">PREDICTED VISIT MODE</div>
                     <div class="big">🧳 {pred_label}</div></div>""", unsafe_allow_html=True)

        tier_label, tier_text = A.confidence_tier(proba.iloc[0])
        rc1, rc2 = st.columns([1.3, 1])
        with rc1:
            fig = px.bar(x=proba.values, y=proba.index, orientation="h",
                         color=proba.values, color_continuous_scale=["#EFE8DA", "#C97A3D", "#163A36"],
                         labels={"x": "Probability", "y": "Visit Mode"},
                         title="Prediction Confidence Across All Visit Modes")
            fig.update_layout(height=320, margin=dict(t=50), coloraxis_showscale=False,
                               yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, width="stretch")
        with rc2:
            st.metric("Estimated Attraction Rating", f"{est_rating:.2f} / 5")
            st.metric("Model Confidence", f"{proba.iloc[0]*100:.1f}%")
            st.markdown(f"**{tier_label}**")
            st.caption(tier_text)

        with st.expander("🔍 Why did the model make this prediction? (SHAP)"):
            try:
                cls_features = data["feature_lists"]["cls_features"]
                enc = data["encoders"]["features"]
                row = pd.DataFrame([{
                    "VisitYear": pi["year"], "VisitMonth": pi["month"],
                    "Continent_enc": enc["Continent"].transform([pi["continent"]])[0] if pi["continent"] in enc["Continent"].classes_ else 0,
                    "Region_enc": enc["Region"].transform([pi["region"]])[0] if pi["region"] in enc["Region"].classes_ else 0,
                    "Country_enc": enc["Country"].transform([pi["country"]])[0] if pi["country"] in enc["Country"].classes_ else 0,
                    "AttractionCategory_enc": enc["AttractionCategory"].transform([pi["category"]])[0] if pi["category"] in enc["AttractionCategory"].classes_ else 0,
                    "UserAvgRating": lookup["global_avg_rating"], "AttractionAvgRating": attr_avg,
                }])[cls_features]
                explainer = A.get_shap_explainer(data["cls_model"])
                target_le = data["encoders"]["target"]
                pred_idx = list(target_le.classes_).index(pred_label)
                contrib = A.shap_explain_row(explainer, row, predicted_class_idx=pred_idx)
                fig2 = px.bar(contrib, orientation="h",
                              color=contrib.values, color_continuous_scale=["#A85A2A", "#EFE8DA", "#163A36"],
                              title=f"Feature Contributions Toward '{pred_label}'",
                              labels={"value": "SHAP contribution", "index": "Feature"})
                fig2.update_layout(height=340, yaxis=dict(categoryorder="total ascending"),
                                    showlegend=False, coloraxis_showscale=False, margin=dict(t=50))
                st.plotly_chart(fig2, width="stretch")
                st.caption("Positive bars push the model toward the predicted class; negative bars push away from it.")
            except Exception as e:
                st.info(f"Explainability unavailable for this input combination ({e}).")

    st.markdown("---")
    st.markdown("### 🎛️ What-If Scenario Simulator")
    st.caption("Change one thing at a time and watch the prediction update instantly.")
    wc1, wc2, wc3, wc4 = st.columns(4)
    with wc1:
        wi_continent = st.selectbox("Continent ", lookup["continent_list"], index=2, key="wi_continent")
    with wc2:
        wi_region_opts = lookup["region_by_continent"].get(wi_continent, [])
        wi_region = st.selectbox("Region ", wi_region_opts if wi_region_opts else ["—"], key="wi_region")
    with wc3:
        wi_category = st.selectbox("Category ", lookup["category_list"], key="wi_category")
    with wc4:
        wi_month = st.slider("Month ", 1, 12, 6, key="wi_month")

    wi_country_opts = lookup["country_by_region"].get(wi_region, [])
    wi_country = wi_country_opts[0] if wi_country_opts else "—"
    wi_cat_attractions = [aid for aid, c in attraction_category.items() if c == wi_category]
    wi_label, wi_proba, wi_rating, _, _ = _run_prediction(
        data, lookup, wi_continent, wi_region, wi_country, wi_category, 2025, wi_month,
        "Not sure yet", wi_cat_attractions, attraction_names,
    )
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Predicted Visit Mode", wi_label)
    sc2.metric("Confidence", f"{wi_proba.iloc[0]*100:.1f}%")
    sc3.metric("Estimated Rating", f"{wi_rating:.2f} / 5")

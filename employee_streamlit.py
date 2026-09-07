# ============================================================
# EMPLOYEE ATTRITION RISK DASHBOARD
# PALO ALTO NETWORKS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Risk Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

h1 {
    color: #17365D;
}

h2 {
    color: #17365D;
}

h3 {
    color: #17365D;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e0e6ed;
    text-align: center;
}

.risk-high {
    background-color: #ffe5e5;
    padding: 15px;
    border-radius: 10px;
    color: #b00020;
    font-weight: bold;
}

.risk-medium {
    background-color: #fff1d6;
    padding: 15px;
    border-radius: 10px;
    color: #9a6200;
    font-weight: bold;
}

.risk-low {
    background-color: #e3f7e8;
    padding: 15px;
    border-radius: 10px;
    color: #18733c;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Employee Attrition Risk Dashboard")

st.write(
    "Predict employee attrition risk using machine learning "
    "and identify the major factors influencing employee exit."
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(r"C:\Users\SAKSHI LATE\Downloads\Palo Alto Networks.csv")

    # Remove duplicate records
    data = data.drop_duplicates()

    # Convert Attrition
    if data["Attrition"].dtype == "object":

        data["Attrition"] = (
            data["Attrition"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "y": 1,
                "n": 0,
                "1": 1,
                "0": 0
            })
        )

    data = data.dropna(
        subset=["Attrition"]
    )

    data["Attrition"] = data["Attrition"].astype(int)

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(data):

    data = data.copy()

    # Income to experience ratio
    if (
        "MonthlyIncome" in data.columns
        and "TotalWorkingYears" in data.columns
    ):

        data["IncomeExperienceRatio"] = (
            data["MonthlyIncome"] /
            (data["TotalWorkingYears"] + 1)
        )


    # Promotion delay
    if "YearsSinceLastPromotion" in data.columns:

        data["PromotionDelay"] = (
            data["YearsSinceLastPromotion"] >= 3
        ).astype(int)


    # Promotion delay ratio
    if (
        "YearsSinceLastPromotion" in data.columns
        and "YearsAtCompany" in data.columns
    ):

        data["PromotionDelayRatio"] = (
            data["YearsSinceLastPromotion"] /
            (data["YearsAtCompany"] + 1)
        )


    # Engagement score
    engagement_columns = []

    for column in [
        "JobInvolvement",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance"
    ]:

        if column in data.columns:

            engagement_columns.append(column)


    if len(engagement_columns) > 0:

        data["EngagementScore"] = (
            data[engagement_columns].mean(axis=1)
        )


    # Workload stress flag
    stress_conditions = []

    if "OverTime" in data.columns:

        condition = (
            data["OverTime"]
            .astype(str)
            .str.lower()
            .eq("yes")
        )

        stress_conditions.append(
            condition.astype(int)
        )


    if "JobInvolvement" in data.columns:

        condition = (
            data["JobInvolvement"] <= 2
        )

        stress_conditions.append(
            condition.astype(int)
        )


    if "WorkLifeBalance" in data.columns:

        condition = (
            data["WorkLifeBalance"] <= 2
        )

        stress_conditions.append(
            condition.astype(int)
        )


    if len(stress_conditions) > 0:

        data["WorkloadStressFlag"] = (
            pd.concat(
                stress_conditions,
                axis=1
            ).sum(axis=1) >= 2
        ).astype(int)


    return data


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

try:

    df = load_data()

    df = feature_engineering(df)

except Exception as error:

    st.error(
        "Dataset could not be loaded. "
        "Make sure 'Palo Alto Networks.csv' "
        "is in the same folder as app.py."
    )

    st.error(error)

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

total_employees = len(df)

employees_left = int(
    df["Attrition"].sum()
)

employees_stayed = (
    total_employees -
    employees_left
)

actual_attrition_rate = (
    employees_left /
    total_employees
) * 100


# ============================================================
# MACHINE LEARNING DATA
# ============================================================

X = df.drop(
    columns=[
        "Attrition",
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ],
    errors="ignore"
)

y = df["Attrition"]


# Remove constant columns

constant_columns = []

for column in X.columns:

    if X[column].nunique() <= 1:

        constant_columns.append(
            column
        )


if len(constant_columns) > 0:

    X = X.drop(
        columns=constant_columns
    )


# ============================================================
# NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)


# ============================================================
# TRAIN GRADIENT BOOSTING
# ============================================================

gradient_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ]
)

gradient_model.fit(
    X_train,
    y_train
)


# ============================================================
# MODEL EVALUATION
# ============================================================

models = {

    "Logistic Regression":
        logistic_model,

    "Random Forest":
        random_forest_model,

    "Gradient Boosting":
        gradient_model
}


model_results = {}

model_probabilities = {}

for model_name, model in models.items():

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    model_results[model_name] = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "ROC-AUC": roc_auc
    }

    model_probabilities[
        model_name
    ] = probabilities


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = max(
    model_results,
    key=lambda x:
    model_results[x]["ROC-AUC"]
)

best_model = models[
    best_model_name
]


# ============================================================
# CREATE RISK SCORES
# ============================================================

all_probabilities = (
    best_model
    .predict_proba(X)[:, 1]
)


df["AttritionProbability"] = (
    all_probabilities
)


# ============================================================
# RISK CATEGORY FUNCTION
# ============================================================

def get_risk_category(probability):

    if probability < 0.30:

        return "Low Risk"

    elif probability < 0.60:

        return "Medium Risk"

    else:

        return "High Risk"


df["RiskCategory"] = (
    df["AttritionProbability"]
    .apply(get_risk_category)
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ Dashboard Controls")

st.sidebar.write(
    "Use the filters below to explore employee risk."
)


# ============================================================
# DEPARTMENT FILTER
# ============================================================

if "Department" in df.columns:

    department_options = [
        "All"
    ] + sorted(
        df["Department"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_department = st.sidebar.selectbox(
        "Department",
        department_options
    )

else:

    selected_department = "All"


# ============================================================
# JOB ROLE FILTER
# ============================================================

if "JobRole" in df.columns:

    role_options = [
        "All"
    ] + sorted(
        df["JobRole"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_role = st.sidebar.selectbox(
        "Job Role",
        role_options
    )

else:

    selected_role = "All"


# ============================================================
# RISK THRESHOLD SLIDER
# ============================================================

risk_threshold = st.sidebar.slider(
    "High Risk Threshold (%)",
    min_value=30,
    max_value=90,
    value=60,
    step=5
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_department != "All":

    filtered_df = filtered_df[
        filtered_df["Department"]
        == selected_department
    ]


if selected_role != "All":

    filtered_df = filtered_df[
        filtered_df["JobRole"]
        == selected_role
    ]


# ============================================================
# HIGH RISK BASED ON SLIDER
# ============================================================

high_risk_filtered = filtered_df[
    filtered_df["AttritionProbability"]
    >= risk_threshold / 100
]


# ============================================================
# DASHBOARD KPI CARDS
# ============================================================

st.subheader("📌 Attrition Risk Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Employees",
        len(filtered_df)
    )


with col2:

    st.metric(
        "Employees Left",
        int(
            filtered_df["Attrition"].sum()
        )
    )


with col3:

    st.metric(
        "High Risk Employees",
        len(high_risk_filtered)
    )


with col4:

    average_risk = (
        filtered_df[
            "AttritionProbability"
        ].mean() * 100
    )

    st.metric(
        "Average Risk",
        f"{average_risk:.1f}%"
    )


st.divider()


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader("📊 Overall Risk Distribution")


risk_counts = (
    filtered_df["RiskCategory"]
    .value_counts()
    .reindex(
        [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        fill_value=0
    )
)


col1, col2 = st.columns(2)


with col1:

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        risk_counts.index,
        risk_counts.values,
        color=[
            "seagreen",
            "orange",
            "tomato"
        ]
    )

    ax.set_title(
        "Employee Risk Distribution"
    )

    ax.set_xlabel(
        "Risk Category"
    )

    ax.set_ylabel(
        "Number of Employees"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    st.pyplot(fig)


with col2:

    st.write("### Risk Summary")

    st.success(
        f"🟢 Low Risk: "
        f"{risk_counts['Low Risk']}"
    )

    st.warning(
        f"🟡 Medium Risk: "
        f"{risk_counts['Medium Risk']}"
    )

    st.error(
        f"🔴 High Risk: "
        f"{risk_counts['High Risk']}"
    )

    st.write(
        f"High-risk threshold selected: "
        f"**{risk_threshold}%**"
    )


st.divider()


# ============================================================
# HIGH RISK EMPLOYEES
# ============================================================

st.subheader("🔴 High-Risk Employees")


high_risk_columns = [
    column
    for column in [
        "EmployeeNumber",
        "Age",
        "Department",
        "JobRole",
        "MonthlyIncome",
        "OverTime",
        "JobSatisfaction",
        "YearsAtCompany",
        "AttritionProbability",
        "RiskCategory"
    ]
    if column in high_risk_filtered.columns
]


high_risk_display = (
    high_risk_filtered[
        high_risk_columns
    ]
    .sort_values(
        "AttritionProbability",
        ascending=False
    )
    .copy()
)


if "AttritionProbability" in high_risk_display.columns:

    high_risk_display[
        "AttritionProbability"
    ] = (
        high_risk_display[
            "AttritionProbability"
        ] * 100
    ).round(2)


st.dataframe(
    high_risk_display.head(20),
    use_container_width=True
)


st.divider()


# ============================================================
# DEPARTMENT LEVEL RISK
# ============================================================

st.subheader("🏢 Department-Level Risk")


if "Department" in filtered_df.columns:

    department_risk = (
        filtered_df
        .groupby("Department")
        ["AttritionProbability"]
        .mean()
        .sort_values(
            ascending=False
        ) * 100
    )

    col1, col2 = st.columns(2)


    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            department_risk.index,
            department_risk.values,
            color="steelblue"
        )

        ax.set_title(
            "Average Attrition Risk by Department"
        )

        ax.set_xlabel(
            "Department"
        )

        ax.set_ylabel(
            "Average Risk (%)"
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )

        st.pyplot(fig)


    with col2:

        department_table = pd.DataFrame({

            "Department":
                department_risk.index,

            "Average Risk (%)":
                department_risk.values.round(2)
        })

        st.dataframe(
            department_table,
            use_container_width=True
        )


# ============================================================
# JOB ROLE LEVEL RISK
# ============================================================

st.subheader("👥 Job Role-Level Risk")


if "JobRole" in filtered_df.columns:

    role_risk = (
        filtered_df
        .groupby("JobRole")
        ["AttritionProbability"]
        .mean()
        .sort_values()
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        role_risk.index,
        role_risk.values,
        color="mediumpurple"
    )

    ax.set_title(
        "Average Attrition Risk by Job Role"
    )

    ax.set_xlabel(
        "Average Risk (%)"
    )

    ax.set_ylabel(
        "Job Role"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    st.pyplot(fig)


st.divider()


# ============================================================
# EMPLOYEE RISK PROFILE
# ============================================================

st.subheader("👤 Individual Employee Risk Profile")


# Create employee identifiers

if "EmployeeNumber" in filtered_df.columns:

    employee_ids = (
        filtered_df[
            "EmployeeNumber"
        ]
        .dropna()
        .tolist()
    )

    selected_employee = st.selectbox(
        "Select Employee ID",
        employee_ids
    )

    employee_row = filtered_df[
        filtered_df["EmployeeNumber"]
        == selected_employee
    ].iloc[0]

else:

    employee_indices = (
        filtered_df.index.tolist()
    )

    selected_employee = st.selectbox(
        "Select Employee",
        employee_indices
    )

    employee_row = filtered_df.loc[
        selected_employee
    ]


# ============================================================
# EMPLOYEE PROBABILITY
# ============================================================

employee_probability = (
    employee_row[
        "AttritionProbability"
    ]
)


employee_risk = get_risk_category(
    employee_probability
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Employee ID",
        str(
            employee_row.get(
                "EmployeeNumber",
                selected_employee
            )
        )
    )


with col2:

    st.metric(
        "Attrition Probability",
        f"{employee_probability * 100:.2f}%"
    )


with col3:

    st.metric(
        "Risk Category",
        employee_risk
    )


# Risk message

if employee_risk == "High Risk":

    st.error(
        "🔴 HIGH RISK — Employee may have a higher "
        "probability of attrition."
    )

elif employee_risk == "Medium Risk":

    st.warning(
        "🟡 MEDIUM RISK — Employee requires monitoring."
    )

else:

    st.success(
        "🟢 LOW RISK — Employee currently has lower "
        "predicted attrition risk."
    )


# ============================================================
# EMPLOYEE DETAILS
# ============================================================

st.write("### Employee Details")


employee_detail_columns = [
    "Age",
    "Department",
    "JobRole",
    "MonthlyIncome",
    "OverTime",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    "TotalWorkingYears",
    "JobInvolvement"
]


available_details = [
    column
    for column in employee_detail_columns
    if column in filtered_df.columns
]


employee_details = pd.DataFrame(
    {
        "Feature":
            available_details,

        "Value":
            [
                employee_row[column]
                for column
                in available_details
            ]
    }
)


st.dataframe(
    employee_details,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# KEY CONTRIBUTING FACTORS
# ============================================================

st.subheader("🔎 Key Contributing Factors")


reasons = []


if "OverTime" in filtered_df.columns:

    if str(
        employee_row["OverTime"]
    ).lower() == "yes":

        reasons.append(
            "Overtime workload may increase attrition risk."
        )


if "JobSatisfaction" in filtered_df.columns:

    if employee_row[
        "JobSatisfaction"
    ] <= 2:

        reasons.append(
            "Low job satisfaction."
        )


if "EnvironmentSatisfaction" in filtered_df.columns:

    if employee_row[
        "EnvironmentSatisfaction"
    ] <= 2:

        reasons.append(
            "Low satisfaction with work environment."
        )


if "WorkLifeBalance" in filtered_df.columns:

    if employee_row[
        "WorkLifeBalance"
    ] <= 2:

        reasons.append(
            "Poor work-life balance."
        )


if "JobInvolvement" in filtered_df.columns:

    if employee_row[
        "JobInvolvement"
    ] <= 2:

        reasons.append(
            "Low job involvement."
        )


if "YearsSinceLastPromotion" in filtered_df.columns:

    if employee_row[
        "YearsSinceLastPromotion"
    ] >= 3:

        reasons.append(
            "Long delay since last promotion."
        )


if "DistanceFromHome" in filtered_df.columns:

    if employee_row[
        "DistanceFromHome"
    ] >= 15:

        reasons.append(
            "Long distance from home."
        )


if len(reasons) == 0:

    st.success(
        "No major predefined risk factor detected."
    )

else:

    for reason in reasons:

        st.write(
            "•",
            reason
        )


st.divider()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("📈 Feature Importance")

st.write(
    f"Feature importance from the selected best model: "
    f"**{best_model_name}**"
)


fitted_preprocessor = (
    best_model
    .named_steps[
        "preprocessor"
    ]
)


fitted_model = (
    best_model
    .named_steps[
        "model"
    ]
)


feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


if hasattr(
    fitted_model,
    "feature_importances_"
):

    importance_values = (
        fitted_model
        .feature_importances_
    )

else:

    importance_values = np.abs(
        fitted_model.coef_[0]
    )


importance_df = pd.DataFrame({

    "Feature":
        feature_names,

    "Importance":
        importance_values

})


importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)
)


fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1],
    color="slateblue"
)

ax.set_title(
    "Top Factors Influencing Attrition Prediction"
)

ax.set_xlabel(
    "Importance"
)

ax.set_ylabel(
    "Feature"
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

st.pyplot(fig)


st.dataframe(
    importance_df,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# WHAT-IF SCENARIO ANALYSIS
# ============================================================

st.subheader("🔮 What-If Scenario Analysis")

st.write(
    "Change employee conditions and observe how the "
    "predicted attrition probability changes."
)


# Start with selected employee

scenario_employee = (
    employee_row[
        X.columns
    ].copy()
)


# ------------------------------------------------------------
# OVERTIME
# ------------------------------------------------------------

if "OverTime" in scenario_employee.index:

    current_overtime = str(
        scenario_employee["OverTime"]
    )

    overtime_options = [
        "Yes",
        "No"
    ]

    selected_overtime = st.selectbox(
        "Overtime",
        overtime_options,
        index=(
            overtime_options.index(
                current_overtime
            )
            if current_overtime
            in overtime_options
            else 0
        )
    )

    scenario_employee[
        "OverTime"
    ] = selected_overtime


# ------------------------------------------------------------
# JOB SATISFACTION
# ------------------------------------------------------------

if "JobSatisfaction" in scenario_employee.index:

    current_satisfaction = int(
        scenario_employee[
            "JobSatisfaction"
        ]
    )

    new_satisfaction = st.slider(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=current_satisfaction
    )

    scenario_employee[
        "JobSatisfaction"
    ] = new_satisfaction


# ------------------------------------------------------------
# WORK LIFE BALANCE
# ------------------------------------------------------------

if "WorkLifeBalance" in scenario_employee.index:

    current_balance = int(
        scenario_employee[
            "WorkLifeBalance"
        ]
    )

    new_balance = st.slider(
        "Work-Life Balance",
        min_value=1,
        max_value=4,
        value=current_balance
    )

    scenario_employee[
        "WorkLifeBalance"
    ] = new_balance


# ------------------------------------------------------------
# MONTHLY INCOME
# ------------------------------------------------------------

if "MonthlyIncome" in scenario_employee.index:

    current_income = int(
        scenario_employee[
            "MonthlyIncome"
        ]
    )

    new_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=current_income,
        step=1000
    )

    scenario_employee[
        "MonthlyIncome"
    ] = new_income


# ------------------------------------------------------------
# YEARS SINCE LAST PROMOTION
# ------------------------------------------------------------

if (
    "YearsSinceLastPromotion"
    in scenario_employee.index
):

    current_promotion = int(
        scenario_employee[
            "YearsSinceLastPromotion"
        ]
    )

    new_promotion = st.slider(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=current_promotion
    )

    scenario_employee[
        "YearsSinceLastPromotion"
    ] = new_promotion


# ============================================================
# UPDATE ENGINEERED FEATURES FOR WHAT-IF
# ============================================================

if (
    "MonthlyIncome" in scenario_employee.index
    and "TotalWorkingYears" in scenario_employee.index
):

    scenario_employee[
        "IncomeExperienceRatio"
    ] = (
        scenario_employee["MonthlyIncome"] /
        (
            scenario_employee[
                "TotalWorkingYears"
            ] + 1
        )
    )


if (
    "YearsSinceLastPromotion"
    in scenario_employee.index
):

    scenario_employee[
        "PromotionDelay"
    ] = int(
        scenario_employee[
            "YearsSinceLastPromotion"
        ] >= 3
    )


if (
    "YearsSinceLastPromotion"
    in scenario_employee.index
    and "YearsAtCompany"
    in scenario_employee.index
):

    scenario_employee[
        "PromotionDelayRatio"
    ] = (
        scenario_employee[
            "YearsSinceLastPromotion"
        ] /
        (
            scenario_employee[
                "YearsAtCompany"
            ] + 1
        )
    )


# Engagement score

scenario_engagement_columns = []

for column in [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]:

    if column in scenario_employee.index:

        scenario_engagement_columns.append(
            column
        )


if len(
    scenario_engagement_columns
) > 0:

    scenario_employee[
        "EngagementScore"
    ] = np.mean(
        [
            scenario_employee[column]
            for column
            in scenario_engagement_columns
        ]
    )


# Workload stress

stress_score = 0


if "OverTime" in scenario_employee.index:

    if str(
        scenario_employee["OverTime"]
    ).lower() == "yes":

        stress_score += 1


if "JobInvolvement" in scenario_employee.index:

    if (
        scenario_employee[
            "JobInvolvement"
        ] <= 2
    ):

        stress_score += 1


if "WorkLifeBalance" in scenario_employee.index:

    if (
        scenario_employee[
            "WorkLifeBalance"
        ] <= 2
    ):

        stress_score += 1


if "WorkloadStressFlag" in X.columns:

    scenario_employee[
        "WorkloadStressFlag"
    ] = int(
        stress_score >= 2
    )


# Keep only model columns

scenario_employee = (
    scenario_employee[
        X.columns
    ]
)


# ============================================================
# WHAT-IF PREDICTION
# ============================================================

scenario_probability = (
    best_model
    .predict_proba(
        pd.DataFrame(
            [scenario_employee]
        )
    )[0][1]
)


original_probability = (
    employee_probability
)


risk_difference = (
    scenario_probability -
    original_probability
) * 100


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Original Risk",
        f"{original_probability * 100:.2f}%"
    )


with col2:

    st.metric(
        "What-If Risk",
        f"{scenario_probability * 100:.2f}%"
    )


with col3:

    st.metric(
        "Risk Change",
        f"{risk_difference:+.2f}%"
    )


if scenario_probability < original_probability:

    st.success(
        "✅ The modified scenario reduces predicted attrition risk."
    )

elif scenario_probability > original_probability:

    st.error(
        "⚠️ The modified scenario increases predicted attrition risk."
    )

else:

    st.info(
        "The modified scenario does not change predicted risk."
    )


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🤖 Model Performance")


performance_df = pd.DataFrame(
    model_results
).T


performance_df = (
    performance_df
    .round(4)
)


st.dataframe(
    performance_df,
    use_container_width=True
)


# ============================================================
# MODEL PERFORMANCE CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

performance_df.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Machine Learning Model Comparison"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# PROJECT METHODOLOGY
# ============================================================

st.divider()

st.subheader("📚 Data Science Methodology")

st.write("""
**1. Data Preprocessing**
- Missing value handling
- Categorical variable encoding
- Numerical feature scaling
- Class imbalance handling
- Stratified train-test split

**2. Feature Engineering**
- Income-to-experience ratio
- Promotion delay indicator
- Engagement composite score
- Workload stress flag

**3. Model Development**
- Logistic Regression
- Random Forest
- Gradient Boosting

**4. Model Evaluation**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**5. Risk Scoring**
- Low Risk: below 30%
- Medium Risk: 30% to 60%
- High Risk: above 60%

**6. Explainability**
- Feature importance
- Individual contributing factors
- What-if scenario analysis
""")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Best Model: {best_model_name} | "
    f"ROC-AUC: "
    f"{model_results[best_model_name]['ROC-AUC']:.3f}"
)

st.caption(
    "Employee Attrition Risk Analysis | "
    "Data Science Project"
)
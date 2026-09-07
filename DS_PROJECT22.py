# ============================================================
# PALO ALTO NETWORKS
# EMPLOYEE ATTRITION RISK ANALYSIS & PREDICTION
#
# Technologies:
# Pandas, NumPy, Matplotlib, Scikit-learn
#
# PROJECT FLOW:
# 1. Load Dataset
# 2. Data Understanding
# 3. Data Cleaning
# 4. Exploratory Data Analysis
# 5. Feature Engineering
# 6. Train-Test Split with Stratification
# 7. Logistic Regression - Baseline Model
# 8. Random Forest
# 9. Gradient Boosting
# 10. Model Evaluation
# 11. Model Comparison
# 12. Risk Probability
# 13. Risk Categories
# 14. Feature Importance
# 15. Individual Employee Risk Profile
# 16. Key Contributing Factors
# 17. What-if Scenario Analysis
# 18. Save Predictions
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

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
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(r"C:\Users\SAKSHI LATE\Downloads\Palo Alto Networks.csv")

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 3. BASIC DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

print("\nMissing Value Percentage:")
print((missing / len(df) * 100).round(2))


# ============================================================
# 5. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)


# ============================================================
# 6. CONVERT ATTRITION INTO 0 AND 1
# ============================================================

# If Attrition contains Yes/No
if df["Attrition"].dtype == "object":

    df["Attrition"] = (
        df["Attrition"]
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

# Remove records where Attrition could not be converted
df = df.dropna(subset=["Attrition"])

df["Attrition"] = df["Attrition"].astype(int)

print("\nAttrition Encoding:")
print("0 = Stayed")
print("1 = Left")

print("\nAttrition Count:")
print(df["Attrition"].value_counts())


# ============================================================
# 7. INITIAL ATTRITION ANALYSIS
# ============================================================

total_employees = len(df)

employees_left = df["Attrition"].sum()

employees_stayed = total_employees - employees_left

attrition_rate = (
    employees_left / total_employees
) * 100

print("\n" + "=" * 70)
print("INITIAL ATTRITION ANALYSIS")
print("=" * 70)

print("Total Employees:", total_employees)
print("Employees Stayed:", employees_stayed)
print("Employees Left:", employees_left)
print("Attrition Rate:", round(attrition_rate, 2), "%")


# ============================================================
# 8. ATTRITION VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

plt.bar(
    ["Stayed", "Left"],
    [employees_stayed, employees_left],
    color=["seagreen", "tomato"]
)

plt.title("Overall Employee Attrition")
plt.xlabel("Employee Status")
plt.ylabel("Number of Employees")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.text(
    0,
    employees_stayed,
    str(employees_stayed),
    ha="center",
    va="bottom"
)

plt.text(
    1,
    employees_left,
    str(employees_left),
    ha="center",
    va="bottom"
)

plt.tight_layout()
plt.show()


# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# 9.1 INCOME TO EXPERIENCE RATIO
# ------------------------------------------------------------

if "MonthlyIncome" in df.columns and "TotalWorkingYears" in df.columns:

    df["IncomeExperienceRatio"] = (
        df["MonthlyIncome"] /
        (df["TotalWorkingYears"] + 1)
    )

    print("Created: IncomeExperienceRatio")


# ------------------------------------------------------------
# 9.2 PROMOTION DELAY INDICATOR
# ------------------------------------------------------------

if "YearsSinceLastPromotion" in df.columns:

    df["PromotionDelay"] = (
        df["YearsSinceLastPromotion"] >= 3
    ).astype(int)

    print("Created: PromotionDelay")


# ------------------------------------------------------------
# 9.3 ROLE PROMOTION DELAY
# ------------------------------------------------------------

if (
    "YearsSinceLastPromotion" in df.columns
    and "YearsAtCompany" in df.columns
):

    df["PromotionDelayRatio"] = (
        df["YearsSinceLastPromotion"] /
        (df["YearsAtCompany"] + 1)
    )

    print("Created: PromotionDelayRatio")


# ------------------------------------------------------------
# 9.4 ENGAGEMENT COMPOSITE SCORE
# ------------------------------------------------------------

engagement_columns = []

for column in [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]:

    if column in df.columns:
        engagement_columns.append(column)

if len(engagement_columns) > 0:

    df["EngagementScore"] = (
        df[engagement_columns].mean(axis=1)
    )

    print("Created: EngagementScore")


# ------------------------------------------------------------
# 9.5 WORKLOAD STRESS FLAG
# ------------------------------------------------------------

stress_conditions = []

if "OverTime" in df.columns:

    overtime_condition = (
        df["OverTime"]
        .astype(str)
        .str.lower()
        .eq("yes")
    )

    stress_conditions.append(
        overtime_condition.astype(int)
    )


if "JobInvolvement" in df.columns:

    involvement_condition = (
        df["JobInvolvement"] <= 2
    )

    stress_conditions.append(
        involvement_condition.astype(int)
    )


if "WorkLifeBalance" in df.columns:

    worklife_condition = (
        df["WorkLifeBalance"] <= 2
    )

    stress_conditions.append(
        worklife_condition.astype(int)
    )


if len(stress_conditions) > 0:

    df["WorkloadStressFlag"] = (
        pd.concat(
            stress_conditions,
            axis=1
        ).sum(axis=1) >= 2
    ).astype(int)

    print("Created: WorkloadStressFlag")


# ============================================================
# 10. DISPLAY ENGINEERED FEATURES
# ============================================================

print("\nEngineered Features:")

engineered_features = [
    "IncomeExperienceRatio",
    "PromotionDelay",
    "PromotionDelayRatio",
    "EngagementScore",
    "WorkloadStressFlag"
]

for feature in engineered_features:

    if feature in df.columns:
        print(
            feature,
            "->",
            df[feature].describe()[["mean", "min", "max"]].to_dict()
        )


# ============================================================
# 11. EXPLORATORY ANALYSIS
# ============================================================


# ------------------------------------------------------------
# AGE DISTRIBUTION
# ------------------------------------------------------------

if "Age" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["Age"],
        bins=15,
        edgecolor="black",
        color="steelblue"
    )

    plt.title("Employee Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Employees")

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# OVERTIME VS ATTRITION
# ------------------------------------------------------------

if "OverTime" in df.columns:

    overtime_data = pd.crosstab(
        df["OverTime"],
        df["Attrition"]
    )

    plt.figure(figsize=(8, 5))

    x = np.arange(len(overtime_data.index))

    stayed_values = overtime_data.get(
        0,
        pd.Series(
            0,
            index=overtime_data.index
        )
    )

    left_values = overtime_data.get(
        1,
        pd.Series(
            0,
            index=overtime_data.index
        )
    )

    width = 0.35

    plt.bar(
        x - width / 2,
        stayed_values,
        width,
        label="Stayed",
        color="seagreen"
    )

    plt.bar(
        x + width / 2,
        left_values,
        width,
        label="Left",
        color="tomato"
    )

    plt.xticks(
        x,
        overtime_data.index
    )

    plt.title("Overtime vs Employee Attrition")
    plt.xlabel("Overtime")
    plt.ylabel("Number of Employees")

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# DEPARTMENT VS ATTRITION
# ------------------------------------------------------------

if "Department" in df.columns:

    department_data = pd.crosstab(
        df["Department"],
        df["Attrition"]
    )

    plt.figure(figsize=(9, 5))

    x = np.arange(
        len(department_data.index)
    )

    stayed_values = department_data.get(
        0,
        pd.Series(
            0,
            index=department_data.index
        )
    )

    left_values = department_data.get(
        1,
        pd.Series(
            0,
            index=department_data.index
        )
    )

    width = 0.35

    plt.bar(
        x - width / 2,
        stayed_values,
        width,
        label="Stayed",
        color="steelblue"
    )

    plt.bar(
        x + width / 2,
        left_values,
        width,
        label="Left",
        color="orange"
    )

    plt.xticks(
        x,
        department_data.index
    )

    plt.title("Department vs Employee Attrition")
    plt.xlabel("Department")
    plt.ylabel("Number of Employees")

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 12. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

print("\n" + "=" * 70)
print("PREPARING DATA FOR MACHINE LEARNING")
print("=" * 70)


# Remove columns that are not useful for prediction

columns_to_remove = [
    "Attrition",
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

columns_to_remove = [
    column
    for column in columns_to_remove
    if column in df.columns
]


X = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

y = df["Attrition"]


# ------------------------------------------------------------
# Remove constant columns
# ------------------------------------------------------------

constant_columns = []

for column in X.columns:

    if X[column].nunique() <= 1:
        constant_columns.append(column)

if len(constant_columns) > 0:

    X = X.drop(
        columns=constant_columns
    )

    print(
        "Removed constant columns:",
        constant_columns
    )


# ============================================================
# 13. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 14. PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
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
# 15. TRAIN-TEST SPLIT WITH STRATIFICATION
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

print(
    "\nTraining Attrition Distribution:"
)

print(
    y_train.value_counts(
        normalize=True
    )
)


# ============================================================
# 16. MODEL 1 - LOGISTIC REGRESSION
# BASELINE INTERPRETABLE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION MODEL")
print("=" * 70)

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

logistic_pred = logistic_model.predict(
    X_test
)

logistic_probability = (
    logistic_model.predict_proba(X_test)[:, 1]
)


# ============================================================
# 17. MODEL 2 - RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST MODEL")
print("=" * 70)

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

rf_pred = random_forest_model.predict(
    X_test
)

rf_probability = (
    random_forest_model.predict_proba(X_test)[:, 1]
)


# ============================================================
# 18. MODEL 3 - GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING MODEL")
print("=" * 70)

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

gradient_pred = gradient_model.predict(
    X_test
)

gradient_probability = (
    gradient_model.predict_proba(X_test)[:, 1]
)


# ============================================================
# 19. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    actual,
    predicted,
    probability
):

    accuracy = accuracy_score(
        actual,
        predicted
    )

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        actual,
        probability
    )

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(
        "Accuracy :", round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1-Score :",
        round(f1, 4)
    )

    print(
        "ROC-AUC  :",
        round(roc_auc, 4)
    )

    return [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]


# ============================================================
# 20. EVALUATE ALL MODELS
# ============================================================

logistic_scores = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_probability
)

rf_scores = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_probability
)

gradient_scores = evaluate_model(
    "Gradient Boosting",
    y_test,
    gradient_pred,
    gradient_probability
)


# ============================================================
# 21. MODEL COMPARISON TABLE
# ============================================================

results = pd.DataFrame(
    [
        logistic_scores,
        rf_scores,
        gradient_scores
    ],
    columns=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    index=[
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ]
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results.round(4)
)


# ============================================================
# 22. MODEL COMPARISON VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

results.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score")

plt.xticks(rotation=0)

plt.ylim(0, 1)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.legend(
    title="Evaluation Metrics"
)

plt.tight_layout()
plt.show()


# ============================================================
# 23. SELECT BEST MODEL USING ROC-AUC
# ============================================================

model_probabilities = {
    "Logistic Regression": logistic_probability,
    "Random Forest": rf_probability,
    "Gradient Boosting": gradient_probability
}

model_scores = {
    "Logistic Regression": logistic_scores[4],
    "Random Forest": rf_scores[4],
    "Gradient Boosting": gradient_scores[4]
}

best_model_name = max(
    model_scores,
    key=model_scores.get
)

best_probability = model_probabilities[
    best_model_name
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best Model:",
    best_model_name
)

print(
    "Best ROC-AUC:",
    round(
        model_scores[best_model_name],
        4
    )
)


# ============================================================
# 24. ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for name, probability in model_probabilities.items():

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probability
    )

    auc_value = roc_auc_score(
        y_test,
        probability
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc_value:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.grid(
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ============================================================
# 25. CONFUSION MATRIX FOR BEST MODEL
# ============================================================

if best_model_name == "Logistic Regression":

    best_model = logistic_model
    best_pred = logistic_pred

elif best_model_name == "Random Forest":

    best_model = random_forest_model
    best_pred = rf_pred

else:

    best_model = gradient_model
    best_pred = gradient_pred


cm = confusion_matrix(
    y_test,
    best_pred
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Stayed", "Left"]
)

plt.yticks(
    [0, 1],
    ["Stayed", "Left"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.colorbar()

plt.tight_layout()
plt.show()


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        best_pred,
        target_names=[
            "Stayed",
            "Left"
        ]
    )
)


# ============================================================
# 27. CREATE RISK SCORES FOR ALL EMPLOYEES
# ============================================================

all_probabilities = (
    best_model.predict_proba(X)[:, 1]
)

df["AttritionProbability"] = (
    all_probabilities
)


# ============================================================
# 28. CREATE RISK CATEGORIES
# ============================================================

def risk_category(probability):

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


df["RiskCategory"] = (
    df["AttritionProbability"]
    .apply(risk_category)
)


# ============================================================
# 29. RISK DISTRIBUTION
# ============================================================

risk_counts = (
    df["RiskCategory"]
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

print("\n" + "=" * 70)
print("RISK DISTRIBUTION")
print("=" * 70)

print(risk_counts)

plt.figure(figsize=(8, 5))

plt.bar(
    risk_counts.index,
    risk_counts.values,
    color=[
        "seagreen",
        "orange",
        "tomato"
    ]
)

plt.title("Employee Attrition Risk Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Employees")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

for i, value in enumerate(
    risk_counts.values
):

    plt.text(
        i,
        value,
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 30. HIGH-RISK EMPLOYEE COUNT
# ============================================================

high_risk_count = (
    df["RiskCategory"]
    .eq("High Risk")
    .sum()
)

medium_risk_count = (
    df["RiskCategory"]
    .eq("Medium Risk")
    .sum()
)

low_risk_count = (
    df["RiskCategory"]
    .eq("Low Risk")
    .sum()
)

print("\nHigh-Risk Employees:", high_risk_count)
print("Medium-Risk Employees:", medium_risk_count)
print("Low-Risk Employees:", low_risk_count)


# ============================================================
# 31. HIGH-RISK EMPLOYEE TABLE
# ============================================================

high_risk_employees = (
    df[
        df["RiskCategory"] == "High Risk"
    ]
    .sort_values(
        "AttritionProbability",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("TOP HIGH-RISK EMPLOYEES")
print("=" * 70)

columns_to_show = [
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
    if column in high_risk_employees.columns
]

print(
    high_risk_employees[
        columns_to_show
    ].head(20)
)


# ============================================================
# 32. DEPARTMENT-LEVEL RISK
# ============================================================

if "Department" in df.columns:

    department_risk = (
        df.groupby("Department")
        ["AttritionProbability"]
        .mean()
        .sort_values(
            ascending=False
        ) * 100
    )

    print("\n" + "=" * 70)
    print("DEPARTMENT LEVEL RISK")
    print("=" * 70)

    print(
        department_risk.round(2)
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        department_risk.index,
        department_risk.values,
        color="steelblue"
    )

    plt.title(
        "Average Attrition Risk by Department"
    )

    plt.xlabel("Department")
    plt.ylabel("Average Risk (%)")

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 33. JOB ROLE LEVEL RISK
# ============================================================

if "JobRole" in df.columns:

    role_risk = (
        df.groupby("JobRole")
        ["AttritionProbability"]
        .mean()
        .sort_values(
            ascending=True
        ) * 100
    )

    print("\n" + "=" * 70)
    print("JOB ROLE LEVEL RISK")
    print("=" * 70)

    print(
        role_risk.round(2)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        role_risk.index,
        role_risk.values,
        color="mediumpurple"
    )

    plt.title(
        "Average Attrition Risk by Job Role"
    )

    plt.xlabel("Average Risk (%)")
    plt.ylabel("Job Role")

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 34. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


# Get transformed feature names

preprocessor_fitted = (
    best_model.named_steps["preprocessor"]
)

model_fitted = (
    best_model.named_steps["model"]
)

feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)


# ------------------------------------------------------------
# Random Forest / Gradient Boosting
# ------------------------------------------------------------

if hasattr(
    model_fitted,
    "feature_importances_"
):

    importance_values = (
        model_fitted
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(15)
    )


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

else:

    coefficient_values = (
        np.abs(
            model_fitted.coef_[0]
        )
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": coefficient_values
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(15)
    )


print(
    importance_df
)


# ============================================================
# 35. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1],
    color="slateblue"
)

plt.title(
    f"Top Feature Importance - {best_model_name}"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ============================================================
# 36. INDIVIDUAL EMPLOYEE RISK PROFILE
# ============================================================

print("\n" + "=" * 70)
print("INDIVIDUAL EMPLOYEE RISK PROFILE")
print("=" * 70)


# Select first employee as example

employee_index = df.index[0]

employee_data = df.loc[
    employee_index
]

employee_probability = (
    employee_data["AttritionProbability"]
)

employee_risk = (
    employee_data["RiskCategory"]
)

print(
    "\nEmployee Index:",
    employee_index
)

if "EmployeeNumber" in df.columns:

    print(
        "Employee ID:",
        employee_data["EmployeeNumber"]
    )

print(
    "Attrition Probability:",
    round(
        employee_probability * 100,
        2
    ),
    "%"
)

print(
    "Risk Category:",
    employee_risk
)


# ============================================================
# 37. INDIVIDUAL EMPLOYEE DETAILS
# ============================================================

important_employee_columns = [
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
    "TotalWorkingYears"
]

available_employee_columns = [
    column
    for column in important_employee_columns
    if column in df.columns
]

print("\nEmployee Details:")

print(
    employee_data[
        available_employee_columns
    ]
)


# ============================================================
# 38. SIMPLE INDIVIDUAL REASON CODES
# ============================================================

print("\n" + "=" * 70)
print("KEY CONTRIBUTING FACTORS FOR EMPLOYEE")
print("=" * 70)

reasons = []


if "OverTime" in df.columns:

    if str(
        employee_data["OverTime"]
    ).lower() == "yes":

        reasons.append(
            "High workload due to overtime"
        )


if "JobSatisfaction" in df.columns:

    if employee_data["JobSatisfaction"] <= 2:

        reasons.append(
            "Low job satisfaction"
        )


if "WorkLifeBalance" in df.columns:

    if employee_data["WorkLifeBalance"] <= 2:

        reasons.append(
            "Poor work-life balance"
        )


if "YearsSinceLastPromotion" in df.columns:

    if employee_data[
        "YearsSinceLastPromotion"
    ] >= 3:

        reasons.append(
            "Long promotion delay"
        )


if "DistanceFromHome" in df.columns:

    if employee_data[
        "DistanceFromHome"
    ] >= 15:

        reasons.append(
            "Long distance from home"
        )


if "JobInvolvement" in df.columns:

    if employee_data["JobInvolvement"] <= 2:

        reasons.append(
            "Low job involvement"
        )


if len(reasons) == 0:

    reasons.append(
        "No major predefined risk factor detected"
    )


for number, reason in enumerate(
    reasons,
    start=1
):

    print(
        number,
        ".",
        reason
    )


# ============================================================
# 39. WHAT-IF SCENARIO ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("WHAT-IF SCENARIO ANALYSIS")
print("=" * 70)


# Copy employee data

what_if_employee = (
    employee_data[
        X.columns
    ].copy()
)


# Original prediction

original_probability = (
    best_model
    .predict_proba(
        pd.DataFrame(
            [what_if_employee]
        )
    )[0][1]
)


print(
    "Original Risk:",
    round(
        original_probability * 100,
        2
    ),
    "%"
)


# ------------------------------------------------------------
# Scenario 1: Remove overtime
# ------------------------------------------------------------

if "OverTime" in what_if_employee.index:

    if str(
        what_if_employee["OverTime"]
    ).lower() == "yes":

        scenario_employee = (
            what_if_employee.copy()
        )

        scenario_employee[
            "OverTime"
        ] = "No"

        scenario_probability = (
            best_model
            .predict_proba(
                pd.DataFrame(
                    [scenario_employee]
                )
            )[0][1]
        )

        print(
            "\nScenario: Overtime changed Yes -> No"
        )

        print(
            "New Risk:",
            round(
                scenario_probability * 100,
                2
            ),
            "%"
        )

        print(
            "Risk Change:",
            round(
                (
                    scenario_probability
                    -
                    original_probability
                ) * 100,
                2
            ),
            "percentage points"
        )


# ------------------------------------------------------------
# Scenario 2: Improve job satisfaction
# ------------------------------------------------------------

if "JobSatisfaction" in what_if_employee.index:

    scenario_employee = (
        what_if_employee.copy()
    )

    scenario_employee[
        "JobSatisfaction"
    ] = 4

    scenario_probability = (
        best_model
        .predict_proba(
            pd.DataFrame(
                [scenario_employee]
            )
        )[0][1]
    )

    print(
        "\nScenario: Job Satisfaction changed to 4"
    )

    print(
        "New Risk:",
        round(
            scenario_probability * 100,
            2
        ),
        "%"
    )

    print(
        "Risk Change:",
        round(
            (
                scenario_probability
                -
                original_probability
            ) * 100,
            2
        ),
        "percentage points"
    )


# ============================================================
# 40. RISK DISTRIBUTION BY DEPARTMENT
# ============================================================

if "Department" in df.columns:

    risk_department_table = pd.crosstab(
        df["Department"],
        df["RiskCategory"]
    )

    risk_department_table = (
        risk_department_table
        .reindex(
            columns=[
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ],
            fill_value=0
        )
    )

    print("\n" + "=" * 70)
    print("RISK DISTRIBUTION BY DEPARTMENT")
    print("=" * 70)

    print(
        risk_department_table
    )

    risk_department_table.plot(
        kind="bar",
        figsize=(10, 6),
        color=[
            "seagreen",
            "orange",
            "tomato"
        ]
    )

    plt.title(
        "Risk Category Distribution by Department"
    )

    plt.xlabel("Department")
    plt.ylabel("Number of Employees")

    plt.xticks(rotation=0)

    plt.legend(
        title="Risk Category"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 41. SAVE COMPLETE PREDICTION DATASET
# ============================================================

df.to_csv(
    "Employee_Attrition_Risk_Results.csv",
    index=False
)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    "File: Employee_Attrition_Risk_Results.csv"
)


# ============================================================
# 42. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)

print(
    "Total Employees:",
    len(df)
)

print(
    "Actual Employees Left:",
    int(df["Attrition"].sum())
)

print(
    "Actual Attrition Rate:",
    round(
        df["Attrition"].mean() * 100,
        2
    ),
    "%"
)

print(
    "Low Risk Employees:",
    low_risk_count
)

print(
    "Medium Risk Employees:",
    medium_risk_count
)

print(
    "High Risk Employees:",
    high_risk_count
)

print(
    "Best Model:",
    best_model_name
)

print(
    "Best Model ROC-AUC:",
    round(
        model_scores[best_model_name],
        4
    )
)

print("\nRisk Framework:")
print("Low Risk    : Probability < 30%")
print("Medium Risk : Probability 30% - 60%")
print("High Risk   : Probability > 60%")

print("\n" + "=" * 70)
print("EMPLOYEE ATTRITION RISK ANALYSIS COMPLETED")
print("=" * 70)
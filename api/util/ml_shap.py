import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sympy import principal_branch
from xgboost import XGBRegressor


def generate_shap(input_df, col_dependent_var="output"):
    df = input_df.copy()
    X = df.drop([col_dependent_var], axis=1)
    y = df[col_dependent_var]


    model = XGBRegressor(random_state=42)

    scaler = MinMaxScaler()

    # pipe = Pipeline(steps)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)

    y_pred_test = model.predict(X_test)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test, check_additivity=True)
    df_XGB = pd.DataFrame(data={'Y_Actual':y_test, 'Prediction':y_pred_test}, index=X_test.index)

    df_SHAP = pd.DataFrame(shap_values,
                           index=X_test.index,
                           columns=[f"{col} contr." for col in X_test.columns])
    df_XGB_SHAP = pd.concat([df_XGB, df_SHAP], axis=1)

    df_report = pd.merge(left=X_test,
                         right=df_XGB_SHAP,
                         how='left',
                         left_index=False,
                         right_index=True,
                         left_on=X_test.index,
                         right_on=df_XGB_SHAP.index)
    return df_report
import pandas as pd
from scipy.__config__ import show
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sympy import principal_branch
from xgboost import XGBRegressor, XGBClassifier
import io


def generate_shap(input_df, col_dependent_var="output"):
    df = input_df.copy()
    X = df.drop([col_dependent_var], axis=1)
    y = df[col_dependent_var]

    model = XGBRegressor(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)

    y_pred_test = model.predict(X_test)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test, check_additivity=True)
    df_XGB = pd.DataFrame(
        data={'Y_Actual': y_test, 'Prediction': y_pred_test}, index=X_test.index)

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
    # return df_report
    return score_train, score_test


def gen_test_shap_plot(df):
    # train XGBoost model
    output_column_name = 'output'
    X = df.loc[:, df.columns != output_column_name]
    y = df.iloc[:, df.columns == output_column_name]
    # X, y = shap.datasets.adult()
    # X = X[:20]
    # y = y[:20]
    model = XGBClassifier().fit(X, y)

    # compute SHAP values
    explainer = shap.TreeExplainer(model, X)
    shap_values = explainer.shap_values(X)

    buf = io.BytesIO()
    f = plt.figure()

    shap.summary_plot(shap_values, features=X,
                      feature_names=X.columns, plot_type='bar', show=False)
    # shap.summary_plot(shap_values, X, show=False)
    f.savefig(buf, format='png', bbox_inches='tight', dpi=1200)
    plt.close()
    buf.seek(0)
    img = buf.read()
    return img

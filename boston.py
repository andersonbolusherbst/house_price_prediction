import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



st.write("""
# Boston House Price Prediction App
This app helps you to predict **House Prices**!
""")

st.image("US_Boston_US_Header.jpeg")

st.write("🏘 Our sophisticated regression models are used to predict continuous values.") 
st.write("🏘 In this example our model has been employed to predict the price of a house given a variety of features.") 
st.write("🏘 To properly illustrate this we have made use of the popular Boston House Price dataset")
st.write("🏘 Use the sliders on the left of the page to predict what your house in Boston may be worth!")
# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

st.header("Description of Features")
st.markdown(
    """
| Feature | Description |
| --- | --- |
| `CRIM` | per capita crime rate by town |
| `ZN` | proportion of residential land zoned for lots over 25,000 sq.ft. |
| `INDUS` | proportion of non-retail business acres per town |
| `CHAS` | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) |
| `NOX` | nitric oxides concentration (parts per 10 million) |
| `RM` | average number of rooms per dwelling |
| `AGE` | proportion of owner-occupied units built prior to 1940 |
| `DIS` | weighted distances to ﬁve Boston employment centers |
| `RAD` | index of accessibility to radial highways |
| `TAX` | full-value property-tax rate per $10,000 |
| `PTRATIO` | pupil-teacher ratio by town 12.|
| `LSTAT` | percentage of low income residents in the area |
| `DIV`| sliding scale of racial diversity in neighbourhood| 
"""
)


def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    DIV = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'DIV': DIV,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input Parameters')
st.write(df)


#Create a XGBoost Regressor
model = XGBRegressor()
model.fit(X, Y)

# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of Median House Price of Boston House ($)')
st.write((prediction*1000).round(2))


# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance (Bar Chart)')
fig, ax = plt.subplots()
plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(fig,bbox_inches='tight')

st.write("🏘  We can see from the above bar chart that the two features that the model placed the most weight on were LSTAT and RM")


st.header('Feature Importance (Waterfall)')
st.write("🏘  The above bar chart shows us the features with the biggest impact on the model but not which direction these features impacted our findings")
st.write("🏘  Below we can get a better understanding of this as our waterfall chart shows the effect of these features in both positive and negative directions")
fig, ax = plt.subplots()
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(fig,bbox_inches='tight')









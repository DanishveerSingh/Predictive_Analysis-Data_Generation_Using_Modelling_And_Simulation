import numpy as np
import pandas as pd
from gekko import GEKKO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def simulate_population(r, K):
    try:
        m = GEKKO(remote = False)
        m.time = np.linspace(0, 5, 50)
        P = m.Var(value = 10, lb = 0)
        m.Equation(P.dt() == r * P * (1 - P/K))
        m.options.IMODE = 4
        m.options.NODES = 3
        m.options.SOLVER = 1
        m.solve(disp = False)
        final_population = P.value[-1]
        max_population = max(P.value)
        avg_population = np.mean(P.value)
        return final_population, max_population, avg_population
    except:
        return None

parameter_bounds = {
    "r": (0.1, 2.0),    
    "K": (50, 500)        
}

records = []
while len(records) < 1000:
    r = np.random.uniform(*parameter_bounds["r"])
    K = np.random.uniform(*parameter_bounds["K"])
    result = simulate_population(r, K)
    if result is not None:
        final_p, max_p, avg_p = result
        records.append([r, K, final_p, max_p, avg_p])

df = pd.DataFrame(records, columns = ["r", "K","final_population","max_population","avg_population"])
print(df.head(20))

X = df[["r", "K"]]
y = df["final_population"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    results.append([name, r2, rmse, mse])

comparison_df = pd.DataFrame(results, columns = ["Model","R2 Score","RMSE","MSE"])
comparison_df = comparison_df.sort_values(by = "R2 Score", ascending = False)

print("\nModel Comparison:\n", comparison_df)
print("\nBest Model:\n", comparison_df.iloc[0])
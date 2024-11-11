import numpy as np
from sklearn.linear_model import LinearRegression

def main(wldb, spl):
    def helper(wldb, n, start, path, best_path, best_correlation):
        if n == 0:
            # check if the current path is better than the best path by correlation of the path and spl
            current_correlation = np.corrcoef(path, spl)[0, 1]
            if current_correlation > best_correlation:
                best_path[:] = path
                best_correlation = current_correlation
            return
        for i in range(start, len(wldb)):
            if not path or wldb[i] > path[-1]:
                helper(wldb, n-1, i+1, path + [wldb[i]], best_path, best_correlation)
    best_path = wldb[:len(spl)]
    n = len(spl)
    best_correlation = 0
    helper(wldb, n, 0, [], best_path, best_correlation)
    return best_path, best_correlation

pix = [25.97, 44.73, 149.19, 190.28, 294.58, 319.01]
wl = [534.10938, 534.32834, 535.5164, 536.00121, 537.2311, 537.4975]

best_path, best_correlation = main(wl, pix)
print("Best path:", best_path)
print("Best correlation:", best_correlation)

# fit a linear regression model
def fit_linear_regression(pix, best_path):
    model = LinearRegression()
    model.fit(np.array(pix).reshape(-1, 1), np.array(best_path).reshape(-1, 1))
    wl_predict = model.predict(np.array(pix).reshape(-1, 1))
    score = model.score(np.array(pix).reshape(-1, 1), np.array(best_path).reshape(-1, 1))
    return score, wl_predict

score, wl_predict = fit_linear_regression(pix, best_path)
print("Score:", score)

# plot the matched wavelengths and pixels
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.scatter(best_path, pix, c='red', label='Matched wavelengths')
plt.plot(wl_predict, pix, label='Predicted wavelengths')
plt.show()

# remove the pair with highest residual in descending order
# to find whether the correlation improves significantly
# otherwise, the pair with highest residual should be left in the list

residuals = np.abs(np.array(wl_predict).flatten() - np.array(wl))
max_residual_idx = np.argmax(residuals)
pix.pop(max_residual_idx)
wl.pop(max_residual_idx)

score, wl_predict = fit_linear_regression(pix, best_path)
print("Score after removing the pair with highest residual:", score)
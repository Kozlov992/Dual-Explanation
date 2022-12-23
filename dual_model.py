from sklearn.linear_model import LinearRegression
import numpy as np

#dual_func
class dual_func():
    def __init__(self, X_extr, explainable_func):
        self.X_extr = X_extr
        self.expl_f = explainable_func
        
    def __call__(self, lambdas):
        _generated_set = lambdas @ self.X_extr
        return self.expl_f(_generated_set)

#Model for feature-based explanations
class dual_MSE_explainer():
    def __init__(self, explainable_function, extreme_points=None, fit_intercept=False):
        self.dual_regressor = LinearRegression() #regression on lambdas, yields dual coefficients
        self.inverse_regressor = LinearRegression(fit_intercept=fit_intercept)# obtains coefficients in feature space.
                                                                              # if "fit_intercept" is True, returns coefficients
                                                                              # of model with independent term (a1*x1+...+an*xn+a0)
        self.extreme_points = extreme_points
        self.explainable_function = explainable_function
        self.dual_function = None
        self.explainble_point = None
        self.b = None
        self.a = None
        self.coef_ = None
        self.lambdas = None
        self.z = None
        self.fit_intercept = fit_intercept
        
    def set_extreme_points(self, extreme_points):
        self.extreme_points = extreme_points
    
    def explain(self, dual_samples_count='default'):
        alphas = np.ones(self.extreme_points.shape[0])
        if dual_samples_count == 'default':
            points_to_generate_num = 3 * self.extreme_points.shape[0]
        else:
            points_to_generate_num = dual_samples_count
        self.lambdas = np.random.dirichlet(alphas, points_to_generate_num)
        self.dual_function = dual_func(self.extreme_points, self.explainable_function)
        self.z = self.dual_function(self.lambdas)
        self.dual_regressor.fit(self.lambdas[:, :-1], self.z)
        
        self.b = np.append(self.dual_regressor.coef_ + self.dual_regressor.intercept_, self.dual_regressor.intercept_)
        self.inverse_regressor.fit(self.extreme_points, self.b)
        if self.fit_intercept:
            self.a = np.array([*self.inverse_regressor.coef_, self.inverse_regressor.intercept_])
        else:
            self.a = self.inverse_regressor.coef_
        return self.a
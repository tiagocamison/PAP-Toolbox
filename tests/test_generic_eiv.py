"""Generic nonlinear errors-in-variables regression checks."""
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import minimize

import pap_toolbox as pap


def _linear(x, m, b):
    return m*x+b


def _quadratic(x, a, b):
    return a*x*x+b


def _data():
    x = np.array([.4,.9,1.6,2.1,2.7,3.2,3.8])
    y = np.array([.7,2.3,3.4,6.0,9.4,11.3,18.1])
    dx = np.array([.15,.2,0,.35,.25,.3,.5])
    dy = np.array([.3,.4,.2,.4,.3,.5,.4])
    return dict(x=x,y=y,dx=dx,dy=dy)


def test_generic_linear_matches_profiled_linear_wrapper():
    d = _data()
    generic = pap.fit_curve(_linear,d,"x","y",x_err_col="dx",y_err_col="dy")
    linear = pap.linear_fit(d,"x","y",xerr="dx",yerr="dy")
    np.testing.assert_allclose(generic.params,linear.params,rtol=2e-6,atol=1e-7)
    assert generic.chi2 == pytest.approx(linear.chi2,rel=1e-8)
    np.testing.assert_allclose(generic.pulls,linear.pulls,rtol=2e-5,atol=1e-7)


def test_nonlinear_against_independent_joint_objective_and_covariance():
    d = _data()
    x,y,dx,dy = (d[k] for k in ("x","y","dx","dy"))
    free = dx>0
    fit = pap.fit_curve(_quadratic,d,"x","y",x_err_col="dx",y_err_col="dy",
                        p0=[1,.5])
    def objective(z):
        X=x.copy()
        X[free]=z[2:]
        return np.sum(((X[free]-x[free])/dx[free])**2)+np.sum(((y-_quadratic(X,*z[:2]))/dy)**2)
    ref = minimize(objective,np.r_[1.,.5,x[free]],method="BFGS",options={"gtol":1e-7})
    # BFGS can report precision loss close to the optimum; compare objective
    # and parameters directly, not a library-specific termination message.
    np.testing.assert_allclose(fit.params,ref.x[:2],rtol=3e-6,atol=3e-6)
    assert fit.chi2 == pytest.approx(ref.fun,rel=1e-8)
    assert fit.fit_method=="errors_in_variables"
    X=fit.x_adjusted
    assert X[2]==x[2]
    assert np.any(np.abs(X[free]-x[free])>1e-3)
    np.testing.assert_array_equal(fit.x_data,x)
    expected=np.sum(((X[free]-x[free])/dx[free])**2)+np.sum(((y-fit.evaluate(X))/dy)**2)
    assert fit.chi2==pytest.approx(expected)
    assert np.dot(fit.pulls,fit.pulls)==pytest.approx(expected)
    assert fit.ndof==len(x)-2
    # Analytic nuisance-eliminated Gauss-Newton information for a*x²+b.
    A=np.column_stack((X**2,np.ones_like(X)))
    sigma=np.hypot(dy,2*fit.params[0]*X*dx)
    expected_cov=np.linalg.inv((A/sigma[:,None]).T@(A/sigma[:,None]))
    np.testing.assert_allclose(fit.cov_matrix,expected_cov,rtol=3e-6,atol=1e-9)


def test_bounds_and_disjoint_ranges_with_eiv():
    d=_data()
    fit=pap.fit_curve(_quadratic,d,"x","y",x_err_col="dx",y_err_col="dy",
                      p0=[.6,1],bounds=([.1,-5],[.8,5]),fitranges=[(.4,1.6),(2.7,3.8)])
    assert fit.params[0]==pytest.approx(.8,abs=1e-7)
    np.testing.assert_array_equal(fit.x_data,d["x"][[0,1,2,4,5,6]])
    assert fit.ndof==4
    assert np.isfinite(fit.cov_matrix).all()


def test_all_zero_horizontal_errors_preserve_curve_fit():
    d=_data()
    a=pap.fit_curve(_quadratic,d,"x","y",y_err_col="dy")
    b=pap.fit_curve(_quadratic,d,"x","y",x_err_col=0.,y_err_col="dy")
    np.testing.assert_array_equal(a.params,b.params)
    np.testing.assert_array_equal(a.cov_matrix,b.cov_matrix)
    assert a.chi2==b.chi2
    assert b.fit_method=="vertical"


def test_exact_y_matches_inverse_linear_and_has_correct_chi2():
    d=_data()
    dx=np.full(len(d["x"]),.25)
    fit=pap.fit_curve(_linear,d,"x","y",x_err_col=dx,p0=[4.,-1.])
    ref=pap.linear_fit(d,"x","y",xerr=dx)
    np.testing.assert_allclose(fit.params,ref.params,rtol=2e-6,atol=1e-7)
    np.testing.assert_allclose(fit.cov_matrix,ref.cov_matrix,rtol=3e-6,atol=1e-9)
    np.testing.assert_allclose(fit.evaluate(fit.x_adjusted),d["y"],atol=1e-7)
    assert fit.chi2==pytest.approx(np.sum(((fit.x_adjusted-d["x"])/dx)**2))
    assert np.sum(fit.pulls**2)==pytest.approx(fit.chi2)


def test_exact_y_nonlinear_matches_analytic_inverse_model():
    x=np.array([.5,1.1,1.8,2.2,2.9,3.4])
    y=np.array([.6,2.3,5.,9.,16.,22.])
    dx=np.array([.12,.2,.15,.22,.3,.25])
    # y=a*x² (positive branch). Exact y => X=sqrt(y)/sqrt(a).
    # Inverse weighted linear model with coefficient c=1/sqrt(a).
    root=np.sqrt(y)
    c=np.sum(root*x/dx**2)/np.sum(root**2/dx**2)
    ref_a=1/c**2
    ref_var=(2/c**3)**2/np.sum(root**2/dx**2)
    fit=pap.fit_curve(lambda x,a:a*x*x,dict(x=x,y=y),"x","y",
                      x_err_col=dx,p0=[1.5],bounds=(.1,10))
    assert fit.params[0]==pytest.approx(ref_a,rel=2e-6)
    assert fit.cov_matrix[0,0]==pytest.approx(ref_var,rel=3e-6)
    np.testing.assert_allclose(fit.evaluate(fit.x_adjusted),y,atol=1e-7)


def test_exact_y_rejects_mixed_fixed_and_movable_x():
    d=_data()
    with pytest.raises(ValueError,match="strikt positiv"):
        pap.fit_curve(_quadratic,d,"x","y",x_err_col="dx")


def test_sparse_path_matches_analytic_linear_objective():
    x=np.linspace(0,8,120)
    y=2*x+1+.2*np.sin(x)
    d=dict(x=x,y=y)
    fit=pap.fit_curve(_linear,d,"x","y",x_err_col=.2,y_err_col=.15,p0=[2,1])
    ref=pap.linear_fit(d,"x","y",xerr=.2,yerr=.15)
    np.testing.assert_allclose(fit.params,ref.params,rtol=1e-6,atol=1e-7)
    assert fit.chi2==pytest.approx(ref.chi2,rel=1e-8)


def test_eiv_residual_plot_and_original_errors(monkeypatch):
    monkeypatch.setattr(plt,"show",lambda:None)
    d=_data()
    fit=pap.fit_curve(_quadratic,d,"x","y",x_err_col="dx",y_err_col="dy",
                      show=True,residuals=True,standardized=True,plot_band=False)
    try:
        assert len(fit.axes)==2
        assert fit.axes[1].get_ylabel()=="Pulls"
        np.testing.assert_array_equal(fit.x_err,d["dx"])
        np.testing.assert_array_equal(fit.y_err,d["dy"])
        expected=np.hypot(d["dy"],2*fit.params[0]*d["x"]*d["dx"])
        np.testing.assert_allclose(fit.effective_y_err,expected,rtol=1e-7)
    finally:
        plt.close(fit.axes[0].figure)

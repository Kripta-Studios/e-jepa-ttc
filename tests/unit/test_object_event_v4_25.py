import numpy as np
from e_jepa_ttc.training.object_event_v4_25 import (
    apply_geometry_calibration,design_matrix,fit_geometry_calibration,
    nonnegative_ridge_with_prior,predict_readout,
)


def test_geometry_calibration_orients_and_scales():
    x=np.array([-2.,-1.,1.,2.]); y=0.5*x
    c=fit_geometry_calibration(x,y)
    assert c.orientation==1.0
    np.testing.assert_allclose(apply_geometry_calibration(x,c),y,rtol=1e-6,atol=1e-6)


def test_geometry_calibration_flips_wrong_orientation():
    x=np.array([-2.,-1.,1.,2.]); y=-0.25*x
    c=fit_geometry_calibration(x,y)
    assert c.orientation==-1.0
    assert c.slope>0


def test_nonnegative_ridge_recovers_positive_solution():
    x=np.array([[1.,0.],[1.,1.],[1.,2.],[1.,3.]])
    y=x@np.array([0.8,0.2])
    c=nonnegative_ridge_with_prior(x,y,ridge=0.0,prior=np.array([1.,0.]))
    np.testing.assert_allclose(c,[0.8,0.2],atol=1e-8)


def test_nonnegative_ridge_rejects_negative_geometry_coefficient():
    x=np.array([[1.,0.],[1.,1.],[1.,2.],[1.,3.]])
    y=1.0-0.5*x[:,1]
    c=nonnegative_ridge_with_prior(x,y,ridge=0.0,prior=np.array([1.,0.]))
    assert c[1]==0.0
    assert np.all(c>=0)


def test_prior_keeps_baseline_anchor_when_geometry_is_noise():
    rng=np.random.default_rng(4); baseline=np.linspace(-1,1,200); noise=rng.normal(size=200)
    x=np.column_stack([baseline,noise]); y=baseline.copy()
    c=nonnegative_ridge_with_prior(x,y,ridge=0.1,prior=np.array([1.,0.]))
    assert c[0]>0.95
    assert c[1]<0.05


def test_design_matrix_requires_baseline_and_prediction_shape():
    b=np.array([1.,2.]); d=np.array([3.,4.]); v=np.array([5.,6.])
    x,names,prior=design_matrix(b,d,v,['baseline','divergence','vertical'])
    assert names==('baseline','divergence','vertical')
    np.testing.assert_allclose(prior,[1.,0.,0.])
    pred=predict_readout(x,np.array([1.,0.,0.]))
    np.testing.assert_allclose(pred,b)

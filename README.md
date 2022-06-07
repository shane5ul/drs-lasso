# drs-lasso

Given oscillatory shear measurements G*(w), use LASSO regression to (i) **SMEL test**: assess whether data are compliant with Kramers-Kronig relations (KKR); and (ii) infer a smooth discrete relaxation spectrum (DRS) using the crude DRS obtained during SMEL test


## SMEL Test

The file DRS_SMELtest.py shows an example (using the data test3.dat for a linear-linear blend) of extracting the discrete relaxation spectrum from G*(w) data. The function:

`g, tau, G0, alpha, score = lassoFit(wexp, Gexp, isPlateau=False, decade_density=5, verbose=False)`

returns the spectrum denoted by DRS-ST (`g, tau`) in the associated paper. `G0` is the plateau modulus, which is also fitted if the flag `isPlateau` is set to `True`. `score` is the coefficient of determination or $R^2$; when it is greater than 0.95, the data pass the SMEL test and can be considered KKR-compliant.

DRS-ST is generally noisy, but the computation is very efficient. If the goal is only to test the KKR-compliance of data efficiently, then this function can be recommended.

## Fine-Tuned Spectrum

The additional function `g, tau, G0 = get_DRS_ST_FT(wexp, Gexp, isPlateau)`

returns a smoother relaxation spectrum, by (i) running `lassoFit(..., decade_density=2, ...)` with a smaller mode density, and (ii) solving a nonlinear regression problem to fine-tune the spectrum.

This spectrum is more meaningful than DRS-ST, and is recommended if the goal is to obtain a parisimonious and meaningful spectrum.

## Reference

S. Poudel and S. Shanbhag, "Efficient Test to Evaluate the Consistency of Elastic and Viscous Moduli with Kramers-Kronig Relations", to be submitted to Korea-Australia Journal of Rheology.

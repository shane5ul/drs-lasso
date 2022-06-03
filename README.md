# drs-lasso
Use Lasso regression to infer DRS from oscillatory shear experiments G*(w)

The file DRS_SMELtest.py shows an example (using test3.dat) of extracting the discrete relaxation spectrum from G*(w) data. The function:

`g, tau, G0, alpha, score = lassoFit(wexp, Gexp, isPlateau, decade_density=5, verbose=False)`

returns the spectrum denoted by DRS-ST (`g, tau`) in the paper to be submitted on this work. `G0` is the plateau modulus, which is also fitted if the flag `isPlateau` is set to `True`. `score` is the coefficient of determination or $R^2$; when it is greater than 0.95, the data are KKR-compliant.

This spectrum is generally noisy, but the computation is very efficient. If the goal is only to test the KKR-compliance of data efficiently, then this function can be recommended.

The additional function `g, tau, G0 = get_DRS_ST_FT(wexp, Gexp, isPlateau)`

returns a smoother relaxation spectrum, by (i) running `lassoFit(..., decade_density=2, ...)` with a smaller mode density, and (ii) solving a nonlinear regression problem to fine-tune the spectrum.

This spectrum is more meaningful than DRS-ST, and is recommended if the goal is to obtain a parisimonious and meaningful spectrum.


# Targeted Halekas Method Questions

These questions are motivated by the completed clean-room reconstruction and
are intended for a concise author inquiry. They identify private implementation
details that can distinguish remaining hypotheses without asking the author to
reconstruct the whole project.

1. Which exact LP ER spectrum or UTC record produced Figure 5? The archived
   raster morphology and beam amplitude favor spectrum 654
   (`14:45:28-14:46:38`) over spectrum 653.
2. Was normalization performed by a conjugate incident pitch bin, an
   incident-hemisphere mean, or an incident distribution model?
3. Were native sectors normalized before pitch binning, or were fluxes binned
   before normalization? Were pitch bins equally weighted or weighted by solid
   angle, dwell, or uncertainty?
4. How many pitch bins were used, and was the synthetic boundary evaluated at
   bin centers, oversampled, or convolved with angular response?
5. Was `B_S/B_M` completely free in Figure 5, constrained by magnetic geometry,
   or initialized/regularized near an independently expected value?
6. What exact loss-cone interior value, backscatter/refilling treatment, and
   logarithmic floor entered the fit rather than only the display?
7. How were beam amplitude, energy width, and pitch width set or fitted? Was
   the beam evaluated through the effective `Delta E / E ~= 0.48` response?
8. Which energy channels entered the objective, particularly around the beam,
   sunlight photoelectrons, high-energy low-count cells, and plotted gaps?
9. Was the least-squares log objective weighted by counts, flux errors, energy,
   pitch support, or any manual quality factor?
10. What automatic error statistic and threshold were used before the reported
    by-hand quality-control pass?
11. Is the original IDL routine, parameter grid, or Figure 5 intermediate
    normalized array still available for methodological comparison?

The current evidence and exact experiment commit should accompany the inquiry;
the request should make clear that publication-scale voltage has been
reproduced, while independent boundary and field-ratio identification remains
unresolved.

# Current Loss-Cone Forward Model

File reference: `src/model.py:13-49`

## Inputs
- `energy_grid`: 1D array of ER sweep energies (eV) for a 15-row chunk.
- `pitch_grid`: 2D array of pitch angles (degrees) aligned with ER channels.
- `U_surface`: Trial surface potential offset U_surface (V).
- `bs_over_bm`: Mirror ratio \(B_s/B_m\).
- `beam_width_eV`, `beam_amp`: optional controls for a Gaussian “beam” term (default 0).

## Constructed Features
For each energy \(E\), the model computes:
\[
x(E) = \frac{B_s}{B_m} \left( 1 + \frac{U_{\text{surface}}}{E} \right)
\]
and maps it to a loss-cone cutoff angle
\[
\alpha_c(E) = 
\begin{cases}
0^\circ, & x \le 0 \\
90^\circ, & x \ge 1 \\
\arcsin\left(\sqrt{x}\right), & \text{otherwise}.
\end{cases}
\]
Downward-going channels (\(\text{pitch} \le 180^\circ - \alpha_c\)) are set to 1.0; the remainder stay 0. This encodes only the magnetic loss cone opening.

## Optional Gaussian Beam
If `beam_width_eV > 0` and `beam_amp > 0`, an additive term
\[
G(E,\alpha) = \text{beam\_amp}\,
\exp\!\left[-\frac{1}{2}\Bigl(\frac{E - |U_{\text{surface}}|}{\text{beam\_width}}\Bigr)^2\right]
\exp\!\left[-\frac{1}{2}\left(\frac{\alpha - 180^\circ}{\sigma_\alpha}\right)^2\right]
\]
is applied, where \(\alpha\) is the pitch angle, \(\sigma_\alpha=\texttt{LOSS\_CONE\_BEAM\_PITCH\_SIGMA\_DEG}\),
and \(\text{beam\_width} = \max\bigl(|U_{\text{surface}}|\,\texttt{LOSS\_CONE\_BEAM\_WIDTH\_FACTOR},\,\varepsilon\bigr)\).
The loss-cone fitter optimizes only the scalar `beam_amp` (bounded by `LOSS_CONE_BEAM_AMP_MIN/MAX`),
while the energy width and pitch spread follow the configured constants.

## Consequences
- The forward model matches the paper's magnetic cut-off geometry and includes the upward-going secondary beam signature.
- Beam amplitude defaults to zero when the optimizer favors no additional emission, but can grow (within bounds) to match observed beams near \(E \approx |U_{\text{surface}}|\).
- Future refinement can focus on data-driven tuning of the beam width and amplitude priors.

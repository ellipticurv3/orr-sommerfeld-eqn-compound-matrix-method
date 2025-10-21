import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, root

###################################
# 1. Orr-Sommerfeld operator for plane Poiseuille flow
####################################
def A_matrix_plane(y, k, R, c):
    """
    Construct the 4×4 Orr–Sommerfeld matrix for plane Poiseuille flow.

    The base flow is:
         U(y) = 1 - y²,       U''(y) = -2,
    with y in [-1, 1].

    Starting from the Orr–Sommerfeld equation
         ϕ'''' - 2 k² ϕ'' + k⁴ ϕ = i k R [ (U-c)(ϕ''-k²ϕ) - U'' ϕ ],
    we obtain a first‑order system by letting:
         x₁ = ϕ,  x₂ = ϕ',  x₃ = ϕ'',  x₄ = ϕ'''.
    Then, writing:
         x₄' = A₃₀ ϕ + A₃₂ ϕ'',
    the coefficients are defined as:
         A₃₀ = - k⁴ - i k R [ k² (U-c) + U'' ],
         A₃₂ = 2 k² + i k R (U-c).
    """
    U = 1 - y**2
    U_dd = -2.0
    ikR = 1j * k * R

    A = np.zeros((4, 4), dtype=np.complex128)
    A[0, 1] = 1.0
    A[1, 2] = 1.0
    A[2, 3] = 1.0
    A[3, 0] = - k**4 - ikR * ( k**2 * (U - c) + U_dd )
    A[3, 1] = 0.0
    A[3, 2] = 2 * k**2 + ikR * (U - c)
    A[3, 3] = 0.0
    return A

#######################
# 2. Compound matrix construction
#######################
def compound_matrix(A):
    """
    Constructs the second compound (6×6) matrix from the 4×4 matrix A
    by forming all 2×2 minors with respect to the basis:
         {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}.
    """
    basis = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    B = np.zeros((6, 6), dtype=np.complex128)
    for row, (p, q) in enumerate(basis):
        for col, (r, s) in enumerate(basis):
            term1 = A[p, r] * (1 if q == s else 0)
            term2 = -A[p, s] * (1 if q == r else 0)
            term3 = A[q, s] * (1 if p == r else 0)
            term4 = -A[q, r] * (1 if p == s else 0)
            B[row, col] = term1 + term2 + term3 + term4
    return B

def compound_rhs_plane(y, psi, k, R, c):
    """
    Right-hand side of the compound system for plane Poiseuille flow:
         psi' = B(y) psi,
    where B(y) is constructed from the matrix A(y, k, R, c).
    """
    A = A_matrix_plane(y, k, R, c)
    B = compound_matrix(A)
    return B @ psi

#########################
# 3. Eigenvalue solvers for plane Poiseuille flow
########################
def solve_eigenvalue_plane(k, R, c_guess, y_span=(-1, 1), num_points=50, plot_flag=False):
    """
    Search for a real eigenvalue (phase speed c) for plane Poiseuille flow.

    An IVP for the compound system is solved from y=-1 to y=1.
    The boundary residual is defined as the real part of psi₁ at y=1.
    Brent's method (via root_scalar) is used to find c such that the residual vanishes.
    """
    psi0 = np.zeros(6, dtype=np.complex128)
    psi0[-1] = 1.0  # normalization

    def boundary_condition_residual(c):
        sol = solve_ivp(
            fun=lambda y, psi: compound_rhs_plane(y, psi, k, R, c),
            t_span=y_span,
            y0=psi0,
            method='DOP853',
            t_eval=np.linspace(y_span[0], y_span[1], num_points),
            rtol=1e-7,
            atol=1e-9
        )
        return np.real(sol.y[0, -1])

    if plot_flag:
        c_values = np.linspace(0.1, 1.5, 200)
        residuals = [boundary_condition_residual(c) for c in c_values]
        plt.figure(figsize=(8, 6))
        plt.plot(c_values, residuals, label='Residual at y = {}'.format(y_span[1]))
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Eigenvalue candidate c')
        plt.ylabel('Residual (Re)')
        plt.title('Residual vs. c for Plane Poiseuille Flow')
        plt.legend()
        plt.grid(True)
        plt.show()

    result = root_scalar(boundary_condition_residual, bracket=[0.1, 1.5], method='brentq')
    if result.converged:
        print(f"Computed eigenvalue c = {result.root}")
        return result.root
    else:
        print("Eigenvalue computation did not converge.")
        return None

def solve_eigenvalue_problem_plane(R, alpha, y_start, y_end, c):
    """
    For a given (possibly complex) eigenvalue candidate c, integrate the compound
    system on y in [y_start, y_end] for plane Poiseuille flow and evaluate the
    discriminant D = psi₁(y_end). The true eigenvalue satisfies D = 0.

    Here, alpha is the wavenumber (alias for k).
    """
    psi0 = np.zeros(6, dtype=np.complex128)
    psi0[-1] = 1.0
    sol = solve_ivp(
        fun=lambda y, psi: compound_rhs_plane(y, psi, alpha, R, c),
        t_span=(y_start, y_end),
        y0=psi0,
        method='DOP853',
        t_eval=np.linspace(y_start, y_end, 50),
        rtol=1e-7,
        atol=1e-9
    )
    Y_final = sol.y[:, -1]
    D = Y_final[0]
    return D, sol

####################
# 4. Complex Eigenvalue Search and Classification
####################
def solve_complex_eigenvalue_plane(k, R, c_initial, y_span=(-1,1), num_points=50):
    """
    Use a 2D root finder to search for a complex eigenvalue c for plane Poiseuille flow.
    We define the residual as F(c) = [Re(D(c)), Im(D(c))],
    where D(c) is the discriminant from the compound system.
    """
    def F(v):
        c_candidate = v[0] + 1j*v[1]
        D, _ = solve_eigenvalue_problem_plane(R, k, y_span[0], y_span[1], c_candidate)
        return [np.real(D), np.imag(D)]

    v0 = [np.real(c_initial), np.imag(c_initial)]
    sol = root(F, v0, method='hybr', tol=1e-8)
    if sol.success:
        return sol.x[0] + 1j*sol.x[1]
    else:
        raise RuntimeError("Complex eigenvalue search did not converge")

def find_all_eigenvalues_complex(k, R, y_span=(-1,1), num_points=50,
                                 N_real=5, N_imag=6, tol=1e-3, max_eigen=30):
    """
    Scan a grid of initial guesses over the complex-c plane and use solve_complex_eigenvalue_plane
    to collect up to max_eigen distinct eigenvalues.

    N_real and N_imag determine the number of grid points for the initial guess.
    """
    initial_guesses = []
    # Adjust these ranges based on expected eigenvalue distribution.
    real_range = np.linspace(0.1, 1.5, N_real)
    imag_range = np.linspace(-0.9, 0.1, N_imag)
    for r in real_range:
        for i in imag_range:
            initial_guesses.append(r + 1j*i)

    found = []
    for guess in initial_guesses:
        try:
            ev = solve_complex_eigenvalue_plane(k, R, guess, y_span, num_points)
            # Check against duplicates
            if not any(np.abs(ev - candidate) < tol for candidate in found):
                found.append(ev)
                print(f"Found eigen: {ev:.6f} from initial guess {guess:.6f}")
        except Exception as e:
            continue
        if len(found) >= max_eigen:
            break
    return found

def classify_mode(ev):
    """
    Simple heuristic classification based on the imaginary part of c.
    (These thresholds are chosen arbitrarily to mimic the families observed in Mack, 1976.)
      - P family: Im(c) > -0.05  (triangles)
      - A family: -0.3 < Im(c) <= -0.05  (circles)
      - S family: Im(c) <= -0.3  (squares)
    """
    if ev.imag > -0.05:
        return 'P'
    elif ev.imag > -0.3:
        return 'A'
    else:
        return 'S'

#########################
# 5. Compute the Eigenfunction for a Selected Eigenvalue
#########################
def compute_eigenfunction_plane(k, R, c, y_span=(-1,1), num_points=200, delta=1e-5):
    """
    Compute the eigenfunction (phi(y)) for a given eigenvalue c by integrating the
    original 4×4 system:
         Y' = A_matrix_plane(y, k, R, c) Y,
    where Y = [phi, phi', phi'', phi'''].

    To avoid the trivial solution imposed by the homogeneous boundary conditions,
    we start slightly inside the boundary at y = y_start + delta.

    Here we choose initial conditions approximating a no-slip boundary in a channel:
         phi(-1+delta) = 0, phi'(-1+delta)=0, phi''(-1+delta) = 1, phi'''(-1+delta) = 0.
    The eigenfunction is returned as phi(y) (i.e. the first component).
    """
    y0 = y_span[0] + delta
    Y0 = [0, 0, 1, 0]  # chosen normalization near the wall
    sol = solve_ivp(lambda y, Y: A_matrix_plane(y, k, R, c) @ Y,
                    t_span=(y0, y_span[1]), y0=Y0,
                    t_eval=np.linspace(y0, y_span[1], num_points),
                    rtol=1e-7, atol=1e-9)
    return sol.t, sol.y[0]  # return y and phi(y)

########################
# 6. Main Script: Eigenvalue Spectrum, Table, and Plots in the Complex c-plane
#######################
if __name__ == "__main__":
    # Parameters for plane Poiseuille flow
    k = 1.0             # Wavenumber (α)
    R = 10000.0         # Reynolds number
    y_domain = (-1, 1)  # Channel walls at y = -1 and y = 1

    print("Starting Orr–Sommerfeld eigenvalue computation for plane Poiseuille flow...")

    # (A) Real eigenvalue search (optional, for comparison)
    c_guess_real = 0.3   # Initial guess for a real eigenvalue
    eigenvalue = solve_eigenvalue_plane(k, R, c_guess_real, y_span=y_domain, num_points=50, plot_flag=True)
    print("Computed eigenvalue (real search):", eigenvalue)

    # (B) PART 2: Evaluate discriminant and build a contour plot in the complex-c plane.
    # Here we assess the discriminant for a sample candidate.
    Re_val = R
    alpha = k   # for notational consistency, alpha = k
    y_start = y_domain[0]
    y_end = y_domain[1]
    c_guess_complex = 0.3 + 0.1j  # A sample complex candidate

    # Evaluate the discriminant for a test candidate:
    D_eval, sol_eval = solve_eigenvalue_problem_plane(Re_val, alpha, y_start, y_end, c_guess_complex)
    print("For c =", c_guess_complex, ", the discriminant D =", D_eval)

    # Build a grid on the complex-c plane.
    Cr_vals = np.linspace(0, 1, 100)
    Ci_vals = np.linspace(-0.9, 0.1, 100)
    Cr, Ci = np.meshgrid(Cr_vals, Ci_vals)
    # Precompute the real and imaginary parts of the discriminant on the grid.
    Dr = np.zeros_like(Cr, dtype=float)
    Di = np.zeros_like(Ci, dtype=float)

    print("Computing the discriminant on the c-grid for contour plotting...")
    for i in range(Cr.shape[0]):
        for j in range(Cr.shape[1]):
            c_candidate = Cr[i, j] + 1j * Ci[i, j]
            D_val, _ = solve_eigenvalue_problem_plane(Re_val, alpha, y_start, y_end, c_candidate)
            Dr[i, j] = np.real(D_val)
            Di[i, j] = np.imag(D_val)

    # --- Graph: Zero Contour Lines ---
    # Create a single-axes figure for the zero-contours.
    fig, ax = plt.subplots(figsize=(8, 6))
    cs1 = ax.contour(Cr, Ci, Dr, levels=[0], colors='blue', linewidths=2)
    cs2 = ax.contour(Cr, Ci, Di, levels=[0], colors='red', linestyles='dashed', linewidths=2)
    ax.clabel(cs1, inline=True, fontsize=10)
    ax.clabel(cs2, inline=True, fontsize=10)
    ax.set_xlabel('Cr (Real part of c)')
    ax.set_ylabel('Ci (Imaginary part of c)')
    ax.set_title('Zero Contours: Re(D)=0 (Blue) and Im(D)=0 (Red Dashed)')
    ax.grid(True)
    # Mark the computed (real) eigenvalue on the real axis.
    if eigenvalue is not None:
        ax.plot(eigenvalue, 0, 'ko', markersize=8, label='Computed Eigenvalue')
        ax.legend()

    plt.tight_layout()
    plt.show()

    # (C) Complex eigenvalue search for up to 30 modes.
    print("\nSearching for complex eigenvalues (up to 30 modes)...")
    eigen_complex_list = find_all_eigenvalues_complex(k, R, y_span=y_domain, num_points=50,
                                                      N_real=5, N_imag=6, tol=1e-3, max_eigen=30)
    # Sort eigenvalues by real part.
    eigen_complex_list.sort(key=lambda x: x.real)

    # Print table of computed eigenvalues.
    print("\nTable of computed eigenvalues (first 30 modes):")
    print("{:<5s} {:>12s} {:>12s} {:>6s}".format("Mode", "Re(c)", "Im(c)", "Fam"))
    for i, ev in enumerate(eigen_complex_list[:30], 1):
        fam = classify_mode(ev)
        print("{:<5d} {:12.6f} {:12.6f} {:>6s}".format(i, ev.real, ev.imag, fam))

    # (D) Plot the eigenvalue distribution in the complex c-plane.
    # Markers: P -> triangle ('^'), A -> circle ('o'), S -> square ('s')
    marker_dict = {'P': '^', 'A': 'o', 'S': 's'}
    colors = {'P': 'red', 'A': 'blue', 'S': 'green'}

    plt.figure(figsize=(8, 6))
    for ev in eigen_complex_list:
        fam = classify_mode(ev)
        plt.scatter(ev.real, ev.imag, marker=marker_dict[fam], color=colors[fam], s=80,
                    label=f"{fam} family" if fam not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.title("Distribution of Eigenvalues of Plane Poiseuille Flow\n"
              "at α = 1 and Re = 10,000; Symmetric Mode (After Mack, 1976)")
    plt.gca().invert_yaxis()  # In many OS studies, Im(c) is plotted inverted.
    plt.grid(True)
    plt.legend()
    plt.show()

    # (E) Compute the eigenfunction for the unstable mode.
    # Here we select the eigenvalue with the maximum imaginary part as the unstable mode.
    if eigen_complex_list:
        unstable_mode = max(eigen_complex_list, key=lambda x: x.imag)
        print("Unstable eigenvalue (maximum Im):", unstable_mode)
        y_vals, phi_vals = compute_eigenfunction_plane(k, R, unstable_mode, y_span=y_domain, num_points=200)

        plt.figure(figsize=(8, 6))
        plt.plot(y_vals, phi_vals, 'b-', linewidth=2)
        plt.xlabel("y")
        plt.ylabel("Eigenfunction φ(y)")
        plt.title(f"Eigenfunction for Unstable Mode c = {unstable_mode:.6f}")
        plt.grid(True)
        plt.show()

    input("Press Enter to exit...")
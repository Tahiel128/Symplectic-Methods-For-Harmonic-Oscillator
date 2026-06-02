import numpy as np
from scipy.fft import fft, ifft, fftfreq

def paquete_gaussiano(x, q0, p0, sigma, hbar):
    psi = (
        (1 / (np.pi * sigma**2))**0.25 *
        np.exp(-(x - q0)**2 / (2 * sigma**2)) *
        np.exp(1j * p0 * x / hbar)
    )
    dx = x[1] - x[0]
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi

def exp_val_pos(psi, x, dx):
    # Expectation value of position: <psi| x |psi>.
    return np.real(np.sum(np.conj(psi) * x * psi) * dx)

def exp_val_mom(psi, k, hbar, dx, N):
    # Expectation value of momentum via Fourier transform.
    psi_k = fft(psi)
    return np.real(np.sum(np.conj(psi_k) * (hbar * k) * psi_k) * dx) / N

def exp_val_energy(psi, V, k, hbar, m, dx, N):
    # Expectation value of the Hamiltonian: kinetic + potential.
    psi_k = fft(psi)
    kin = np.real(
        np.sum(np.conj(psi_k) * (hbar**2 * k**2 / (2 * m)) * psi_k) * dx
    ) / N
    pot = np.real(np.sum(np.conj(psi) * V * psi) * dx)
    return kin + pot

def exp_norma(psi, dx):
    return np.sum(np.abs(psi)**2) * dx

def lie_trotter(psi0, n_steps, T_op, V_full, x, dx, k, V, hbar, m):
    # Lie–Trotter splitting (order 1):
    #   exp(-i H dt) ≈ exp(-i V dt) exp(-i T dt)
    N = len(x)
    qq, pp, HH = np.zeros(n_steps+1), np.zeros(n_steps+1), np.zeros(n_steps+1)
    psi = psi0.copy()
    for step in range(n_steps + 1):
        qq[step] = exp_val_pos(psi, x, dx)
        pp[step] = exp_val_mom(psi, k, hbar, dx, N)
        HH[step] = exp_val_energy(psi, V, k, hbar, m, dx, N)
        if step < n_steps:
            psi *= V_full
            psi = ifft(fft(psi) * T_op)
    return qq, pp, HH

def strang(psi0, n_steps, T_op, V_half, x, dx, k, V, hbar, m):
    # Strang splitting (order 2):
    #   exp(-i H dt) ≈ exp(-i V dt/2) exp(-i T dt) exp(-i V dt/2)
    N = len(x)
    qq, pp, HH = np.zeros(n_steps+1), np.zeros(n_steps+1), np.zeros(n_steps+1)
    psi = psi0.copy()
    for step in range(n_steps + 1):
        qq[step] = exp_val_pos(psi, x, dx)
        pp[step] = exp_val_mom(psi, k, hbar, dx, N)
        HH[step] = exp_val_energy(psi, V, k, hbar, m, dx, N)
        if step < n_steps:
            psi *= V_half
            psi = ifft(fft(psi) * T_op)
            psi *= V_half
    return qq, pp, HH

def simular(m=1.0, omega=2.0, hbar=1.0, q0=1.5, p0=1.0,
            dt=0.05, periodos=10, L=16.0, Nx=1024):
    T = periodos * 2 * np.pi / omega
    dx = L / Nx
    x = np.linspace(-L/2, L/2, Nx, endpoint=False)
    n_steps = int(T / dt)
    t = np.linspace(0, n_steps * dt, n_steps + 1)

    # Initial coherent state (Gaussian wave packet).
    sigma = np.sqrt(hbar / (2 * m * omega))
    psi0 = paquete_gaussiano(x, q0, p0, sigma, hbar)

    # Pre-compute operators for the splitting schemes.
    k = 2 * np.pi * fftfreq(Nx, dx)
    V = 0.5 * m * omega**2 * x**2
    T_op = np.exp(-1j * hbar * k**2 * dt / (2 * m))
    V_half = np.exp(-1j * V * dt / (2 * hbar))
    V_full = np.exp(-1j * V * dt / hbar)

    q_lt, p_lt, H_lt = lie_trotter(psi0, n_steps, T_op, V_full,
                                    x, dx, k, V, hbar, m)
    q_st, p_st, H_st = strang(psi0, n_steps, T_op, V_half,
                              x, dx, k, V, hbar, m)
    # Classical reference via Ehrenfest's theorem.
    q_cl = q0 * np.cos(omega * t) + p0 / (m * omega) * np.sin(omega * t)

    # Norm conservation check (unitarity).
    norm_lt = np.array([exp_norma(psi0, dx)])
    psi = psi0.copy()
    for step in range(n_steps):
        psi *= V_full
        psi = ifft(fft(psi) * T_op)
        norm_lt = np.append(norm_lt, exp_norma(psi, dx))

    norm_st = np.zeros(n_steps+1)
    psi = psi0.copy()
    for step in range(n_steps + 1):
        norm_st[step] = exp_norma(psi, dx)
        if step < n_steps:
            psi *= V_half
            psi = ifft(fft(psi) * T_op)
            psi *= V_half

    # Final wave function for density plots.
    psi_f_lt = psi0.copy()
    for _ in range(n_steps):
        psi_f_lt *= V_full
        psi_f_lt = ifft(fft(psi_f_lt) * T_op)

    psi_f_st = psi0.copy()
    for _ in range(n_steps):
        psi_f_st *= V_half
        psi_f_st = ifft(fft(psi_f_st) * T_op)
        psi_f_st *= V_half

    # Convergence study.
    dts_c = 0.08 / 2.0**np.arange(0, 5)
    err_lt_list, err_st_list = [], []
    for h in dts_c:
        n = int(T / h)
        tt = np.linspace(0, n * h, n + 1)
        k_h = 2 * np.pi * fftfreq(Nx, dx)
        T_op_h = np.exp(-1j * hbar * k_h**2 * h / (2 * m))
        V_full_h = np.exp(-1j * V * h / hbar)
        V_half_h = np.exp(-1j * V * h / (2 * hbar))
        ql, _, _ = lie_trotter(psi0, n, T_op_h, V_full_h,
                               x, dx, k_h, V, hbar, m)
        qs, _, _ = strang(psi0, n, T_op_h, V_half_h,
                          x, dx, k_h, V, hbar, m)
        qc = q0 * np.cos(omega * tt) + p0 / (m * omega) * np.sin(omega * tt)
        err_lt_list.append(np.max(np.abs(ql[-50:] - qc[-50:])))
        err_st_list.append(np.max(np.abs(qs[-50:] - qc[-50:])))

    c_lt = (
        np.polyfit(np.log(dts_c[-3:]), np.log(err_lt_list[-3:]), 1)
        if len(err_lt_list) > 3 else (0, 0)
    )
    c_st = (
        np.polyfit(np.log(dts_c[-3:]), np.log(err_st_list[-3:]), 1)
        if len(err_st_list) > 3 else (0, 0)
    )

    return {
        "t": t, "x": x, "psi0": psi0,
        "psi_f_lt": psi_f_lt, "psi_f_st": psi_f_st,
        "q_lt": q_lt, "q_st": q_st, "q_cl": q_cl,
        "p_lt": p_lt, "p_st": p_st,
        "H_lt": H_lt, "H_st": H_st,
        "norm_lt": norm_lt, "norm_st": norm_st,
        "dts_c": dts_c, "err_lt": err_lt_list, "err_st": err_st_list,
        "c_lt": c_lt, "c_st": c_st,
    }

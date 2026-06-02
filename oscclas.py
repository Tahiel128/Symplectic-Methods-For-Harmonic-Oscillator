import numpy as np

def flujo_exacto(t, m, omega, q0, p0):
    # Analytical solution of the harmonic oscillator.
    q = q0 * np.cos(omega * t) + p0 / (m * omega) * np.sin(omega * t)
    p = -m * omega * q0 * np.sin(omega * t) + p0 * np.cos(omega * t)
    return q, p

def euler_simplectico(q0, p0, dt, N, m, omega):
    # Symplectic Euler (order 1): update momentum first, then position.
    q, p = np.zeros(N), np.zeros(N)
    q[0], p[0] = q0, p0
    for i in range(N - 1):
        p[i+1] = p[i] - dt * m * omega**2 * q[i]
        q[i+1] = q[i] + dt * p[i+1] / m
    return q, p

def verlet(q0, p0, dt, N, m, omega):
    # Størmer–Verlet (order 2): symmetric composition.
    q, p = np.zeros(N), np.zeros(N)
    q[0], p[0] = q0, p0
    for i in range(N - 1):
        p_half = p[i] - 0.5 * dt * m * omega**2 * q[i]
        q[i+1] = q[i] + dt * p_half / m
        p[i+1] = p_half - 0.5 * dt * m * omega**2 * q[i+1]
    return q, p

def euler_explicito(q0, p0, dt, N, m, omega):
    # Explicit Euler (NOT symplectic): used as a counterexample.
    q, p = np.zeros(N), np.zeros(N)
    q[0], p[0] = q0, p0
    for i in range(N - 1):
        q[i+1] = q[i] + dt * p[i] / m
        p[i+1] = p[i] - dt * m * omega**2 * q[i]
    return q, p

def H(q, p, m, omega):
    # Hamiltonian (total energy).
    return 0.5 * p**2 / m + 0.5 * m * omega**2 * q**2

def simular(m=1.0, omega=2.0, q0=1.0, p0=0.0, dt=0.05, periodos=10):
    T = periodos * 2 * np.pi / omega
    t = np.arange(0, T, dt)
    N = len(t)

    q_ex, p_ex = flujo_exacto(t, m, omega, q0, p0)
    q_eu, p_eu = euler_simplectico(q0, p0, dt, N, m, omega)
    q_vl, p_vl = verlet(q0, p0, dt, N, m, omega)
    q_ns, p_ns = euler_explicito(q0, p0, dt, N, m, omega)

    H_ex = H(q_ex, p_ex, m, omega)
    H_eu = H(q_eu, p_eu, m, omega)
    H_vl = H(q_vl, p_vl, m, omega)
    H_ns = H(q_ns, p_ns, m, omega)
    H0 = H(q0, p0, m, omega)

    # Convergence study over multiple time steps.
    dts = 2.0**-np.arange(3, 10)
    err_eu, err_vl = [], []
    for h in dts:
        n = int(T / h)
        tt = np.arange(0, T, h)
        qe, _ = euler_simplectico(q0, p0, h, n, m, omega)
        qv, _ = verlet(q0, p0, h, n, m, omega)
        qr = q0 * np.cos(omega * tt) + p0 / (m * omega) * np.sin(omega * tt)
        err_eu.append(np.max(np.abs(qe[-50:] - qr[-50:])))
        err_vl.append(np.max(np.abs(qv[-50:] - qr[-50:])))

    # Fit slopes to estimate convergence order.
    c_eu = np.polyfit(np.log(dts[-4:]), np.log(err_eu[-4:]), 1)
    c_vl = np.polyfit(np.log(dts[-4:]), np.log(err_vl[-4:]), 1)

    return {
        "t": t, "q_ex": q_ex, "p_ex": p_ex,
        "q_eu": q_eu, "p_eu": p_eu, "H_eu": H_eu,
        "q_vl": q_vl, "p_vl": p_vl, "H_vl": H_vl,
        "q_ns": q_ns, "p_ns": p_ns, "H_ns": H_ns,
        "H_ex": H_ex, "H0": H0,
        "dts": dts, "err_eu": err_eu, "err_vl": err_vl,
        "c_eu": c_eu, "c_vl": c_vl,
    }

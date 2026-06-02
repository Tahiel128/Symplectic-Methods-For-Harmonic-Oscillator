# Symplectic Methods for Harmonic Oscillator

Numerical simulation of the classical and quantum harmonic oscillator using four symplectic integrators. Part of an ongoing project with mathematician Franz Chouly, PhD.

## Contents

- `oscclas.py` — Classical simulation: Symplectic Euler, Verlet, explicit Euler
- `osccuan.py` — Quantum simulation: Lie–Trotter, Strang (spectral splitting)
- `interfaz/gui.py` — Tkinter GUI for parameter control and live plotting

## Requirements

Following packages are required:
- NumPy
- SciPy
- MatPlotLib

On Linux, install Tk as well:
- Debian-based distros:
  ```
  sudo apt install python3-tk
  ```
- Arch-based distros:
  ```
  sudo pacman -S tk
  ```

## Usage

```bash
python interfaz/gui.py
```

Adjust parameters in the left panel and click **SIMULATE**.

## Methods

| System | Order 1 | Order 2 |
|---|---|---|
| Classical | Symplectic Euler | Størmer–Verlet |
| Quantum | Lie–Trotter | Strang |

Both types preserve the underlying geometric structure, which keeps the energy error bounded with no secular drift. In the classical case, the symplectic form is conserved, and in the quantum case, the state remains unitary.

## References

1. E. Hairer, C. Lubich and G. Wanner, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*, 2nd ed., Springer, 2006.
2. J. M. Sanz-Serna and M. P. Calvo, *Numerical Hamiltonian Problems*, Chapman & Hall, 1994.
3. H. F. Trotter, "On the product of semi-groups of operators," *Proc. Am. Math. Soc.* **10**, 545 (1959).
4. G. Strang, "On the construction and comparison of difference schemes," *SIAM J. Numer. Anal.* **5**, 506 (1968).
5. H. Yoshida, "Construction of higher order symplectic integrators," *Phys. Lett. A* **150**, 262 (1990).
6. B. Leimkuhler and S. Reich, *Simulating Hamiltonian Dynamics*, Cambridge University Press, 2004.
7. V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed., Springer, 1989.

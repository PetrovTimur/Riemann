Here is the formulation of the **Second Phase (Characteristic Phase)** of the CABARET scheme as proposed in the paper, specifically focusing on the novel **Locally Implicit Sonic Point Processing Algorithm**.

This algorithm replaces the standard extrapolation step when a sonic point (transonic flow) is detected in a cell.

### **Context and Inputs**
*   **Current State:** You are at a spatial node $i$ transitioning from time $n$ to $n+1$.
*   **Grid:** $h_{i+1/2} = x_{i+1} - x_i$. Time step $\tau_n$.
*   **Inputs (from Phase 1):**
    *   Riemann Invariants at cell centers at the intermediate time step ($n+1/2$): $(I_k)_{i-1/2}^{n+1/2}$ and $(I_k)_{i+1/2}^{n+1/2}$.
    *   Riemann Invariants at node $i$ at the previous time step ($n$): $(I_k)_{i}^{n}$.
    *   Eigenvalues at cell centers: $(\lambda_k)_{i\pm 1/2}^{n+1/2}$.
*   **Definitions:**
    *   $c = \sqrt{gH}$ (celerity/speed of sound).
    *   $\lambda_1 = u + c$, $\lambda_2 = u - c$.
    *   $I_1 = u + 2c$, $I_2 = u - 2c$.

---

### **Step 1: Sonic Point Detection**
For each invariant $k \in \{1, 2\}$ at node $i$, check for a sonic transition:
$$ \text{Sonic}_k = \left[ (\lambda_k)^{n+1/2}_{i-1/2} \cdot (\lambda_k)^{n+1/2}_{i+1/2} \leq 0 \right] $$

*   **If $\text{Sonic}_k$ is False:** Use the **Standard Update** (Step 2).
*   **If $\text{Sonic}_k$ is True:** Use the **Implicit Update** (Step 3).

---

### **Step 2: Standard Update (Non-Sonic)**
If the flow is strictly subsonic or supersonic for invariant $k$:
Determine the upwind direction based on the sign of the eigenvalue and extrapolate:
$$
(\tilde{I}_k)_i^{n+1} =
\begin{cases}
2(I_k)_{i-1/2}^{n+1/2} - (I_k)_{i-1}^n & \text{if } (\lambda_k)_{i-1/2}^{n+1/2} > 0 \\
2(I_k)_{i+1/2}^{n+1/2} - (I_k)_{i+1}^n & \text{if } (\lambda_k)_{i+1/2}^{n+1/2} < 0
\end{cases}
$$
*(Proceed to Step 4 for monotonization).*

---

### **Step 3: Locally Implicit Sonic Point Update**
If a sonic point is detected, we solve a system of non-linear equations to find the provisional values $\tilde{u}_i^{n+1}$ and $\tilde{c}_i^{n+1}$.

#### **3.1 The Implicit Equations**
The paper derives two residual equations, $\mathcal{F}_1$ (for $I_1$) and $\mathcal{F}_2$ (for $I_2$).
Let the unknowns be $u = \tilde{u}_i^{n+1}$ and $c = \tilde{c}_i^{n+1}$.
Let the estimated eigenvalue at the next step be $\tilde{\lambda}_1 = u + c$ and $\tilde{\lambda}_2 = u - c$.

**For Invariant 1 ($\mathcal{F}_1 = 0$):**
Define local coordinate $\bar{\xi}_1 = 0.5 h_{i-1/2} - 0.5 \tau_n (u + c)$.
$$
\mathcal{F}_1(u, c) = (u + 2c) - \left[ \tau_n (I_1)_{i-1/2}^{n+1/2} \frac{A_1 \cdot B_1}{4 h_i \bar{\xi}_1} - \tau_n (I_1)_{i+1/2}^{n+1/2} \frac{C_1 \cdot D_1}{4 h_i (h_i - \bar{\xi}_1)} + (I_1)_i^n \frac{C_1 \cdot A_1}{4 \bar{\xi}_1 (h_i - \bar{\xi}_1)} \right]
$$
Where terms $A, B, C, D$ are derived from Eq (15):
*   $A_1 = h_{i+1/2} + \tau_n(u + c + \lambda_1^n)$
*   $B_1 = h_{i+1/2} + \tau_n(u + c)$
*   $C_1 = h_{i-1/2} - \tau_n(u + c)$
*   $D_1 = h_{i-1/2} - \tau_n(u + c + \lambda_1^n)$
*   $\lambda_1^n = u_i^n + \sqrt{gH_i^n}$

**For Invariant 2 ($\mathcal{F}_2 = 0$):**
Define local coordinate $\bar{\xi}_2 = 0.5 h_{i-1/2} + 0.5 \tau_n (u - c)$. *Note the sign change in definition relative to $\xi_1$ in text, but derived from Eq 16.*
$$
\mathcal{F}_2(u, c) = (u - 2c) - \left[ \tau_n (I_2)_{i-1/2}^{n+1/2} \frac{A_2 \cdot B_2}{4 h_i \bar{\xi}_2} - \tau_n (I_2)_{i+1/2}^{n+1/2} \frac{C_2 \cdot D_2}{4 h_i (h_i - \bar{\xi}_2)} + (I_2)_i^n \frac{C_2 \cdot A_2}{4 \bar{\xi}_2 (h_i - \bar{\xi}_2)} \right]
$$
Where:
*   $A_2 = h_{i+1/2} + \tau_n(u - c + \lambda_2^n)$
*   $B_2 = h_{i+1/2} + \tau_n(u - c)$
*   $C_2 = h_{i-1/2} - \tau_n(u - c)$
*   $D_2 = h_{i-1/2} - \tau_n(u - c + \lambda_2^n)$
*   $\lambda_2^n = u_i^n - \sqrt{gH_i^n}$

#### **3.2 Solving Strategy (Three Cases)**
Identify which invariants are experiencing a sonic point and solve accordingly.

*   **Case A: Sonic only in $I_1$**
    1.  Calculate $(I_2)_i^{n+1}$ using the **Standard Update** (Step 2).
    2.  Substitute $I_2 = u - 2c \implies u = I_2 + 2c$ into equation $\mathcal{F}_1$.
    3.  Solve $\mathcal{F}_1(I_2 + 2c, c) = 0$ for unknown $c$.
    4.  Recover $u = I_2 + 2c$.

*   **Case B: Sonic only in $I_2$**
    1.  Calculate $(I_1)_i^{n+1}$ using the **Standard Update** (Step 2).
    2.  Substitute $I_1 = u + 2c \implies u = I_1 - 2c$ into equation $\mathcal{F}_2$.
    3.  Solve $\mathcal{F}_2(I_1 - 2c, c) = 0$ for unknown $c$.
    4.  Recover $u = I_1 - 2c$.

*   **Case C: Sonic in both $I_1$ and $I_2$**
    1.  Solve the system $\begin{cases} \mathcal{F}_1(u, c) = 0 \\ \mathcal{F}_2(u, c) = 0 \end{cases}$ simultaneously for unknowns $u$ and $c$.

*   **Solver Recommendation:** Use Newton's Method.
*   **Initial Guess:** Use the explicit average (Eq 11) as the starting point $(u^{(0)}, c^{(0)})$ for the solver:
    $$ (\tilde{I}_k)^{(0)} = 0.5 \left[ (I_k)_{i-1/2}^{n+1/2} + (I_k)_{i+1/2}^{n+1/2} \right] $$
    Calculate $u^{(0)}, c^{(0)}$ from these averaged invariants.

#### **3.3 Output of Step 3**
You now have the un-monotonized implicit values $\tilde{u}_i^{n+1}$ and $\tilde{c}_i^{n+1}$.
Calculate the un-monotonized invariants:
$$ (\tilde{I}_{1,2})_i^{n+1} = \tilde{u}_i^{n+1} \pm 2\tilde{c}_i^{n+1} $$

---

### **Step 4: Non-Linear Correction (Monotonization)**
This step is applied to **both** Standard and Sonic updates to prevent oscillations (Eq 19).

For each invariant $k \in \{1, 2\}$, clamp the provisional value $(\tilde{I}_k)_i^{n+1}$ between the min and max of the surrounding cell centers:

1.  Find bounds:
    $$ \min_k = \min \left( (I_k)_{i-1/2}^{n+1/2}, (I_k)_{i}^{n}, (I_k)_{i+1/2}^{n+1/2} \right) $$
    $$ \max_k = \max \left( (I_k)_{i-1/2}^{n+1/2}, (I_k)_{i}^{n}, (I_k)_{i+1/2}^{n+1/2} \right) $$

2.  Apply correction:
    $$
    (I_k)_i^{n+1} = \begin{cases}
    \min_k & \text{if } (\tilde{I}_k)_i^{n+1} < \min_k \\
    \max_k & \text{if } (\tilde{I}_k)_i^{n+1} > \max_k \\
    (\tilde{I}_k)_i^{n+1} & \text{otherwise}
    \end{cases}
    $$

---

### **Step 5: Final Flux Variable Calculation**
Reconstruct the physical flux variables at the node for the next time step (Eq 6):

$$ H_i^{n+1} = \frac{1}{g} \left( \frac{(I_1)_i^{n+1} - (I_2)_i^{n+1}}{4} \right)^2 $$
$$ u_i^{n+1} = \frac{(I_1)_i^{n+1} + (I_2)_i^{n+1}}{2} $$
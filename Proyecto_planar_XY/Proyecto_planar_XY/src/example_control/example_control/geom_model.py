#!/usr/bin/env python3
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import threading

class RobotKinematics():
  def __init__(self):
    pass

  def direct_kinematics(self):
    print("Definiendo variables del modelo en sympy (plano XY)")
    self.theta_0_1, self.theta_1_2, self.theta_2_3 = symbols("theta_0_1, theta_1_2, theta_2_3")
    self.l1 = 0.3; self.l2 = 0.3; self.l3 = 0.3

    # Plano XY: Rz + Tx
    self.T_0_1 = self.trans_homo_xy(0, 0, self.theta_0_1)
    self.T_1_2 = self.trans_homo_xy(self.l1, 0, self.theta_1_2)
    self.T_2_3 = self.trans_homo_xy(self.l2, 0, self.theta_2_3)
    self.T_3_p = self.trans_homo_xy(self.l3, 0, 0)

    T_0_p = simplify(self.T_0_1 * self.T_1_2 * self.T_2_3 * self.T_3_p)

    x_0_p = T_0_p[0, 3]
    y_0_p = T_0_p[1, 3]
    th_0_p = self.theta_0_1 + self.theta_1_2 + self.theta_2_3

    self.xi_0_p = Matrix([[x_0_p],[y_0_p],[th_0_p]])

    self.J = Matrix.hstack(
      diff(self.xi_0_p, self.theta_0_1),
      diff(self.xi_0_p, self.theta_1_2),
      diff(self.xi_0_p, self.theta_2_3)
    )

    self.x_dot, self.y_dot, self.th_dot = symbols("x_dot, y_dot, th_dot")
    self.xi_dot = Matrix([[self.x_dot],[self.y_dot],[self.th_dot]])

    print("Definidas todas las variables (XY)")

  def trajectory_generator(self, q_in=[0.1,0.1,0.1], xi_fn=[0.8,0.1,0], duration=4):
    self.freq = 30
    print("Definiendo trayectoria")

    self.t, a0, a1, a2, a3, a4, a5 = symbols("t, a0, a1, a2, a3, a4, a5")
    self.lam = a0 + a1*self.t + a2*self.t**2 + a3*self.t**3 + a4*self.t**4 + a5*self.t**5
    self.lam_dot = diff(self.lam, self.t)
    self.lam_dot_dot = diff(self.lam_dot, self.t)

    T = duration
    ec1 = self.lam.subs(self.t, 0)
    ec2 = self.lam.subs(self.t, T) - 1
    ec3 = self.lam_dot.subs(self.t, 0)
    ec4 = self.lam_dot.subs(self.t, T)
    ec5 = self.lam_dot_dot.subs(self.t, 0)
    ec6 = self.lam_dot_dot.subs(self.t, T)

    terms = solve([ec1,ec2,ec3,ec4,ec5,ec6],[a0,a1,a2,a3,a4,a5], dict=True)
    self.lam_s         = self.lam.subs(terms[0])
    self.lam_dot_s     = self.lam_dot.subs(terms[0])
    self.lam_dot_dot_s = self.lam_dot_dot.subs(terms[0])

    xi_in = self.xi_0_p.subs({
      self.theta_0_1: q_in[0],
      self.theta_1_2: q_in[1],
      self.theta_2_3: q_in[2]
    })

    xi = xi_in + Matrix([
      [self.lam_s * (xi_fn[0] - xi_in[0])],
      [self.lam_s * (xi_fn[1] - xi_in[1])],
      [self.lam_s * (xi_fn[2] - xi_in[2])]
    ])
    xi_dot = Matrix([
      [self.lam_dot_s * (xi_fn[0] - xi_in[0])],
      [self.lam_dot_s * (xi_fn[1] - xi_in[1])],
      [self.lam_dot_s * (xi_fn[2] - xi_in[2])]
    ])
    xi_dot_dot = Matrix([
      [self.lam_dot_dot_s * (xi_fn[0] - xi_in[0])],
      [self.lam_dot_dot_s * (xi_fn[1] - xi_in[1])],
      [self.lam_dot_dot_s * (xi_fn[2] - xi_in[2])]
    ])

    self.samples = int(self.freq * duration + 1)
    self.dt = 1/self.freq

    self.xi_m         = Matrix.zeros(3, self.samples)
    self.xi_dot_m     = Matrix.zeros(3, self.samples)
    self.xi_dot_dot_m = Matrix.zeros(3, self.samples)
    self.t_m = Matrix.zeros(1, self.samples)

    self.t_m[0,0] = 0
    for a in range(self.samples - 1):
      self.t_m[0, a+1] = self.t_m[0, a] + self.dt

    xi_lam         = lambdify([self.t], xi,         modules='numpy')
    xi_dot_lam     = lambdify([self.t], xi_dot,     modules='numpy')
    xi_dot_dot_lam = lambdify([self.t], xi_dot_dot, modules='numpy')

    for a in range(self.samples):
      self.xi_m[:, a]         = xi_lam(float(self.t_m[0,a]))
      self.xi_dot_m[:, a]     = xi_dot_lam(float(self.t_m[0,a]))
      self.xi_dot_dot_m[:, a] = xi_dot_dot_lam(float(self.t_m[0,a]))

    self.q_in = q_in

  def inverse_kinematics(self):
    print("Modelando cinemática inversa (XY, pseudoinversa)")
    J_lam = lambdify([self.theta_0_1, self.theta_1_2, self.theta_2_3], self.J, modules='numpy')

    self.q_m         = Matrix.zeros(3, self.samples)
    self.q_dot_m     = Matrix.zeros(3, self.samples)
    self.q_dot_dot_m = Matrix.zeros(3, self.samples)

    self.q_m[:, 0]         = Matrix([[self.q_in[0]], [self.q_in[1]], [self.q_in[2]]])
    self.q_dot_m[:, 0]     = Matrix.zeros(3, 1)
    self.q_dot_dot_m[:, 0] = Matrix.zeros(3, 1)

    for k in range(self.samples - 1):
      th1 = float(self.q_m[0, k]); th2 = float(self.q_m[1, k]); th3 = float(self.q_m[2, k])
      Jk = np.array(J_lam(th1, th2, th3), dtype=float)

      lam = 1e-4
      JTJ = Jk.T @ Jk
      Jpinv = np.linalg.solve(JTJ + (lam**2)*np.eye(3), Jk.T)

      xi_dot_k = np.array(self.xi_dot_m[:, k], dtype=float).reshape(3,1)
      q_dot_k  = Jpinv @ xi_dot_k

      self.q_dot_m[:, k+1] = Matrix(q_dot_k)
      self.q_m[:, k+1]     = self.q_m[:, k] + self.q_dot_m[:, k+1] * self.dt
      self.q_dot_dot_m[:, k+1] = (self.q_dot_m[:, k+1] - self.q_dot_m[:, k]) / self.dt

    print("Trayectoria de las juntas generada")

  # ====== GRÁFICAS ======
  def _np_time(self):
    return np.array(self.t_m, dtype=float).ravel()

  def ws_graph(self):
    t = self._np_time()
    x  = np.array(self.xi_m[0, :], dtype=float).ravel()
    y  = np.array(self.xi_m[1, :], dtype=float).ravel()
    th = np.array(self.xi_m[2, :], dtype=float).ravel()

    fig, (p_g, v_g, a_g) = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    fig.suptitle("Espacio de trabajo")

    p_g.set_title("Posiciones")
    p_g.plot(t, x, color="#1f77b4", linewidth=2, label="x"); 
    p_g.plot(t, y, color="#ff7f0e", linewidth=2, label="y"); 
    p_g.plot(t, th, color="#2ca02c", linewidth=2, label="θ"); 
    p_g.legend(loc="best"); p_g.grid(True)

    vx  = np.array(self.xi_dot_m[0, :], dtype=float).ravel()
    vy  = np.array(self.xi_dot_m[1, :], dtype=float).ravel()
    vth = np.array(self.xi_dot_m[2, :], dtype=float).ravel()

    v_g.set_title("Velocidades")
    v_g.plot(t, vx,  color="#1f77b4"); v_g.plot(t, vy,  color="#ff7f0e"); v_g.plot(t, vth, color="#2ca02c")
    v_g.grid(True)

    ax  = np.array(self.xi_dot_dot_m[0, :], dtype=float).ravel()
    ay  = np.array(self.xi_dot_dot_m[1, :], dtype=float).ravel()
    ath = np.array(self.xi_dot_dot_m[2, :], dtype=float).ravel()

    a_g.set_title("Aceleraciones")
    a_g.plot(t, ax,  color="#1f77b4"); a_g.plot(t, ay,  color="#ff7f0e"); a_g.plot(t, ath, color="#2ca02c")
    a_g.grid(True)

    plt.tight_layout()
    plt.show()

  def q_graph(self):
    t = self._np_time()
    q1  = np.array(self.q_m[0, :], dtype=float).ravel()
    q2  = np.array(self.q_m[1, :], dtype=float).ravel()
    q3  = np.array(self.q_m[2, :], dtype=float).ravel()
    qd1 = np.array(self.q_dot_m[0, :], dtype=float).ravel()
    qd2 = np.array(self.q_dot_m[1, :], dtype=float).ravel()
    qd3 = np.array(self.q_dot_m[2, :], dtype=float).ravel()
    qdd1= np.array(self.q_dot_dot_m[0, :], dtype=float).ravel()
    qdd2= np.array(self.q_dot_dot_m[1, :], dtype=float).ravel()
    qdd3= np.array(self.q_dot_dot_m[2, :], dtype=float).ravel()

    fig, (p_g, v_g, a_g) = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    fig.suptitle("Espacio de las juntas")

    p_g.set_title("Posiciones")
    p_g.plot(t, q1, color="#8E5572", linewidth=2, label="q1"); 
    p_g.plot(t, q2, color="#5C80BC", linewidth=2, label="q2");
    p_g.plot(t, q3, color="#E55934", linewidth=2, label="q3");
    p_g.legend(loc="best"); p_g.grid(True)

    v_g.set_title("Velocidades")
    v_g.plot(t, qd1, color="#8E5572"); v_g.plot(t, qd2, color="#5C80BC"); v_g.plot(t, qd3, color="#E55934")
    v_g.grid(True)

    a_g.set_title("Aceleraciones")
    a_g.plot(t, qdd1, color="#8E5572"); a_g.plot(t, qdd2, color="#5C80BC"); a_g.plot(t, qdd3, color="#E55934")
    a_g.grid(True)

    plt.tight_layout()
    plt.show()

  def render_plots_async(self):


    pass

  def simple_graph(self, val_m, t_m):
    t = np.array(t_m, dtype=float).ravel()
    v = np.array(val_m[0, :], dtype=float).ravel()
    plt.figure(); plt.plot(t, v); plt.grid(True); plt.tight_layout()

  def trans_homo_xy(self, x=0, y=0, alpha=0)->Matrix:
    Rz = Matrix([[cos(alpha), -sin(alpha), 0],
                 [sin(alpha),  cos(alpha), 0],
                 [0,           0,          1]])
    p  = Matrix([[x],[y],[0]])
    T  = Matrix.vstack(Matrix.hstack(Rz, p), Matrix([[0,0,0,1]]))
    return T

  def trans_homo(self, x, y, z, gamma, beta, alpha):
    R_z = Matrix([[cos(alpha), -sin(alpha), 0],
                  [sin(alpha),  cos(alpha), 0],
                  [0,           0,          1]])
    R_y = Matrix([[cos(beta), 0, sin(beta)],
                  [0,         1, 0        ],
                  [-sin(beta),0, cos(beta)]])
    R_x = Matrix([[1, 0, 0],
                  [0, cos(gamma), -sin(gamma)],
                  [0, sin(gamma),  cos(gamma)]])
    p_x = Matrix([[x],[0],[0]])
    p_y = Matrix([[0],[y],[0]])
    p_z = Matrix([[0],[0],[z]])
    T_x = Matrix.vstack(Matrix.hstack(R_x, p_x), Matrix([[0,0,0,1]]))
    T_y = Matrix.vstack(Matrix.hstack(R_y, p_y), Matrix([[0,0,0,1]]))
    T_z = Matrix.vstack(Matrix.hstack(R_z, p_z), Matrix([[0,0,0,1]]))
    return T_x * T_y * T_z

  def fk(self, q):
    xi = self.xi_0_p.subs({
      self.theta_0_1: float(q[0]),
      self.theta_1_2: float(q[1]),
      self.theta_2_3: float(q[2]),
    })
    return float(xi[0]), float(xi[1]), float(xi[2])

  def redirect_print(self, new_print):
    global print
    print = new_print

def main():
  robot = RobotKinematics()
  robot.direct_kinematics()
  robot.trajectory_generator()
  robot.inverse_kinematics()
  robot.ws_graph()
  robot.q_graph()
  robot.render_plots_async()

if __name__ == "__main__":
  main()

def ws_graph(self):
    import numpy as np
    import matplotlib.pyplot as plt
    t = self._np_time()
    x  = np.array(self.xi_m[0, :], dtype=float).ravel()
    y  = np.array(self.xi_m[1, :], dtype=float).ravel()
    th = np.array(self.xi_m[2, :], dtype=float).ravel()

    fig, (p_g, v_g, a_g) = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    fig.suptitle("Espacio de trabajo")

    p_g.set_title("Posiciones")
    p_g.plot(t, x,  color="#1f77b4", linewidth=2, label="x")
    p_g.plot(t, y,  color="#ff7f0e", linewidth=2, label="y")
    p_g.plot(t, th, color="#2ca02c", linewidth=2, label="θ")
    p_g.legend(loc="best"); p_g.grid(True)

    vx  = np.array(self.xi_dot_m[0, :], dtype=float).ravel()
    vy  = np.array(self.xi_dot_m[1, :], dtype=float).ravel()
    vth = np.array(self.xi_dot_m[2, :], dtype=float).ravel()

    v_g.set_title("Velocidades")
    v_g.plot(t, vx,  color="#1f77b4")
    v_g.plot(t, vy,  color="#ff7f0e")
    v_g.plot(t, vth, color="#2ca02c")
    v_g.grid(True)

    ax  = np.array(self.xi_dot_dot_m[0, :], dtype=float).ravel()
    ay  = np.array(self.xi_dot_dot_m[1, :], dtype=float).ravel()
    ath = np.array(self.xi_dot_dot_m[2, :], dtype=float).ravel()

    a_g.set_title("Aceleraciones")
    a_g.plot(t, ax,  color="#1f77b4")
    a_g.plot(t, ay,  color="#ff7f0e")
    a_g.plot(t, ath, color="#2ca02c")
    a_g.grid(True)

    plt.tight_layout()
    plt.show()

def q_graph(self):
    import numpy as np
    import matplotlib.pyplot as plt
    t = self._np_time()
    q1   = np.array(self.q_m[0, :], dtype=float).ravel()
    q2   = np.array(self.q_m[1, :], dtype=float).ravel()
    q3   = np.array(self.q_m[2, :], dtype=float).ravel()
    qd1  = np.array(self.q_dot_m[0, :], dtype=float).ravel()
    qd2  = np.array(self.q_dot_m[1, :], dtype=float).ravel()
    qd3  = np.array(self.q_dot_m[2, :], dtype=float).ravel()
    qdd1 = np.array(self.q_dot_dot_m[0, :], dtype=float).ravel()
    qdd2 = np.array(self.q_dot_dot_m[1, :], dtype=float).ravel()
    qdd3 = np.array(self.q_dot_dot_m[2, :], dtype=float).ravel()

    fig, (p_g, v_g, a_g) = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    fig.suptitle("Espacio de las juntas")

    p_g.set_title("Posiciones")
    p_g.plot(t, q1, color="#8E5572", linewidth=2, label="q1")
    p_g.plot(t, q2, color="#5C80BC", linewidth=2, label="q2")
    p_g.plot(t, q3, color="#E55934", linewidth=2, label="q3")
    p_g.legend(loc="best"); p_g.grid(True)

    v_g.set_title("Velocidades")
    v_g.plot(t, qd1, color="#8E5572")
    v_g.plot(t, qd2, color="#5C80BC")
    v_g.plot(t, qd3, color="#E55934")
    v_g.grid(True)

    a_g.set_title("Aceleraciones")
    a_g.plot(t, qdd1, color="#8E5572")
    a_g.plot(t, qdd2, color="#5C80BC")
    a_g.plot(t, qdd3, color="#E55934")
    a_g.grid(True)

    plt.tight_layout()
    plt.show()


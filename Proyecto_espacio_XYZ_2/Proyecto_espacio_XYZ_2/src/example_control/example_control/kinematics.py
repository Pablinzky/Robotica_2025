#!/usr/bin/env python3
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

class RobotKinematics():
  def __init__(self):
    pass


  def trans_homo_xz(self, x=0, z=0, gamma=0, alpha=0) -> Matrix:

    Rz = Matrix([[cos(alpha), -sin(alpha), 0],
                 [sin(alpha),  cos(alpha), 0],
                 [0,           0,          1]])
    Rx = Matrix([[1, 0, 0],
                 [0, cos(gamma), -sin(gamma)],
                 [0, sin(gamma),  cos(gamma)]])
    R = Rx * Rz
    p = Rx * Matrix([[0], [0], [z]]) + Matrix([[x], [0], [0]])
    T = Matrix([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]])
    T[:3, :3] = R
    T[:3,  3] = p
    return T
  
  def trans_homo(self, x, y, z, gamma, beta, alpha):

    Rz = Matrix([[cos(alpha), -sin(alpha), 0],
                 [sin(alpha),  cos(alpha), 0],
                 [0,           0,          1]])
    Ry = Matrix([[ cos(beta), 0, sin(beta)],
                 [ 0,         1, 0        ],
                 [-sin(beta), 0, cos(beta)]])
    Rx = Matrix([[1, 0, 0],
                 [0, cos(gamma), -sin(gamma)],
                 [0, sin(gamma),  cos(gamma)]])
    R = Rx * Ry * Rz
    p = Rx * Ry * Matrix([[0], [0], [z]]) + Rx * Matrix([[0], [y], [0]]) + Matrix([[x], [0], [0]])
    T = Matrix([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]])
    T[:3, :3] = R
    T[:3,  3] = p
    return T

  def T_trans(self, x=0, y=0, z=0):
    return self.trans_homo(x, y, z, 0, 0, 0)

  def R_x(self, roll):
    return self.trans_homo(0, 0, 0, roll, 0, 0)

  def R_y(self, pitch):
    return self.trans_homo(0, 0, 0, 0, pitch, 0)

  def R_z(self, yaw):
    return self.trans_homo(0, 0, 0, 0, 0, yaw)

  def trajectory_generator(self, q_in = [0.1, 0.1, 0.1], xi_fn = [0.8, 0.1, 0.0], duration = 4):
    self.freq = 30
    print("Definiendo trayectoria")

    self.t, a0, a1, a2, a3, a4, a5 = symbols("t, a0, a1, a2, a3, a4, a5")
    self.lam = a0 + a1*self.t + a2*self.t**2 + a3*self.t**3 + a4*self.t**4 + a5*self.t**5
    self.lam_dot = diff(self.lam, self.t)
    self.lam_dot_dot = diff(self.lam_dot, self.t)

    ec1 = self.lam.subs(self.t, 0)
    ec2 = self.lam.subs(self.t, duration) - 1
    ec3 = self.lam_dot.subs(self.t, 0)
    ec4 = self.lam_dot.subs(self.t, duration)
    ec5 = self.lam_dot_dot.subs(self.t, 0)
    ec6 = self.lam_dot_dot.subs(self.t, duration)
    terms = solve([ec1, ec2, ec3, ec4, ec5, ec6], [a0, a1, a2, a3, a4, a5], dict=True)

    self.lam_s          = self.lam.subs(terms[0])
    self.lam_dot_s      = self.lam_dot.subs(terms[0])
    self.lam_dot_dot_s  = self.lam_dot_dot.subs(terms[0])

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
    self.t_m          = Matrix.zeros(1, self.samples)

    self.t_m[0, 0] = 0
    for a in range(self.samples - 1):
      self.t_m[0, a+1] = self.t_m[0, a] + self.dt

    xi_lam         = lambdify([self.t], xi,         modules='numpy')
    xi_dot_lam     = lambdify([self.t], xi_dot,     modules='numpy')
    xi_dot_dot_lam = lambdify([self.t], xi_dot_dot, modules='numpy')

    for a in range(self.samples):
      self.xi_m[:, a]         = xi_lam(float(self.t_m[0, a]))
      self.xi_dot_m[:, a]     = xi_dot_lam(float(self.t_m[0, a]))
      self.xi_dot_dot_m[:, a] = xi_dot_dot_lam(float(self.t_m[0, a]))

    self.q_in = q_in

  def _qdot_num(self, xd, yd, zd, q1, q2, q3):
    Jn = np.array(self.J_fun(q1, q2, q3), dtype=float)
    xi = np.array([xd, yd, zd], dtype=float).reshape(3, 1)
    qdot, *_ = np.linalg.lstsq(Jn, xi, rcond=None)
    return Matrix(qdot)

  def inverse_kinematics(self):
    print("Modelando cinemática inversa")

    self.q_m         = Matrix.zeros(3, self.samples)
    self.q_dot_m     = Matrix.zeros(3, self.samples)
    self.q_dot_dot_m = Matrix.zeros(3, self.samples)

    self.q_m[:, 0]         = Matrix([[self.q_in[0]], [self.q_in[1]], [self.q_in[2]]])
    self.q_dot_m[:, 0]     = Matrix.zeros(3, 1)
    self.q_dot_dot_m[:, 0] = Matrix.zeros(3, 1)

    self.q_dot_lam = self._qdot_num

    for a in range(self.samples - 1):
      self.q_m[:, a+1] = self.q_m[:, a] + self.q_dot_m[:, a] * self.dt
      self.q_dot_m[:, a+1] = self.q_dot_lam(float(self.xi_dot_m[0, a]),
                                            float(self.xi_dot_m[1, a]),
                                            float(self.xi_dot_m[2, a]),
                                            float(self.q_m[0, a]),
                                            float(self.q_m[1, a]),
                                            float(self.q_m[2, a]))
      self.q_dot_dot_m[:, a+1] = (self.q_dot_m[:, a+1] - self.q_dot_m[:, a]) / self.dt

    print("Trayectoria de las juntas generada")

  def ws_graph(self):
    fig, (p_g, v_g, a_g) = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("Espacio de trabajo (x,y,z)")
    p_g.set_title("Posiciones")
    p_g.plot(self.t_m.T, self.xi_m[0, :].T, color = "RED")
    p_g.plot(self.t_m.T, self.xi_m[1, :].T, color = (0, 1, 0))
    p_g.plot(self.t_m.T, self.xi_m[2, :].T, color = "blue")
    v_g.set_title("Velocidades")
    v_g.plot(self.t_m.T, self.xi_dot_m[0, :].T, color = "RED")
    v_g.plot(self.t_m.T, self.xi_dot_m[1, :].T, color = (0, 1, 0))
    v_g.plot(self.t_m.T, self.xi_dot_m[2, :].T, color = "blue")
    a_g.set_title("Aceleraciones")
    a_g.plot(self.t_m.T, self.xi_dot_dot_m[0, :].T, color = "RED")
    a_g.plot(self.t_m.T, self.xi_dot_dot_m[1, :].T, color = (0, 1, 0))
    a_g.plot(self.t_m.T, self.xi_dot_dot_m[2, :].T, color = "blue")
    plt.show()

  def q_graph(self):
    fig, (p_g, v_g, a_g) = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("Espacio de las juntas (q1,q2,q3)")
    p_g.set_title("Posiciones")
    p_g.plot(self.t_m.T, self.q_m[0, :].T, color = "RED")
    p_g.plot(self.t_m.T, self.q_m[1, :].T, color = (0, 1, 0))
    p_g.plot(self.t_m.T, self.q_m[2, :].T, color = "blue")
    v_g.set_title("Velocidades")
    v_g.plot(self.t_m.T, self.q_dot_m[0, :].T, color = "RED")
    v_g.plot(self.t_m.T, self.q_dot_m[1, :].T, color = (0, 1, 0))
    v_g.plot(self.t_m.T, self.q_dot_m[2, :].T, color = "blue")
    a_g.set_title("Aceleraciones")
    a_g.plot(self.t_m.T, self.q_dot_dot_m[0, :].T, color = "RED")
    a_g.plot(self.t_m.T, self.q_dot_dot_m[1, :].T, color = (0, 1, 0))
    a_g.plot(self.t_m.T, self.q_dot_dot_m[2, :].T, color = "blue")
    plt.show()

  def direct_kinematics(self):
    print("Definiendo variables del modelo en sympy")

    self.theta_0_1, self.theta_1_2, self.theta_2_3 = symbols("theta_0_1, theta_1_2, theta_2_3")

    L_MAG = Rational(1, 5)        
    L_FOR = Float("0.35")          

    q_base = self.theta_0_1
    q_sh   = self.theta_1_2
    q_el   = self.theta_2_3

    R01 = self.R_z(q_base)[:3, :3]
    p01 = Matrix([[0], [0], [Float("0.10")]])
    T_0_1 = Matrix.vstack(Matrix.hstack(R01, p01), Matrix([[0, 0, 0, 1]]))

    R12 = (self.R_x(pi) * self.R_y(q_sh))[:3, :3]
    p12 = Matrix([[0], [0], [Float("0.05")]])
    T_1_2 = Matrix.vstack(Matrix.hstack(R12, p12), Matrix([[0, 0, 0, 1]]))

    R23 = (self.R_x(pi) * self.R_y(-q_el))[:3, :3]
    p23 = Matrix([[L_MAG], [0], [0]])
    T_2_3 = Matrix.vstack(Matrix.hstack(R23, p23), Matrix([[0, 0, 0, 1]]))

    R3p = Matrix([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    p3p = Matrix([[L_FOR], [0], [0]])
    T_3_p = Matrix.vstack(Matrix.hstack(R3p, p3p), Matrix([[0, 0, 0, 1]]))

    T_0_p = simplify(T_0_1 * T_1_2 * T_2_3 * T_3_p)

    x_0_p = T_0_p[0, 3];  y_0_p = T_0_p[1, 3];  z_0_p = T_0_p[2, 3]
    self.xi_0_p = Matrix([x_0_p, y_0_p, z_0_p]).reshape(3, 1)

    self.J = Matrix.hstack(
      diff(self.xi_0_p, self.theta_0_1),
      diff(self.xi_0_p, self.theta_1_2),
      diff(self.xi_0_p, self.theta_2_3)
    )
    self.fk_pos = lambdify([self.theta_0_1, self.theta_1_2, self.theta_2_3], self.xi_0_p, 'numpy')
    self.J_fun  = lambdify([self.theta_0_1, self.theta_1_2, self.theta_2_3], self.J,      'numpy')
    print("Definidas todas las variables")

  def simple_graph(self, val_m, t_m):
    plt.plot(t_m.T, val_m[0, :].T)
    plt.show()

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
  def fk(self):
      """Alias de conveniencia que llama a direct_kinematics sin modificar lógica."""
      return self.direct_kinematics()
if __name__ == "__main__":
  main()

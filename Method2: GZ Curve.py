
import matplotlib.pyplot as plt
import numpy as np

theta_list = np.linspace(-3.14/4,3.14/4,100)

def find_h(theta,rho):
    # The rectangle height
    t = np.tan(theta)
    c = np.cos(theta)
    s = np.sin(theta)

    x = rho - 0.5 * t
    v = [-s, c]
    u = [c, s]

    if rho < 0.5 and abs(t)-2*rho < 0.001: # For rho<1/2 and has trapezium shape, t<2rho
        Coord_1 = [0, x / 2] # the bottom rectangle
        Coord_2 = [1/ 6, rho-t/6] # the upper triangle
        A1 = x
        A2 = t/2
        r = A1 / (A1+A2)
        Coord_w = [Coord_1[0]*r + (1-r) * (Coord_2[0]), Coord_1[1]*r + (1-r) * (Coord_2[1])]
    elif rho < 0.5 and abs(t)-2*rho > 0.001: # For rho<1/2 and has triangle shape
        a = np.sqrt(abs(2*rho*t))
        b = np.sqrt(abs((2*rho)/t))
        Coord_w = [t*(0.5 - b/3)/abs(t), a/3]

    elif rho > 0.5 and abs(t)-2*(1-rho)< 0.001: # For rho<1/2 and has trapezium shape;SAME as case 1
        Coord_1 = [0, x / 2] # the bottom rectangle
        Coord_2 = [1 / 6, rho-t/6] # the upper triangle
        A1 = x
        A2 = t/2
        r = A1 / (A1 + A2)
        Coord_w = [Coord_1[0]*r + (1-r)*(Coord_2[0]), Coord_1[1]*r + (1-r)*(Coord_2[1])]

    elif rho > 0.5 and abs(t)-2*(1-rho)> 0.001:
        Rho = 1 - rho
        a = np.sqrt(abs(2 * Rho * t))
        b = np.sqrt(abs((2 * Rho) / t))
        Coord_1 = [0, 0.5] # The full water vessel
        Coord_2 = [t*(-0.5+(b/3))/abs(t), 1-(a/3)] # The upper air triangle
        A1 = 1
        A2 = 0.5*a*b
        r = A1/(A1+A2)
        Coord_w = [Coord_1[0]*r - Coord_2[0]*(1-r), Coord_1[1]*r - Coord_2[1]*(1-r)]

    #print(x, np.vdot(R, v))
    Coord_v = [0, 1 / 2]
    R = [(Coord_v[0] - Coord_w[0]), (Coord_v[1] - Coord_w[1])]
    return np.vdot(R, u)


h_list_1 = []
h_list_2 = []
h_list_3 = []

for theta in theta_list:
    PE_1 = find_h(theta,rho=0.2115)
    h_list_1.append(-PE_1)

for theta in theta_list:
    PE_2 = find_h(theta, rho=0.250)
    h_list_2.append(-PE_2)

for theta in theta_list:
    PE_3 = find_h(theta, rho=0.30)
    h_list_3.append(-PE_3)



xpoints = theta_list
ypoints_1 = h_list_1
ypoints_2 = h_list_2
ypoints_3 = h_list_3

plt.plot(xpoints, ypoints_1, label = 'Rho = 0.2115')
plt.title('Method2: GZ curve')
plt.ylabel('GZ Distance')
plt.xlabel('Heel Angle')
plt.legend()
plt.grid()
plt.show()
plt.plot(xpoints, ypoints_2, label = 'Rho = 0.250')
plt.title('Method2: GZ curve')
plt.ylabel('GZ Distance')
plt.xlabel('Heel Angle')
plt.legend()
plt.grid()
plt.show()
plt.plot(xpoints, ypoints_3, label = 'Rho = 0.30')
plt.title('Method2: GZ curve')
plt.ylabel('GZ Distance')
plt.xlabel('Heel Angle')
plt.legend()
plt.grid()
plt.show()
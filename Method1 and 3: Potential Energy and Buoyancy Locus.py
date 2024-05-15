
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

    Coord_v = [0, 1 / 2]
    R = [(Coord_v[0] - Coord_w[0]), (Coord_v[1] - Coord_w[1])]
    return np.vdot(R, v), Coord_w[0], Coord_w[1]


h_list_1 = []
h_list_2 = []
h_list_3 = []

wx_list_1 = []
wy_list_1 = []
wx_list_2 = []
wy_list_2 = []
wx_list_3 = []
wy_list_3 = []

# Forming list of h values and water coordinates
for theta in theta_list:
    PE_1 = find_h(theta,rho=0.15)
    h_list_1.append(PE_1[0])
    wx_list_1.append(PE_1[1])
    wy_list_1.append(PE_1[2])
for theta in theta_list:
    PE_2 = find_h(theta, rho=0.250)
    h_list_2.append(PE_2[0])
    wx_list_2.append(PE_2[1])
    wy_list_2.append(PE_2[2])
for theta in theta_list:
    PE_3 = find_h(theta, rho=0.3)
    h_list_3.append(PE_3[0])
    wx_list_3.append(PE_3[1])
    wy_list_3.append(PE_3[2])

# Finding the stable points of three potential energy curves
i = np.argmin(h_list_1)
x1_min = theta_list[i]
y1_min = h_list_1[i]
wx1 = wx_list_1[i]
wy1 = wy_list_1[i]

j = np.argmin(h_list_2)
x2_min = [theta_list[j],-theta_list[j]]
y2_min = [h_list_2[j], h_list_2[j]]
wx2 = [wx_list_2[j], -wx_list_2[j]]
wy2 = [wy_list_2[j], wy_list_2[j]]

k = np.argmin(h_list_3)
x3_min = [theta_list[k],-theta_list[k]]
y3_min = [h_list_3[k], h_list_3[k]]
wx3 = [wx_list_3[k], -wx_list_3[k]]
wy3 = [wy_list_3[k], wy_list_3[k]]


# changing names for plot, easier for understanding
xpoints = theta_list
ypoints_1 = h_list_1
ypoints_2 = h_list_2
ypoints_3 = h_list_3
# Plotting Potential Curves for three cases
plt.plot(xpoints, ypoints_1, label = 'Rho = 0.15')
plt.plot(x1_min, y1_min, marker = 'o')
plt.title('Method1: Potential Energy')
plt.ylabel('Total Potential Energy')
plt.xlabel('Heel Angle')
plt.legend()
plt.grid()
plt.show()
plt.plot(xpoints, ypoints_2, label = 'Rho = 0.250')
plt.scatter(x2_min, y2_min, marker = 'o', color='orange')
plt.title('Method1: Potential Energy')
plt.ylabel('Total Potential Energy')
plt.xlabel('Heel Angle')
plt.legend()
plt.grid()
plt.show()
plt.plot(xpoints, ypoints_3, label = 'Rho = 0.40')
plt.scatter(x3_min, y3_min, marker = 'o', color='orange')
plt.title('Method1: Potential Energy')
plt.ylabel('Total Potential Energy')
plt.xlabel('Heel Angle')
plt.legend()
plt.grid()
plt.show()
# Plotting buoyancy locus and the corresponding evolute
plt.plot(wx_list_1, wy_list_1, label = 'Rho = 0.15')
plt.scatter(wx1, wy1, marker = 'o')
k = 3
print(len(wx_list_1))
for i in range(32):
    tan = (wy_list_1[k*i+2]-wy_list_1[k*i+1])/(wx_list_1[k*i+2]-wx_list_1[k*i+1])
    print(tan)
    y_alpha_list = []
    x_alpha_list = []
    for y in range(2):
        y = y*0.5
        x_alpha = wx_list_1[k * i + 1] - y*tan
        y_alpha = wy_list_1[k*i+1] + y
        x_alpha_list.append(x_alpha)
        y_alpha_list.append(y_alpha)
    plt.plot(x_alpha_list, y_alpha_list, color = 'orange')
plt.title('CoG of water locus')
plt.xlim(-0.5, 0.5)
plt.ylim(0, 1)
plt.legend()
plt.show()

plt.plot(wx_list_2, wy_list_2, label = 'Rho = 0.25')
plt.scatter(wx2, wy2, marker = 'o')
k = 3
for i in range(32):
    tan = (wy_list_2[k*i+2]-wy_list_2[k*i+1])/(wx_list_2[k*i+2]-wx_list_2[k*i+1])
    print(tan)
    y_alpha_list = []
    x_alpha_list = []
    for y in range(2):
        y = y*0.5
        x_alpha = wx_list_2[k * i + 1] - y*tan
        y_alpha = wy_list_2[k*i+1] + y
        x_alpha_list.append(x_alpha)
        y_alpha_list.append(y_alpha)
    plt.plot(x_alpha_list, y_alpha_list, color = 'orange')
plt.title('CoG of water locus')
plt.xlim(-0.5, 0.5)
plt.ylim(0, 1)
plt.legend()
plt.show()

plt.plot(wx_list_3, wy_list_3, label = 'Rho = 0.40')
plt.scatter(wx3, wy3, marker = 'o')
k = 3
print(len(wx_list_1))
for i in range(32):
    tan = (wy_list_3[k*i+2]-wy_list_3[k*i+1])/(wx_list_3[k*i+2]-wx_list_3[k*i+1])
    print(tan)
    y_alpha_list = []
    x_alpha_list = []
    for y in range(2):
        y = y*0.5
        x_alpha = wx_list_3[k * i + 1] - y*tan
        y_alpha = wy_list_3[k*i+1] + y
        x_alpha_list.append(x_alpha)
        y_alpha_list.append(y_alpha)
    plt.plot(x_alpha_list, y_alpha_list, color = 'orange')
plt.title('CoG of water locus')
plt.xlim(-0.5, 0.5)
plt.ylim(0, 1)
plt.legend()
plt.show()
#Plotting three buoyancy locus on the graph
plt.plot(wx_list_2, wy_list_2, label = 'Rho = 0.25')
plt.plot(wx_list_1, wy_list_1, label = 'Rho = 0.15')
plt.plot(wx_list_3, wy_list_3, label = 'Rho = 0.40')
plt.title('CoG of water locus')
plt.xlim(-0.5, 0.5)
plt.ylim(0, 1)
plt.legend()
plt.show()
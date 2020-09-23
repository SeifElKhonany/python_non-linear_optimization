from math import sin, cos, exp, pi
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def expn(value):    # In case a value of an exponential tends to infinity
    try:
        ans = exp(value)
    except OverflowError:
        ans = float('inf')
    return ans


def g1(x):
    y = 3*x[0] - cos(x[1]*x[2]) - 0.5
    return y


def g2(x):
    y = x[0]**2 - (81*(x[1] + 0.1))**2 - sin(x[2]) + 1.06
    return y


def g3(x):
    y = expn(-x[0]*x[1]) + 20*x[2] +((10*pi-3)/3)
    return y


def f(x):
    y = (1/2)*((g1(x)**2)+(g2(x)**2)+(g3(x)**2))
    return y


def dg1_dx1(x):  # g First derivatives
    return 3


def dg1_dx2(x):
    return x[2]*sin(x[1]*x[2])


def dg1_dx3(x):
    return x[1]*sin(x[1]*x[2])


def dg2_dx1(x):
    return 2*x[0]


def dg2_dx2(x):
    return -162*(x[1]+0.1)


def dg2_dx3(x):
    return cos(x[2])


def dg3_dx1(x):
    return -x[1]*expn(-x[0]*x[1])


def dg3_dx2(x):
    return -x[0]*expn(-x[0]*x[1])


def dg3_dx3(x):
    return 20


def dg1_dx1_2(x):   # g Second derivatives
    return 0


def dg1_dx2_2(x):
    return (x[2]**2)*cos(x[1]*x[2])


def dg1_dx3_2(x):
    return (x[1]**2)*cos(x[1]*x[2])


def dg1_dx1_dx2(x):
    return 0


def dg1_dx1_dx3(x):
    return 0


def dg1_dx2_dx3(x):
    return sin(x[1]*x[2])+x[1]*x[2]*cos(x[1]*x[2])


def dg2_dx1_2(x):
    return 2


def dg2_dx2_2(x):
    return -162


def dg2_dx3_2(x):
    return -sin(x[2])


def dg2_dx1_dx2(x):
    return 0


def dg2_dx1_dx3(x):
    return 0


def dg2_dx2_dx3(x):
    return 0


def dg3_dx1_2(x):
    return (x[1]**2)*exp(-x[0]*x[1])


def dg3_dx2_2(x):
    return (x[0]**2)*exp(-x[0]*x[1])


def dg3_dx3_2(x):
    return 0


def dg3_dx1_dx2(x):
    return x[1]*x[0]*exp(-x[0]*x[1])-exp(-x[0]*x[1])


def dg3_dx1_dx3(x):
    return 0


def dg3_dx2_dx3(x):
    return 0


def df_dx1(x):  # f First derivatives
    return g1(x)*dg1_dx1(x)+g2(x)*dg2_dx1(x)+g3(x)*dg3_dx1(x)


def df_dx2(x):
    return g1(x)*dg1_dx2(x)+g2(x)*dg2_dx2(x)+g3(x)*dg3_dx2(x)


def df_dx3(x):
    return g1(x)*dg1_dx3(x)+g2(x)*dg2_dx3(x)+g3(x)*dg3_dx3(x)


def grad(x):  # f gradient
    g = np.array([df_dx1(x), df_dx2(x), df_dx3(x)])
    return g


def df_dx1_2(x):  # f Second derivatives
    return g1(x)*dg1_dx1_2(x)+(dg1_dx1(x)**2)+g2(x)*dg2_dx1_2(x)+(dg2_dx1(x)**2)+g3(x)*dg3_dx1_2(x)+(dg3_dx1(x)**2)


def df_dx2_2(x):
    return g1(x)*dg1_dx2_2(x)+(dg1_dx2(x)**2)+g2(x)*dg2_dx2_2(x)+(dg2_dx2(x)**2)+g3(x)*dg3_dx2_2(x)+(dg3_dx2(x)**2)


def df_dx3_2(x):
    return g1(x)*dg1_dx3_2(x)+(dg1_dx3(x)**2)+g2(x)*dg2_dx3_2(x)+(dg2_dx3(x)**2)+g3(x)*dg3_dx3_2(x)+(dg3_dx3(x)**2)


def df_dx1_dx2(x):
    return (g1(x)*dg1_dx1_dx2(x)+dg1_dx2(x)*dg1_dx1(x)+g2(x)*dg2_dx1_dx2(x)+dg2_dx2(x)*dg2_dx1(x)+g3(x)*dg3_dx1_dx2(x)
            + dg3_dx2(x)*dg3_dx1(x))


def df_dx1_dx3(x):
    return (g1(x)*dg1_dx1_dx3(x)+dg1_dx3(x)*dg1_dx1(x)+g2(x)*dg2_dx1_dx3(x)+dg2_dx3(x)*dg2_dx1(x)+g3(x)*dg3_dx1_dx3(x)
            + dg3_dx3(x)*dg3_dx1(x))


def df_dx2_dx3(x):
    return (g1(x)*dg1_dx2_dx3(x)+dg1_dx2(x)*dg1_dx3(x)+g2(x)*dg2_dx2_dx3(x)+dg2_dx2(x)*dg2_dx3(x)+g3(x)*dg3_dx2_dx3(x)
            + dg3_dx2(x)*dg3_dx3(x))


def hessian(x):
    h = np.array([[df_dx1_2(x), df_dx1_dx2(x), df_dx1_dx3(x)],
                  [df_dx1_dx2(x), df_dx2_2(x), df_dx2_dx3(x)],
                  [df_dx1_dx3(x), df_dx2_dx3(x), df_dx3_2(x)]])
    return h


def f_in_terms_of_eta(x):
    def phi(learning_rate):
        return f(x - learning_rate*grad(x))
    return phi


def gradient_descent(f, grad, learning_rate=0.00001, iterations=1000, x0=[-0.1, -0.1, -0.1], epsilon=0.001):
    x_prev = x0
    i = 1
    total_grad = 10
    while (i <= iterations) and (abs(total_grad) > epsilon):
        x_current = x_prev - learning_rate * grad(x_prev)
        total_grad = (1/3)*np.sum(grad(x_current))
        print("iteration " + str(i) + ":")
        print("x = " + str(x_current))
        print("f = " + str(f(x_current)))
        print("grad = " + str(total_grad))
        plt.plot(i, total_grad, 'bo')
        plt.plot(i, f(x_current), 'go')
        i += 1
        x_prev = x_current
    plt.plot(i, total_grad, 'bo', label='Gradient')
    plt.plot(i, f(x_current), 'go', label='Function')
    plt.legend()
    plt.grid(True)
    plt.title("GD Function magnitude and Gradient magnitude")
    plt.show()
    print("x = " + str(x_current))
    print("f = " + str(f(x_current)))
    return x_current


def nrfd(f, grad, hessian, iterations=1000, x0=[-0.1, -0.1, -0.1], epsilon=0.001):
    x_prev = x0
    i = 1
    total_grad = 10
    while (i <= iterations) and (abs(total_grad) > epsilon):
        x_current = x_prev - np.dot(np.linalg.inv(hessian(x_prev)), grad(x_prev))
        total_grad = (1 / 3) * np.sum(grad(x_current))
        print("iteration " + str(i) + ":")
        print("x = " + str(x_current))
        print("f = " + str(f(x_current)))
        print("grad = " + str(total_grad))
        plt.plot(i, total_grad, 'bo')
        plt.plot(i, f(x_current), 'go')
        i += 1
        x_prev = x_current
    plt.plot(i, total_grad, 'bo', label='Gradient')
    plt.plot(i, f(x_current), 'go', label='Function')
    plt.legend()
    plt.grid(True)
    plt.title("NRFD Function magnitude and Gradient magnitude")
    plt.show()
    print("x = " + str(x_current))
    print("f = " + str(f(x_current)))
    return x_current


def steepest_descent(f, grad, iterations=1000, n0=0.1, x0=[-0.1, -0.1, -0.1], epsilon=0.001):
    x_prev = x0
    prev_learning_rate = n0
    i = 1
    total_grad = 10
    while (i <= iterations) and (abs(total_grad) > epsilon):
        phi = f_in_terms_of_eta(x_prev)
        current_learning_rate = float(minimize(phi, prev_learning_rate).x)
        x_current = x_prev - current_learning_rate * grad(x_prev)
        total_grad = (1 / 3) * np.sum(grad(x_current))
        print("iteration " + str(i) + ":")
        print("n = " + str(current_learning_rate))
        print("x = " + str(x_current))
        print("f = " + str(f(x_current)))
        print("grad = " + str(total_grad))
        plt.plot(i, total_grad, 'bo')
        plt.plot(i, f(x_current), 'go')
        i += 1
        x_prev = x_current
    plt.plot(i, total_grad, 'bo', label='Gradient')
    plt.plot(i, f(x_current), 'go', label='Function')
    plt.legend()
    plt.grid(True)
    plt.title("SD Function magnitude and Gradient magnitude")
    plt.show()
    print("x = " + str(x_current))
    print("f = " + str(f(x_current)))
    return x_current


# For trying many random initializations:
# best_solution = float('inf')
# for i in range(100):
#    print("iteration " + str(i) + ":")
#    x0 = np.random.randn(3)
#    print(x0)
#    x = gradient_descent(f, grad, x0=np.random.randn(3)*0.1)
#    if x < best_solution:
#        best_solution = x
# print("Gradient Descent answer: " + str(best_solution))

print("Gradient Descent answer: " + str(gradient_descent(f, grad)))

print("NRFD answer: " + str(nrfd(f, grad, hessian)))

print("Steepest Descent answer: " + str(steepest_descent(f, grad)))

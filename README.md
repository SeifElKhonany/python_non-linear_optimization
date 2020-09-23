# python_non-linear_optimization
Non-linear optimization using Gradient Descent, Steepest Descent and Newton-Raphson’s method.  

The objective of this project is to find a solution for the following set of nonlinear equations using different optimization techniques:-  

𝑔1(𝑥1,𝑥2,𝑥3) = 3𝑥1 − cos(𝑥2𝑥3) − 0.5 = 0  
𝑔2(𝑥1,𝑥2,𝑥3) = 𝑥1 2 − 81(𝑥2 + 0.1)2 + sin(𝑥3) + 1.06 = 0  
𝑔3(𝑥1,𝑥2,𝑥3) = exp(−𝑥1𝑥2) + 20𝑥3 + (10𝜋 − 3)/3 = 0  

The problem now is to minimize the following suggested objective function:-  

𝐹(𝑥1,𝑥2,𝑥3) = 1/2 * ([𝑔1(𝑥1,𝑥2,𝑥3)]2 + [𝑔2(𝑥1,𝑥2,𝑥3)]2 + [𝑔3(𝑥1,𝑥2,𝑥3)]2)

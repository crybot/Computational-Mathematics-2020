### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ c08080a4-2a50-11eb-29fa-2f9caa2ed92c
using LinearAlgebra, Plots

# ╔═╡ 2c389edc-2919-11eb-0ade-c77d3223df6f
md"# Steepest descent method"

# ╔═╡ 67f2a030-2919-11eb-2f2c-b15bb975af36
md"## Quadratic functions "

# ╔═╡ d94588ec-2919-11eb-0fab-935856446b79
md"""
Let $f(x) = x^TQx + q^Tx$ with $Q \succeq 0$

We want to find $x_* = \underset{x}{\arg\min}{f(x)}$

Given that $d_i = -\nabla{f(x_i)}$ is the direction of steepest descent at iteration $i$, one can then perform a line search to find the optimal step size $\alpha_i$ for which: $\alpha_i = \underset{\alpha}{\arg\min}\{ f(x_i + \alpha d_i) \mid \alpha \geq 0 \}$.

In the quadratic case, one can solve this \"exact\" line search analytically:

Let $\varphi(\alpha) = f(x + \alpha d)$, We can then compute $\varphi'(\alpha)$ and find its roots, thus minimizing $f(x)$ along the ray $x + \alpha d$:


$\begin{align}
\varphi(\alpha) &= f(x + \alpha d) \\
				&= (x + \alpha d)^TQ(x + \alpha d) + q^T(x + \alpha d) \\
				&= x^TQx + \alpha x^TQd + \alpha d^TQx 
					+ \alpha^2 d^TQd + q^Tx + \alpha q^Td \\
				&\text{\{ Since we can assume Q symmetric \}} \\
				&= x^TQx + 2\alpha x^TQd + \alpha^2 d^TQd + q^Tx + \alpha q^Td
\end{align}$


Then

$\begin{align}
\varphi'(\alpha) &= 2x^TQd + 2\alpha d^TQd + q^Td = 0 \iff \\
				 \alpha &= \frac{-x^TQd - q^Td}{d^TQd} \\
				 &= \frac{(-x^TQ - q^T)d}{d^TQd} \\
				 &= \frac{(-Qx - q)^Td}{d^TQd} \\
				 &= \frac{(-\nabla{f(x)})^T(-\nabla{f(x)})}{d^TQd}\\
				 &= \frac{{||d||}^2}{d^TQd}

\end{align}$

Which gives the best $\alpha_i$ at each iteration, given $d_i$

"""

# ╔═╡ b7627e60-2919-11eb-2d47-dd60de41c9f4
function SDQ(Q, q, x0; eps=1e-6, maxiter=1000)
	if Q != Q'
		error("Q not symmetric")
	end
	
	xi = copy(x0)
	iterates = [xi]
	
	for i in 1:maxiter
		g = Q*xi + q
		if (norm(g) <= eps) return xi, iterates end
		
		D = g'Q*g
		
		if D <= 1e-12
			# error("Problem unbounded below")
			return xi, iterates
		end
		
		alpha = norm(g)^2 / D
		xi = xi - alpha*g
		
		push!(iterates, xi)
	end
	
	return xi, iterates
end

# ╔═╡ bbbc1f6e-2a5b-11eb-2973-ad94e6bdf1f7
function plotQ(Q, q; start=-1, stop=1)
	x1 = range(start, stop=stop, length=50)
	x2 = range(start, stop=stop, length=50)
	if size(Q)[1] != 2
		error("Cannot plot function with more than two variables")
	end
	f(x, y) = [x; y]'*Q*[x; y] + q'*[x; y]
	contour(x1, x2, f, levels=10)
end

# ╔═╡ 966d0346-2a50-11eb-101d-2b7ff589d664
Q = [8 8; 8 18]; q = [1.; 1.]; x0 = [0.05; 0.]

# ╔═╡ d01b174e-2a56-11eb-219f-f90f121b4e16
eigen(Q).values

# ╔═╡ 996223ac-2a8d-11eb-3471-dd554dced785
md"#### Gradient descent solution"

# ╔═╡ adca86b2-2a50-11eb-08ed-7bfdb9036116
x, xs = SDQ(Q, q, x0, eps=1e-6)

# ╔═╡ acd10f90-2a91-11eb-2806-716920a99392
md"**Slide to zoom in**"

# ╔═╡ cce658ba-2a90-11eb-21fb-8f64d3ea0b65
@bind zoom html"<input type='range' min=-5 max=0 step=0.01>"

# ╔═╡ 8dcf2c0e-2a92-11eb-2569-51332df2c480
md"**Move the center of the contour plot**"

# ╔═╡ 9fa8756e-2a6c-11eb-0340-51ba20bf2597
begin
	ps = hcat(xs...)
	plotQ(Q, q, start=zoom, stop=-zoom)
	plot!(ps[1, :], ps[2, :], xlim=(zoom, -zoom), ylim=(zoom, -zoom))
	scatter!(ps[1, :], ps[2, :], markersize=3, legend=false)
end

# ╔═╡ 7cbc589e-2a8d-11eb-02b2-2f8915b47e55
md"#### Analytical solution"

# ╔═╡ 159bd1ce-2a51-11eb-237b-675e34681b75
x̄ = Q \ -q

# ╔═╡ 8cc82752-2a5b-11eb-1068-51a4da9cec9e


# ╔═╡ Cell order:
# ╟─2c389edc-2919-11eb-0ade-c77d3223df6f
# ╟─67f2a030-2919-11eb-2f2c-b15bb975af36
# ╟─d94588ec-2919-11eb-0fab-935856446b79
# ╠═c08080a4-2a50-11eb-29fa-2f9caa2ed92c
# ╠═b7627e60-2919-11eb-2d47-dd60de41c9f4
# ╠═bbbc1f6e-2a5b-11eb-2973-ad94e6bdf1f7
# ╠═966d0346-2a50-11eb-101d-2b7ff589d664
# ╠═d01b174e-2a56-11eb-219f-f90f121b4e16
# ╟─996223ac-2a8d-11eb-3471-dd554dced785
# ╠═adca86b2-2a50-11eb-08ed-7bfdb9036116
# ╟─acd10f90-2a91-11eb-2806-716920a99392
# ╠═cce658ba-2a90-11eb-21fb-8f64d3ea0b65
# ╟─8dcf2c0e-2a92-11eb-2569-51332df2c480
# ╠═9fa8756e-2a6c-11eb-0340-51ba20bf2597
# ╟─7cbc589e-2a8d-11eb-02b2-2f8915b47e55
# ╠═159bd1ce-2a51-11eb-237b-675e34681b75
# ╟─8cc82752-2a5b-11eb-1068-51a4da9cec9e

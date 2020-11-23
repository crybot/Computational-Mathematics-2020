### A Pluto.jl notebook ###
# v0.12.11

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

# ╔═╡ 421d2d24-2ce0-11eb-08dd-d3b090d7ee84
using ForwardDiff, Printf

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
				\text{ Since we can assume Q symmetric: } \\
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

# ╔═╡ 966d0346-2a50-11eb-101d-2b7ff589d664
Q = [8 8; 8 18]; q = [0.; 0.]; x0 = rand(2)

# ╔═╡ d01b174e-2a56-11eb-219f-f90f121b4e16
eigen(Q).values

# ╔═╡ 996223ac-2a8d-11eb-3471-dd554dced785
md"#### Gradient descent solution"

# ╔═╡ adca86b2-2a50-11eb-08ed-7bfdb9036116
x, xs = SDQ(Q, q, x0, eps=1e-6)

# ╔═╡ acd10f90-2a91-11eb-2806-716920a99392
md"**Slide to zoom in**"

# ╔═╡ cce658ba-2a90-11eb-21fb-8f64d3ea0b65
@bind zoom html"<input type='range' min=-1 max=-0.000001 step=0.00000001>"

# ╔═╡ 7cbc589e-2a8d-11eb-02b2-2f8915b47e55
md"#### Analytical solution"

# ╔═╡ 159bd1ce-2a51-11eb-237b-675e34681b75
x̄ = Q \ -q

# ╔═╡ 8ecb90e2-2cd7-11eb-2ab1-1b9150101618
md"## General functions "

# ╔═╡ 9c1bed1e-2cd7-11eb-2c0c-29e8e7a0f117
md"To optimize general functions with the gradient method we must usa an approximate \"exact\" line search, or drop the exactness requirement altogether and use an inexact search method."

# ╔═╡ 4f561fc6-2cd8-11eb-28d2-237ba9dd2763
md" ### Approximately exact line search"

# ╔═╡ 791018c8-2cd8-11eb-2f5b-69283d1cea0c
md"""
The main result of approximately exact line search methods is that we can prove convergence of the gradient method, having fixed the required precision $\epsilon$:

$|\varphi(\alpha)| \leq \epsilon \implies ||\nabla{f(x)}|| \leq \epsilon$
"""

# ╔═╡ 32a111aa-2cd9-11eb-19ff-49f773b3ac97
md"""
We then need to find an interval in which to perform the line search, that is we need to find an $\bar{\alpha}$ s.t.:

$\exists \alpha \in (0,\, \bar{\alpha}] : \varphi(\bar{\alpha})$
"""

# ╔═╡ 6a45a812-2cdb-11eb-2216-5fe5bf962451
function findinterval(φ, ϵ)
	dφ(α) = ForwardDiff.derivative(φ, α)
	α = 1.0
	while dφ(α) <= -ϵ
		α = 2α
	end
	return α
end

# ╔═╡ ae40407e-2d15-11eb-2e90-7b4cf02b475c
md"The following plot shows the interval found by the algorithm for a generic quadratic function:"

# ╔═╡ 31360e28-2d96-11eb-12a8-5f99d3c6b24b
md"#### Bisection Method"

# ╔═╡ 44e1536c-2d96-11eb-2807-415965ca6568
md"""
In the bisection method we exploit the interval found by the previous algorithm and iteratively reduce it until we find a point in the interval such that $\varphi'(\alpha) \approx 0$.

The naive implementation exponentially reduces the size of the interval by considering the checking, at each iteration, the sign of $\varphi'$ in the middle point of $[\alpha_-, \alpha_+]$: $\quad \displaystyle \frac{\alpha_+ + \alpha_-}{2}$
"""

# ╔═╡ 1893ca8c-2d97-11eb-3366-0b49c4375551
function bisection(φ, ᾱ, ϵ)
	α₋ = 0
	α₊ = ᾱ
	α = α₊
	dφ(α) = ForwardDiff.derivative(φ, α)
	
	iterates = [(α₋, α₊)]
	while abs(dφ(α)) > ϵ
		α = (α₊ + α₋) / 2
		if dφ(α) < 0
			α₋ = α
		else
			α₊ = α
		end
		iterates = [iterates; (α₋, α₊)]
	end
	
	return α, iterates
end

# ╔═╡ 332b267c-2d9e-11eb-3f09-0f54ae7d6481
begin
	# Function to optimize
	f(x::Vector) = 3x'x
	g(x) = ForwardDiff.gradient(f, x)
	
	# Starting point
	x1 = [-2; -1]
	
	# Normalized gradient (trick!)
	d = -g(x1) ./ norm(g(x1))
	
	φ(α) = f(x1 + α*d)

	ᾱ = findinterval(φ, 1e-6)
	
	# Return the found alpha along with the incrementally computed intervals
	α, its = bisection(φ, ᾱ, 1e-6)
end

# ╔═╡ bbbc1f6e-2a5b-11eb-2973-ad94e6bdf1f7
function plotQ(Q, q; start=-1, stop=1)
	x1 = range(start, stop=stop, length=50)
	x2 = range(start, stop=stop, length=50)
	if size(Q)[1] != 2
		error("Cannot plot function with more than two variables")
	end
	f(x, y) = [x; y]'*Q*[x; y] + q'*[x; y]
	contour(x1, x2, f, levels=15)
end

# ╔═╡ 9fa8756e-2a6c-11eb-0340-51ba20bf2597
begin
	ps = hcat(xs...)
	plotQ(Q, q, start=zoom, stop=-zoom)
	plot!(ps[1, :], ps[2, :],
		xlim=(zoom, -zoom),
		ylim=(zoom, -zoom),
		linecolor=:black,
		linewidth=2)
	scatter!(ps[1, :], ps[2, :], markersize=3, legend=false)
end

# ╔═╡ e0eb90fa-2cde-11eb-131e-491f17ba4f79
let
	# Function to optimize
	f(x::Vector) = 2x'x
	g(x) = ForwardDiff.gradient(f, x)
	
	# Starting point
	x0 = [-2; -1]
	
	# Normalized gradient (trick!)
	d = -g(x0) ./ norm(g(x0))
	
	φ(α) = f(x0 + α*d)
	dφ(α) = g(x0 + α*d)' * d

	ᾱ = findinterval(φ, 1e-6)
	
	# Plot x/y limits
	xmin = - 1
	xmax = ᾱ + 2
	ymin = - 10
	ymax = φ(ᾱ) + 20
	
	# Plot φ(α) 
	plot(range(xmin, stop=xmax, length=100), φ, linecolor=1, legend=false, 
		xlabel="α", ylabel="φ(α)",
		xlim=(xmin, xmax), ylim=(ymin, ymax))
	
	# Plot area under φ(α) for α ∈ [0, ᾱ]
	plot!(range(0, stop=ᾱ, length=100), φ, linecolor=1,
		fill=(ymin, :gray), fillalpha=0.2)
	
	# Plot φ'(α)
	plot!(dφ, linewidth=2, linecolor=3, xlim=(xmin, xmax))
	
	# Plot the point ᾱ
	scatter!([ᾱ], [φ(ᾱ)])
	scatter!([ᾱ], [φ(ᾱ) + 4], markersize=0, series_annotations=(["ᾱ"], :bottom))
end


# ╔═╡ e2362cd6-2d9a-11eb-1ab4-0536de884579
function plot_interval(f, a, b; xmin=-1, xmax=6)	
	# Plot x/y limits
	ymin = - 10
	ymax = max(f(a) + 20, f(b) + 20)
	
	# Plot f(x) 
	plot(range(xmin, stop=xmax, length=100), f, linecolor=1, legend=false, 
		xlabel="α", ylabel="φ(α)",
		xlim=(xmin, xmax), ylim=(ymin, ymax))
	
	# Plot area under f(x) for x ∈ [a, b]
	plot!(range(a, stop=b, length=100), f, linecolor=1,
		fill=(ymin, :gray), fillalpha=0.1)
	
	plot!([a, a], [ymin, φ(a)], linecolor=:gray, linewidth=1.5, opacity=0.4)
	plot!([b, b], [ymin, φ(b)], linecolor=:gray, linewidth=1.5, opacity=0.4)

	
	# Plot the point ᾱ
	scatter!([a, b], [f(a), f(b)], color=:green)
	scatter!([a, b], [f(a) + 4, f(b) + 4],
		markersize=0, series_annotations=(["α₋" ,"α₊"], :bottom))
end

# ╔═╡ 6f045826-2da0-11eb-16f7-d7fa4dc1a966
md"**Animation speed**"

# ╔═╡ eabfea54-2d9c-11eb-0597-0f60b99032ca
@bind speed html"<input type=range min=1 max=30 value=1 step=0.5/>"

# ╔═╡ 17722386-2d99-11eb-3aa4-afe95538ced2
let
	# it is the value bound to the html input slider (values between 0 and 100)
	# i = floor(Int, 1 + (length(its) - 1)*(it/100))
	# plot_interval(φ, its[i][1], its[i][2], xmin=-2, xmax=ᾱ+2)
	
	# @gif for it in its
		# plot_interval(φ, it[1], it[2], xmin=-2, xmax=ᾱ+2)
	# end
	
	anim = @animate for it in its
		if abs(it[1] - it[2]) <= 1e-2
			break
		end
		plot_interval(φ, it[1], it[2], xmin=-2, xmax=ᾱ+2)
	end
	
	gif(anim, "anim_fps15.gif", fps = speed)
end

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
# ╟─cce658ba-2a90-11eb-21fb-8f64d3ea0b65
# ╠═9fa8756e-2a6c-11eb-0340-51ba20bf2597
# ╟─7cbc589e-2a8d-11eb-02b2-2f8915b47e55
# ╠═159bd1ce-2a51-11eb-237b-675e34681b75
# ╟─8ecb90e2-2cd7-11eb-2ab1-1b9150101618
# ╟─9c1bed1e-2cd7-11eb-2c0c-29e8e7a0f117
# ╟─4f561fc6-2cd8-11eb-28d2-237ba9dd2763
# ╟─791018c8-2cd8-11eb-2f5b-69283d1cea0c
# ╟─32a111aa-2cd9-11eb-19ff-49f773b3ac97
# ╠═421d2d24-2ce0-11eb-08dd-d3b090d7ee84
# ╠═6a45a812-2cdb-11eb-2216-5fe5bf962451
# ╟─ae40407e-2d15-11eb-2e90-7b4cf02b475c
# ╠═e0eb90fa-2cde-11eb-131e-491f17ba4f79
# ╟─31360e28-2d96-11eb-12a8-5f99d3c6b24b
# ╟─44e1536c-2d96-11eb-2807-415965ca6568
# ╠═1893ca8c-2d97-11eb-3366-0b49c4375551
# ╠═332b267c-2d9e-11eb-3f09-0f54ae7d6481
# ╟─e2362cd6-2d9a-11eb-1ab4-0536de884579
# ╟─6f045826-2da0-11eb-16f7-d7fa4dc1a966
# ╟─eabfea54-2d9c-11eb-0597-0f60b99032ca
# ╠═17722386-2d99-11eb-3aa4-afe95538ced2
# ╟─8cc82752-2a5b-11eb-1068-51a4da9cec9e

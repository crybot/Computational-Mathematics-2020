### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 5e5ff2ca-2455-11eb-3cde-8537738b7ad7
using LinearAlgebra

# ╔═╡ 5a8c6346-24ea-11eb-35ba-bb262de0dbda
md"# QR Factorization"

# ╔═╡ e59e4d0c-246d-11eb-015c-d38300d3d5ba
md"### Lemma"

# ╔═╡ 55e4a00a-2446-11eb-38a3-8731acb74bec
md"""
Let $x,y \in \mathbb{R}^n$ s.t. $||x|| = ||y||$
"""

# ╔═╡ 189095de-244c-11eb-0fa3-fb19c5de8c71
md"""
Let $v = x - y, \quad$ $u = \frac{v}{||v||} \quad$
and
$\quad H = I - 2uu^T, \quad$ then

$Hx = y$
"""

# ╔═╡ 200b003e-244e-11eb-26d4-d5f39d6dfd5a
md"""
We can use this Lemma to construct an Householder reflector $H$ that sends a vector $x$
into a vector

$y = \begin{bmatrix} \pm{||x||} \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}$
"""

# ╔═╡ 09bd2ae2-24ed-11eb-03ea-2b38210f576e
md" ### Householder transformation"

# ╔═╡ c3b68716-244f-11eb-23dd-0da3d7a623d8
function householder(x)
	# Return reflector H such that Hx = y
	n = length(x)
	y = [norm(x); zeros(n-1)]
	# Avoid division by zero: norm(v) = 0 ≡ v = 0 ≡ x = y
	if x == y
		return Matrix{Float64}(I, n, n), zeros(n)
	end
	if x[1] >= 0
		y[1] = -y[1]
	end
	v = x - y
	u = v / norm(v)
	H = I - 2u*u', u
end

# ╔═╡ 2a752918-2463-11eb-0ec3-73a87f1797c8
x = rand(3)

# ╔═╡ 6f011278-2465-11eb-2aeb-2b0c1353064f
H, _ = householder(x[:,1])

# ╔═╡ 7581f5e4-2463-11eb-0eb5-d911b18aeaf9
H*x

# ╔═╡ 5dbcd428-24dc-11eb-28fb-3f4f1e964da9
md"""
### Naive QR implementation

We apply Householder transformations by plain matrix multiplication

$Q_n Q_{n-1} \dots Q_1 A = R$,

$Q_i = \begin{bmatrix}I & 0 \\ 0 & H_i\end{bmatrix},
\quad H_i = I - 2u_iu_i^T \quad$ With each $u_i$ s.t. 

$H_i R_{i:, i} =
\begin{bmatrix}\pm||R_{i,i}||, & 0, & \dots, & 0 \end{bmatrix}^T$
"""

# ╔═╡ a963a046-24dc-11eb-320d-156dd4e3b1ce
function naiveqr(A)
	Q = I
	R = copy(A)
	n = size(A, 1)
	for i = 1:n
		H, _ = householder(R[i:end, i])
		Qi = Matrix{Float64}(I, n, n)
		Qi[i:end, i:end] = H
		R = Qi * R
		Q = Q * Qi
	end
	return Q, R
end

# ╔═╡ f483d5ea-246f-11eb-38d4-93e9a08adc24
A = rand(3,3)

# ╔═╡ d8748b4e-2472-11eb-1578-e169d4ed0ee0
begin 
	Q, R = naiveqr(A)
	Q,R, Q * R
end

# ╔═╡ ebefbe0e-24dc-11eb-261c-95e1b24e86bc
md"""
### Faster QR implementation

We expand each Householder transformation to reduce computational complexity and only perform multiplications on sub-blocks of $ R $.
"""

# ╔═╡ e122c800-246d-11eb-35fe-0ba680821ed6
function fastqr(A)
	Q = I
	R = copy(A)
	n = size(A, 1)
	for i = 1:n
		H, u = householder(R[i:end, i])
		Qi = Matrix{Float64}(I, n, n)
		Qi[i:end, i:end] = H	
		# Manually set R[i,i] = s
		s = norm(R[i:end, i])
		R[i, i] = if (R[i,i] < 0) s else -s end
		# Manually set zeros in i-th column of the current R sub-block
		R[i+1:end, i] .= 0
		# Perform computation only on the current sub-block of R
		M = R[i:end, i+1:end]
		# Expand Householder transformation: H⋅M = (I - 2uuᵀ)M
		R[i:end, i+1:end] = M - 2u*(u'*M)
		# Should do the same for Q, but I'm lazy (use your imagination)
		Q = Q * Qi
	end
	return Q, R
end

# ╔═╡ 7a30658e-24de-11eb-3d9c-11521e6a67e9
Q₁, R₁ = fastqr(A)

# ╔═╡ Cell order:
# ╟─5a8c6346-24ea-11eb-35ba-bb262de0dbda
# ╟─e59e4d0c-246d-11eb-015c-d38300d3d5ba
# ╟─55e4a00a-2446-11eb-38a3-8731acb74bec
# ╟─189095de-244c-11eb-0fa3-fb19c5de8c71
# ╟─200b003e-244e-11eb-26d4-d5f39d6dfd5a
# ╠═5e5ff2ca-2455-11eb-3cde-8537738b7ad7
# ╠═09bd2ae2-24ed-11eb-03ea-2b38210f576e
# ╠═c3b68716-244f-11eb-23dd-0da3d7a623d8
# ╠═2a752918-2463-11eb-0ec3-73a87f1797c8
# ╠═6f011278-2465-11eb-2aeb-2b0c1353064f
# ╠═7581f5e4-2463-11eb-0eb5-d911b18aeaf9
# ╟─5dbcd428-24dc-11eb-28fb-3f4f1e964da9
# ╠═a963a046-24dc-11eb-320d-156dd4e3b1ce
# ╠═f483d5ea-246f-11eb-38d4-93e9a08adc24
# ╠═d8748b4e-2472-11eb-1578-e169d4ed0ee0
# ╟─ebefbe0e-24dc-11eb-261c-95e1b24e86bc
# ╠═e122c800-246d-11eb-35fe-0ba680821ed6
# ╠═7a30658e-24de-11eb-3d9c-11521e6a67e9

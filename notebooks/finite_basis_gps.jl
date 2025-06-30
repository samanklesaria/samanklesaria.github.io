### A Pluto.jl notebook ###
# v0.20.11

#> [frontmatter]
#> title = "Finite Basis Gaussian Processes"
#> date = "2024-04-02"
#> category = "machine_learning"

using Markdown
using InteractiveUtils

# ╔═╡ e1e011e8-3e91-4317-a081-9a23f39349c8
using KernelFunctions,LinearAlgebra, AbstractGPs, Random

# ╔═╡ 4140ca9d-2530-43a9-aee8-73f7fb0770ec
using BenchmarkTools

# ╔═╡ b18c628c-cced-11ee-0033-51bdcc63c29c
md"""
By Mercer's theorem, every positive definite kernel $k(x, y) : \mathcal{X} \to \mathcal{X} \to \mathbb{R}$ that we might want to use in a Gaussian Process corresponds to some inner product $\langle \phi(x), \phi(y) \rangle$, where $\phi : \mathcal{X} \to \mathcal{V}$ maps our inputs into some other space.  For many kernels (like the venerable RBF), this space is infinite dimensional, and we can't work with it directly. But when it's finite dimensional (in say $d$ dimensions), we can! This lets us avoid the usual $O(n^3)$ scaling for Gaussian process regression, getting $O(nd+d^3)$ instead.
"""

# ╔═╡ ca1fa3bc-c6a8-400a-93ed-850821f57b1f
import AbstractGPs: AbstractGP, FiniteGP

# ╔═╡ 3a04e978-770c-45d1-be1c-5cc1c1925eb2
import Statistics

# ╔═╡ 660c7753-ec86-4f36-92b5-eec16d17dbb4
struct FiniteBasis <: KernelFunctions.SimpleKernel end

# ╔═╡ f06e2a85-e77e-492b-ae4d-d12d0a2809e8
md"We can define a finite dimensional kernel in Julia using the `KernelFunctions` library. The library assumes our kernel `k` has the form `k(x,y) = kappa(metric(x,y))`, and lets us fill in the definitions for `kappa` and `metric`."

# ╔═╡ a8a1da9e-78e1-465a-83c9-7d23cb77d926
KernelFunctions.kappa(::FiniteBasis, d::Real) = d

# ╔═╡ 26b76664-6f17-4e4e-9009-fe6ac063c323
KernelFunctions.metric(::FiniteBasis) = KernelFunctions.DotProduct()

# ╔═╡ 8c973260-b0a0-49fe-9b18-7dd8ee6d467b
md"""
We will use the *weight space* view of Gaussian processes, which interprets GP regression as Bayesian linear regression. We assume that there is a weight vector $w : \mathcal{V}$ with prior $\mathcal{N}(0, I)$, and that $y \sim \mathcal{N}(X w, I)$, where $X$ is the matrix for which row $i$ is given by $\phi(x_i)$.
The posterior over $w$ remains Gaussian with precision $\Lambda = I + X^T X$ and mean $\mu = \Lambda^{-1} X^T y$. To make a prediction at $x_*$, we simply find $\langle \phi(x_*), w \rangle$.
"""

# ╔═╡ 8770765c-e519-4cd7-9eed-da2b50190895
md"""
On the face of it, this seems like a very different generative model than the traditional depiction of Gaussian processes in which the observations $y$ are noisy versions of the function values $f$, which are all jointly Gaussian with a covariance matrix given by the associated kernel. But with a little algebra, one can show that the posterior over $f(x_*) = \langle \phi(x_*), w \rangle$ in the weight space view is the same as the posterior over $f(x_*)$ is the traditional function-space view.

First, we can marginalize out $w$ to find that

```math
f(x_*) | y \sim \mathcal{N}(X_* \mu, X_* \Lambda^{-1} X_*^T)
```
The mean expands to $X_*(I + X^T X)^{-1} X^T y$ and the variance expands to
$X_*(I + X^T X)^{-1}X_*^T$.


Now, we can use the Woodbury Matrix Identity, which says that
```math
(I + X^TX)^{-1} = I - X^T(I + XX^T)^{-1}X
```
This lets the mean simplify to
$X_*X^T (XX^T + I)^{-1}y$ and the variance simplify to $X_*X_*^T -X_*X^T(XX^T + I)^{-1}XX_*^T$. Letting $XX^T = K$, we recover the familiar function space representation of Gaussian process. See the first chapter of the [Rasmussen book](http://gaussianprocess.org/gpml/) for a more detailed derivation.
"""

# ╔═╡ 38ccb850-2091-4914-a7fa-0fdfe0e64375
struct DegeneratePosterior{P,T,C} <: AbstractGP
	prior::P
	w_mean::T
	w_prec::C
end

# ╔═╡ 2646ce1f-bd5c-4ea0-a9fb-c6f786984240
weight_form(A::KernelFunctions.ColVecs) = A.X';

# ╔═╡ 60390567-e70f-463b-a0be-480d1b9c3198
weight_form(A::KernelFunctions.RowVecs) = A.X;

# ╔═╡ 8c2d7c8b-8d9e-44fe-88b9-00fd3e3aee14
function Statistics.mean(f::DegeneratePosterior, x::AbstractVector)
	w = f.w_mean
	X = weight_form(x)
	X * w
end

# ╔═╡ 841129b7-0d84-4f6b-9ae0-edf2a6cb8661
function AbstractGPs.posterior(fx::FiniteGP{GP{M, B}}, y::AbstractVector{<:Real}) where {M, B <: FiniteBasis}
	kern = fx.f.kernel
	δ = y - mean(fx)
	X = weight_form(fx.x)
	X_prec = X' * inv(fx.Σy)
	Λμ = X_prec * y
	prec = cholesky(I + Symmetric(X_prec * X))
	w = prec \ Λμ
	DegeneratePosterior(fx.f, w, prec)
end

# ╔═╡ 9fe52847-ae03-461e-a5f9-fb95adc63cb4
function Statistics.cov(f::DegeneratePosterior, x::AbstractVector)
	X = weight_form(x)
	AbstractGPs.Xt_invA_X(f.w_prec, X')
end

# ╔═╡ b623703a-137c-407b-9094-b19abe3593b5
function Statistics.cov(f::DegeneratePosterior, x::AbstractVector, y::AbstractVector)
	X = weight_form(x)
	Y = weight_form(y)
	AbstractGPs.Xt_invA_Y(X', f.w_prec, Y')
end

# ╔═╡ 0f937650-77db-43cb-859a-a0f27dcc464d
function Statistics.var(f::DegeneratePosterior, x::AbstractVector)
	X = weight_form(x)
	AbstractGPs.diag_Xt_invA_X(f.w_prec, X')
end

# ╔═╡ 8a7d7a98-278c-4558-90fd-6a382c022236
function Statistics.rand(rng::AbstractRNG, f::DegeneratePosterior, x::AbstractVector)
	w = f.w_mean
	X = weight_form(x)
	X * (f.w_prec.U \ randn(rng, length(x)))
end

# ╔═╡ 04f817b3-ac62-4597-9969-1232cb416739
md"We can compare the results of this optimized implementation with the standard posterior implementation to ensure that the two agree on the output."

# ╔═╡ 6676671e-e5a6-46d7-b2be-ac56a1038b77
x = rand(2, 2000);

# ╔═╡ 837d7b11-5517-41bc-af12-6dc839645701
y = sin.(norm.(eachcol(x)));

# ╔═╡ 53bcaf58-abbf-4017-9f61-5e22513d4214
kern = FiniteBasis();

# ╔═╡ 87e43518-5c04-4a8a-8f32-10ff1dbff759
f = GP(kern);

# ╔═╡ 55a55d3b-0756-4078-ad2e-aa5a728638ee
fx = f(x, 0.001);

# ╔═╡ 57956b60-9623-42bd-9771-52b4e8d768bc
x2 = ColVecs(rand(2, 2000));

# ╔═╡ 4429526e-8639-47d5-9c67-4844fd38eabc
opt_m, opt_C = @btime mean_and_cov(posterior($fx, $y)($x2));

# ╔═╡ e3ceb531-9ccf-4b1c-a1f3-88d2a6fbbda9
md"To compare against the implementation that uses a function-space perspective, we'll use a bit of a hack: by adding a `ZeroKernel` to our `FiniteBasis` kernel, we get a kernel for which our custom `posterior` method won't be called."

# ╔═╡ c195a0d2-de53-4f9d-962e-56642e4cd01a
fx2 = GP(kern + ZeroKernel())(x, 0.001);

# ╔═╡ 84638922-0fd7-495f-bb2f-a4799596432f
m, C = @btime mean_and_cov(posterior($fx2, $y)($x2));

# ╔═╡ df24f649-f85d-41da-9c35-92eb4a723510
max(maximum(abs.(opt_C .- C)), maximum(abs.(opt_m .- m)))

# ╔═╡ 4e647fbe-f05e-409a-be3b-0ce2b7806aa5
md"Our optimized technique produces the same results!"

# ╔═╡ 2890892f-b8dd-4784-a273-a4aa79549523
md"""
## Random Fourier Features
One application of this technique is the *Random Fourier Features* approximation. By Bochner's theorem, every kernel of the form $k(x,y) = f(x-y)$ for some $f$ can be expressed in the Fourier basis as $f(x-y) = E e^{i\omega (x-y)}$, where the distribution from which $\omega$ is sampled determines the kernel. A Monte Carlo estimate of this expectation is just $\sum_{w_j} e^{i w_j x}e^{-i w_j y}$, which is an inner product of features of the form $\phi_j(x) = e^{i w_j x}$. With some algebraic simplifications (see [here](https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/#a4-alternative-random-fourier-features) for a good derivation) we can ignore the imaginary parts and express this as $\phi_j(x)=(\cos(w_j x), \sin(w_j x))$.
"""

# ╔═╡ 3c9277e0-64d0-4c7e-a416-34b5f71e2056
begin
struct RandomFourierFeature
	ws::Vector{Float64}
end
RandomFourierFeature(kern::SqExponentialKernel, k::Int) = RandomFourierFeature(randn(k))
function (f::RandomFourierFeature)(x)
	Float64[cos.(f.ws .* x); sin.(f.ws .* x)] .* sqrt(2/length(f.ws))
end
end

# ╔═╡ 84aebce9-110d-46f7-9c52-f976d57c9c3c
begin
FFApprox(kern::Kernel, k::Int) = FiniteBasis() ∘ FunctionTransform(RandomFourierFeature(kern, k))
FFApprox(rng::AbstractRNG, kern::Kernel, k::Int) = FiniteBasis() ∘ FunctionTransform(RandomFourierFeature(rng, kern, k))
end

# ╔═╡ f2e36526-aa4e-41bd-88ea-2bf5b172c1c4
md"To support other spectral densities besides the RBF, we could add constructors for `RandomFourierFeature`."

# ╔═╡ 6b42b8a3-81c8-4dc6-a640-c2b9a78de284
rbf = SqExponentialKernel();

# ╔═╡ 20c58a01-f98d-4cf0-9575-e544f605fe1e
flat_x = rand(2000);

# ╔═╡ 36b6d708-7a06-4954-8dd7-4d49278057b7
flat_x2 = rand(100);

# ╔═╡ 9529f9d1-a905-40e3-bc68-428d85d24fcd
ffkern = FFApprox(rbf, 100);

# ╔═╡ a2df8f7e-9077-4150-b4dc-322d3b9591db
ff_m, ff_C = mean_and_cov(posterior(GP(ffkern)(flat_x, 0.001), y)(flat_x2));

# ╔═╡ 608c630e-6af9-4199-8121-72cd1eea0c6b
m2, C2 = mean_and_cov(posterior(GP(rbf)(flat_x, 0.001), y)(flat_x2));

# ╔═╡ 573de207-995d-46b4-bd4d-dd792b20924c
max(maximum(abs.(m2 .- ff_m)), maximum(abs.(C2 .- ff_C)))

# ╔═╡ 2180ee59-779b-491c-ae8b-b87f2f4eb530
md"Even with only 100 samples, we get a pretty close approximation!"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractGPs = "99985d1d-32ba-4be9-9821-2ec096f28918"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
KernelFunctions = "ec8451be-7e33-11e9-00cf-bbf324bd1392"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractGPs = "~0.5.19"
BenchmarkTools = "~1.4.0"
KernelFunctions = "~0.10.63"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "8835ada428dad22a6319da0f1f2c3a3bd4edfe0d"

[[deps.AbstractGPs]]
deps = ["ChainRulesCore", "Distributions", "FillArrays", "IrrationalConstants", "KernelFunctions", "LinearAlgebra", "PDMats", "Random", "RecipesBase", "Reexport", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "8a05cefb7c891378c89576bd4865f34d010c9ece"
uuid = "99985d1d-32ba-4be9-9821-2ec096f28918"
version = "0.5.24"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1f03a9fa24271160ed7e73051fba3c1a759b53f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.4.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.KernelFunctions]]
deps = ["ChainRulesCore", "Compat", "CompositionsBase", "Distances", "FillArrays", "Functors", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Random", "Requires", "SpecialFunctions", "Statistics", "StatsBase", "TensorCore", "Test", "ZygoteRules"]
git-tree-sha1 = "0b8ef8b51580b0d87d0b7a5233bb8ea6d948feb4"
uuid = "ec8451be-7e33-11e9-00cf-bbf324bd1392"
version = "0.10.65"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "b81c5035922cc89c2d9523afc6c54be512411466"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.5"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╟─b18c628c-cced-11ee-0033-51bdcc63c29c
# ╠═e1e011e8-3e91-4317-a081-9a23f39349c8
# ╠═ca1fa3bc-c6a8-400a-93ed-850821f57b1f
# ╠═3a04e978-770c-45d1-be1c-5cc1c1925eb2
# ╠═660c7753-ec86-4f36-92b5-eec16d17dbb4
# ╟─f06e2a85-e77e-492b-ae4d-d12d0a2809e8
# ╠═a8a1da9e-78e1-465a-83c9-7d23cb77d926
# ╠═26b76664-6f17-4e4e-9009-fe6ac063c323
# ╟─8c973260-b0a0-49fe-9b18-7dd8ee6d467b
# ╟─8770765c-e519-4cd7-9eed-da2b50190895
# ╠═38ccb850-2091-4914-a7fa-0fdfe0e64375
# ╠═2646ce1f-bd5c-4ea0-a9fb-c6f786984240
# ╠═60390567-e70f-463b-a0be-480d1b9c3198
# ╠═841129b7-0d84-4f6b-9ae0-edf2a6cb8661
# ╠═8c2d7c8b-8d9e-44fe-88b9-00fd3e3aee14
# ╠═9fe52847-ae03-461e-a5f9-fb95adc63cb4
# ╠═b623703a-137c-407b-9094-b19abe3593b5
# ╠═0f937650-77db-43cb-859a-a0f27dcc464d
# ╠═8a7d7a98-278c-4558-90fd-6a382c022236
# ╟─04f817b3-ac62-4597-9969-1232cb416739
# ╠═6676671e-e5a6-46d7-b2be-ac56a1038b77
# ╠═837d7b11-5517-41bc-af12-6dc839645701
# ╠═53bcaf58-abbf-4017-9f61-5e22513d4214
# ╠═87e43518-5c04-4a8a-8f32-10ff1dbff759
# ╠═55a55d3b-0756-4078-ad2e-aa5a728638ee
# ╠═57956b60-9623-42bd-9771-52b4e8d768bc
# ╠═4140ca9d-2530-43a9-aee8-73f7fb0770ec
# ╠═4429526e-8639-47d5-9c67-4844fd38eabc
# ╟─e3ceb531-9ccf-4b1c-a1f3-88d2a6fbbda9
# ╠═c195a0d2-de53-4f9d-962e-56642e4cd01a
# ╠═84638922-0fd7-495f-bb2f-a4799596432f
# ╠═df24f649-f85d-41da-9c35-92eb4a723510
# ╟─4e647fbe-f05e-409a-be3b-0ce2b7806aa5
# ╟─2890892f-b8dd-4784-a273-a4aa79549523
# ╠═3c9277e0-64d0-4c7e-a416-34b5f71e2056
# ╠═84aebce9-110d-46f7-9c52-f976d57c9c3c
# ╟─f2e36526-aa4e-41bd-88ea-2bf5b172c1c4
# ╠═6b42b8a3-81c8-4dc6-a640-c2b9a78de284
# ╠═20c58a01-f98d-4cf0-9575-e544f605fe1e
# ╠═36b6d708-7a06-4954-8dd7-4d49278057b7
# ╠═9529f9d1-a905-40e3-bc68-428d85d24fcd
# ╠═a2df8f7e-9077-4150-b4dc-322d3b9591db
# ╠═608c630e-6af9-4199-8121-72cd1eea0c6b
# ╠═573de207-995d-46b4-bd4d-dd792b20924c
# ╟─2180ee59-779b-491c-ae8b-b87f2f4eb530
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

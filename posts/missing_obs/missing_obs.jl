### A Pluto.jl notebook ###
# v0.20.20

#> [frontmatter]
#> title = "Fitting Gaussians with Missing Observations"
#> date = "2026-02-05"
#> categories = "statistics"

using Markdown
using InteractiveUtils

# ╔═╡ ca6ec2ff-edd4-4a36-ab05-50e4feeca9de
using LinearAlgebra, StatsBase, StatsModels, LogExpFunctions, Random, Distributions

# ╔═╡ 0f40abb5-1d90-4e82-aab5-4d5968f18225
md"Say you want to fit a multivariate Normal distribution to some data." 

# ╔═╡ 0dd03284-30fc-488e-b5c7-29ce2aaf147d
Random.seed!(0);

# ╔═╡ 7446ba37-182f-4f50-a8f9-126c5c8e06b2
Σ = rand(LKJ(3, 1))

# ╔═╡ fd97a1c3-ad63-44e3-98a5-3bd8a2ff9303
μ = rand(3)

# ╔═╡ 130544a1-9433-4cd7-bcbf-767332712baf
X = (μ .+ sqrt(Σ) * randn(3, 300))'

# ╔═╡ 6b13d16f-d918-45ae-aeca-b71b24ec28d1
md"""
The obvious way to do this is with maximum likelihood estimation. 
In natural parameter form, the log likelihood is given by 

```math
\sum_i x_i^T \Lambda \mu - \frac{1}{2} \text{Tr}(\Lambda x_i x_i^T) - N A(\eta)
```
Derive wrt the natural parameters $\eta = (\Lambda, \Lambda \mu)$ to get 

```math
\begin{align*}
\frac{1}{N}\sum_i x_i &= E[x] \\
\frac{1}{N}\sum_i x_i x_i^T &= E[xx^T]
\end{align*}
```
"""

# ╔═╡ a5def34f-484b-44a7-8c05-0e3308af24f5
md"""
Easy. But now, how might you do this if some subset of the data is missing? We'll assume that the data isn't missing completely at random, so that our observed marginals aren't unbiased estimators of the true marginals.
"""

# ╔═╡ 2a6a9537-5794-46db-a4f6-b7fc9270bd00
begin
missing_mask = rand(size(X, 1)) .< logistic.(-1 .+ X * -rand(size(X, 2), 3))
Xm = convert(Matrix{Union{Float64, Missing}}, X)
Xm[missing_mask] .= missing
Xm
end

# ╔═╡ 41f78cb4-5dc7-4dba-afcf-2cc0ce7504ce
sum(ismissing.(Xm); dims=1) ./ size(Xm, 1)

# ╔═╡ 54c5104f-ecfc-49cf-9f2f-0d53fd1ee562
md"""
We're trying to find a maximum likelihood estimate in the presence of latent variables (in this case, all the unobserved values). It's a classic situation for the EM algorithm!

At each step of EM, we want to find parameters $\mu, \Sigma$ that maximize the expected log likelihood, where the expectation is taken with respect to $p(x_\text{unobserved} | x_\text{observed})$. That's

```math
\begin{align*}
E\left[\sum_i x_i^T \Lambda \mu - \frac{1}{2} \text{Tr}(\Lambda x_i x_i^T) - N A(\eta)\right] = \\
\sum_i E[x_i]^T \Lambda \mu - \frac{1}{2} \text{Tr}(\Lambda E[x_i x_i^T]) - N A(\eta)
\end{align*}
```

Take the gradient as before to get

```math
\begin{align*}
\mu &= \frac{1}{N}\sum_i E[y_i] \\
\Sigma &= \frac{1}{N}\sum_i E[y_i y_i^T]
\end{align*}
```
"""

# ╔═╡ 6da2c35a-64b4-4160-9b5f-97e96d367ad2
md"""
It remains to find out what $E[x_i]$ and $E[x_i x_i^T]$ are. 
For the components of $x_i$ that are observed (call this subvector $x_o$), $E[x_o] = x_o$. For the components $x_u$ that are not observed, the Gaussian conditioning formula gives $x_u = μ_u + Σ_{uo} Σ_{oo}^{-1}(y_o - μ_o)$
"""

# ╔═╡ e0b54feb-5ada-48e6-a9ca-86530eb3540a
function E_m1(x, μ, Σ)
	x2 = copy(x)
	u = ismissing.(x)
	o = .!u
	x2[u] .= μ[u] + Σ[u, o] * (Σ[o, o] \ (x[o] - μ[o]))
	x2
end

# ╔═╡ ee60d165-d27d-42d9-8c4a-b1733e0cab62
md"""
The second moment breaks down similarly. $E[x_i x_i^T]$ can be thought of (up to permutation) as a block matrix: 

```math
\begin{bmatrix}
x_o x_o^T & x_o x_u^T \\
x_u x_o^T & x_u x_u^T
\end{bmatrix}
```

As $x_0$ and $x_o x_o^T$ are known, that just leaves $E[x_o x_u^T] = x_o \mu_u^T$ and $E[x_u x_u^T] = \text{Var}(x_u | x_o) + \mu_u \mu_u^T$. The formula for conditional Gaussians tell us $\text{Var}(x_u) = \Sigma_{uu} - \Sigma_{uo} \Sigma_{oo}^{-1} \Sigma_{ou}$.
"""

# ╔═╡ 1a56ea0a-1382-4450-b2ce-024ffd75dcdc
function E_m2(x, μ, Σ)
	x2 = Matrix{Float64}(undef, size(Σ))
	u = ismissing.(x)
	o = .!u
	x2[o,o] .= x[o] * x[o]'
	x2[o, u] .= x[o] * μ[u]'
	x2[u, o] .= x2[o, u]'
	x2[u,u] = Σ[u,u] - Σ[u, o] * (Σ[o,o] \ Σ[o, u]) + μ[u]*μ[u]'
	x2
end

# ╔═╡ eaf91cee-dc78-4721-95d5-e0830e2823e1
md"Let's put it all together!"

# ╔═╡ 00a57a4c-02f3-11f1-a3a4-619e50b333f2
function em_step(X, μ, Σ)
	μ = mean(E_m1(x, μ, Σ) for x in eachrow(X))
	m2 = mean(E_m2(x, μ, Σ) for x in eachrow(X))
	(μ, m2 - μ * μ')
end

# ╔═╡ 953ec82f-9c35-49c4-bdc4-bf40b32ff0c9
function em_alg(X, μ, Σ)
	for _ in 1:100
		μ2, Σ2 = em_step(X, μ, Σ)
		δ = max(maximum(abs.(μ - μ2)), maximum(abs.(Σ2 - Σ)))
		if δ < 1e-5
			return (μ, Σ)
		end
		μ, Σ = (μ2, Σ2)
	end
end

# ╔═╡ 194ce015-6d10-4f09-83c6-122a33572976
μ_guess, Σ_guess = em_alg(Xm, randn(3), 1.0I(3))

# ╔═╡ 66bae4e6-cbc2-4dca-a6e4-8121b475c0d8
(μ, Σ)

# ╔═╡ 0b8f3242-4d8e-465b-81fa-597dc6981a69
md"Our guess is pretty close to the true value!"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsModels = "3eaba693-59b7-5ba5-a881-562e759f1c8d"

[compat]
Distributions = "~0.25.123"
LogExpFunctions = "~0.3.29"
StatsBase = "~0.34.10"
StatsModels = "~0.7.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "e6988e82e5dd4bcd5838f8efc4b654d6b63f22c8"

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

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "fbcc7610f6d8348428f722ecbe0e6cfe22e672c6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.123"

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
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

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

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

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
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
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

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

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

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

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
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "b12d37d25a2378f01abba02591cfd39a6cc4936f"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.8"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╟─0f40abb5-1d90-4e82-aab5-4d5968f18225
# ╠═ca6ec2ff-edd4-4a36-ab05-50e4feeca9de
# ╠═0dd03284-30fc-488e-b5c7-29ce2aaf147d
# ╠═7446ba37-182f-4f50-a8f9-126c5c8e06b2
# ╠═fd97a1c3-ad63-44e3-98a5-3bd8a2ff9303
# ╠═130544a1-9433-4cd7-bcbf-767332712baf
# ╟─6b13d16f-d918-45ae-aeca-b71b24ec28d1
# ╟─a5def34f-484b-44a7-8c05-0e3308af24f5
# ╠═2a6a9537-5794-46db-a4f6-b7fc9270bd00
# ╠═41f78cb4-5dc7-4dba-afcf-2cc0ce7504ce
# ╟─54c5104f-ecfc-49cf-9f2f-0d53fd1ee562
# ╟─6da2c35a-64b4-4160-9b5f-97e96d367ad2
# ╠═e0b54feb-5ada-48e6-a9ca-86530eb3540a
# ╟─ee60d165-d27d-42d9-8c4a-b1733e0cab62
# ╠═1a56ea0a-1382-4450-b2ce-024ffd75dcdc
# ╟─eaf91cee-dc78-4721-95d5-e0830e2823e1
# ╠═00a57a4c-02f3-11f1-a3a4-619e50b333f2
# ╠═953ec82f-9c35-49c4-bdc4-bf40b32ff0c9
# ╠═194ce015-6d10-4f09-83c6-122a33572976
# ╠═66bae4e6-cbc2-4dca-a6e4-8121b475c0d8
# ╟─0b8f3242-4d8e-465b-81fa-597dc6981a69
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

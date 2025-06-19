#import "@preview/polylux:0.4.0": *
#import "@preview/metropolis-polylux:0.1.0" as metropolis
#import metropolis: new-section, focus

#show: metropolis.setup.with(
  text-font: "Luciole",
  math-font: "Luciole Math",
  code-font: "Fira Code",
  text-size: 23pt,
  // footer: [Sparse automatic differentiation -- G. Dalle], // defaults to none
)
#slide[
  #set page(header: none, footer: none, margin: 3em, fill: white)

  #text(size: 1.3em)[
    *Sparser, better, faster, stronger*
  ]

  Automatic differentiation with a lot of zeros

  #metropolis.divider

  #set text(size: .8em, weight: "light")
  *Guillaume Dalle*#footnote(numbering: "*")[#text(size: 14pt)[joint work with Adrian Hill, Alexis Montoison and Assefaw Gebremedhin]] -- LVMT, École des Ponts (#link("https://gdalle.github.io")[`gdalle.github.io`])

  Laboratoire Jean Kuntzmann, 19.06.2025
]

#slide[
  = Agenda

  #metropolis.outline
]

#new-section[Motivation]

#slide[
  = Newton's method

  #toolbox.side-by-side[
    *Root-finding*

    Solve $F(x) = 0$ by iterating

    $ x_(t+1) = x_t - underbrace([partial F(x_t)], "Jacobian")^(-1) F(x_t) $
  ][
    *Optimization*

    Solve $min_x f(x)$ by iterating

    $ x_(t+1) = x_t - underbrace([nabla^2 f(x_t)], "Hessian")^(-1) nabla f(x_t) $
  ]

  *Linear system* involving a derivative matrix $A$.
]

#slide[
  = Implicit differentiation

  Differentiate $x mapsto y(x)$ knowing *conditions* $c(x, y(x)) = 0$.

  Applications: fixed-point iterations, optimization problems.

  Implicit function theorem:

  $ partial / (partial x) c(x, y(x)) + partial / (partial y) c(x, y(x)) dot partial y(x) = 0 $
  $ partial y(x) = -underbrace([partial / (partial y) c(x, y(x))], "Jacobian")^(-1) partial / (partial x) c(x, y(x)) $

  *Linear system* involving a derivative matrix $A$.
]

#slide[
  = Linear systems of equations

  How to solve $A u = v$?

  #toolbox.side-by-side[
    *Direct method* (LU, Cholesky)

    1. Decompose the matrix $A$.
    2. Get an exact solution by substitution.

    Requires storing $A$ explicitly.
  ][
    *Iterative method* (CG, GMRES)

    1. Rephrase as $min_u norm(A u - v)^2$.
    2. Get an approximate solution.

    Only requires matrix-vector products $u mapsto A u$.
  ]

]

#slide[
  = Conventional wisdom

  - Jacobian and Hessian matrices are *too large* to compute or store

  - We can only access linear maps $u mapsto A u$ (JVPs, VJPs, HVPs)

  - Linear systems $A^(-1) v$ must be solved with *iterative methods*

  - Downsides: each iteration is expensive, convergence is tricky
]

#slide[
  = The benefits of sparsity

  - Jacobian and Hessian matrices have *mostly zero coefficients*

  - We can compute and store $A$ explicitly

  - Linear systems $A^(-1) v$ can be solved with iterative *or direct* methods

  - Upsides: faster iterations or exact solves, efficient linear algebra
]

#new-section[Automatic differentiation]

#slide[
  = Differentiation

  Given $f : bb(R)^n arrow bb(R)^m$, its *differential* $partial f(x)$ is the *linear map* which best approximates $f$ around $x$:

  $ f(x + u) = f(x) + partial f(x)[u] + o(u) $

  It can be represented by the *Jacobian matrix*, which I will denote by $J = partial_"mat" f(x)$ instead.
]

#slide[
  = Numeric differentiation

  #align(center)[
    #table(
      columns: (auto, auto),
      inset: 10pt,
      align: center,
      table.header(
        [*input*],
        [*output*],
      ),

      [program computing the function $ x mapsto f(x) $],
      [approximation of the differential with the same program $ partial f(x)[u] approx (f(x + epsilon u) - f(x)) / epsilon $],
    )
  ]
]

#slide[
  = Automatic / algorithmic differentiation

  #align(center)[
    #table(
      columns: (auto, auto),
      inset: 10pt,
      align: center,
      table.header(
        [*input*],
        [*output*],
      ),

      [program computing the function $ x mapsto f(x) $],
      [new program computing the exact differential $ (x, u) mapsto partial f(x)[u] space "or" partial f(x)^*[u] $],
    )
  ]

]

#slide[
  = AD under the hood

  Two ingredients only:

  1. hardcode basic derivatives ($+$, $times$, $exp$, $log$, ...)
  2. handle composition $f = g compose h$
]

#slide[
  = Composition

  For a function $f = g compose h$, the *chain rule* gives its differential:

  $
    "standard" & wide & partial f(x) & = partial g (h(x)) compose partial h (x) \
    "adjoint" & wide & partial f(x)^* & = partial h (x)^* compose partial g (h(x))^*
  $

  These linear maps apply as follows:

  $
    "forward" & wide & partial f(x): & u stretch(arrow.r.bar)^(partial h(x)) v stretch(arrow.r.bar)^(partial g( h(x) )) w \
    "reverse" & wide & partial f(x)^*: & u stretch(arrow.l.bar)_(partial h(x)^*) v stretch(arrow.l.bar)_(partial g( h(x) )^*) w \
  $
]

#slide[
  = Why linear maps?

  The chain rule has a matrix equivalent:

  $
    partial_"mat" (g compose h)(x) = partial_"mat" g(h(x)) dot partial_"mat" h(x) \
    partial_"mat" (g compose h)(x)^T = partial_"mat" h(x)^T dot partial_"mat" g(h(x))^T
  $

  Working with linear maps avoids allocation and manipulation of *intermediate Jacobian matrices*.

  Essential for neural networks (scalar output but vector encodings)!
]

#slide[
  = Two modes

  Forward-mode AD computes Jacobian-Vector Products (*JVPs*) = "pushforward" of an *input perturbation*:

  $ u mapsto partial f(x)[u] = J u space.quad "with" space.quad J = partial_"mat" f(x) $

  Reverse-mode AD computes Vector-Jacobian Products (*VJPs*) = "pullback" of an *output sensitivity*:

  $ v mapsto partial f(x)^* [v] = J^T v = v^T J space.quad "with" space.quad J = partial_"mat" f(x) $

]

#slide[
  #show: focus
  Theorem (Baur-Strassen): cost of 1 JVP or VJP $prop$ cost of 1 function evaluation
]

#slide[
  = What about gradients?

  #toolbox.side-by-side[
    Reverse mode computes *gradients for roughly the same cost* as the function itself:

    $ nabla f(x) = partial f(x)^* [1] $

    Makes deep learning possible.

    The devil is in the details: higher memory footprint.
  ][
    #set align(center)
    #image("../assets/img/book/reverse_memory.png")
    #cite(<blondelElementsDifferentiableProgramming2024>, form: "prose")
  ]
]

#slide[
  = What about second order?

  The Hessian matrix is the Jacobian matrix of the gradient function.

  A Hessian-Vector Product (HVP) can be computed as the JVP of a VJP, in *forward-over-reverse mode*#footnote(cite(<pearlmutterFastExactMultiplication1994>, form: "prose")):

  $ nabla^2 f(x)[v] = partial (nabla f)(x)[v] = partial (partial^* f(x)[1]) [v] $

]


#slide[
  = Pocket AD

  #only(1)[
    ```julia
    # Basic rules

    using LinearAlgebra

    A, b = rand(2, 3), rand(2)
    residuals(x) = A * x - b
    ∂(::typeof(residuals)) = x -> (u -> A * u)  # ℝ³ → ℝ²
    ∂ᵀ(::typeof(residuals)) = x -> (v -> adjoint(A) * v)  # ℝ² → ℝ³

    sqnorm(r) = sum(abs2, r)
    ∂(::typeof(sqnorm)) = r -> (v -> dot(2r, v))  # ℝ² → ℝ
    ∂ᵀ(::typeof(sqnorm)) = r -> (w -> 2r .* w)  # ℝ → ℝ²
    ```
  ]

  #only(2)[
    ```julia
    # Composition

    function ∂(f::ComposedFunction)
        g, h = f.outer, f.inner
        return x -> ∂(g)(h(x)) ∘ ∂(h)(x)
    end

    function ∂ᵀ(f::ComposedFunction)
        g, h = f.outer, f.inner
        return x -> ∂ᵀ(h)(x) ∘ ∂ᵀ(g)(h(x))
    end
    ```
  ]

  #only(3)[
    ```julia
    julia> import ForwardDiff as FD, Zygote

    julia> f = sqnorm ∘ residuals;

    julia> x, u = rand(3), [1, 0, 0];
    ```
    #set text(size: 16pt)

    #v(10%)

    #toolbox.side-by-side[
      ```julia
      julia> ∂(f)(x)(u)  # directional derivative
      0.8691056836969242

      julia> ∂ᵀ(f)(x)(1)  # gradient
      3-element Vector{Float64}:
       0.8691056836969242
       0.9973491983376236
       0.5768822265195823
      ```
    ][
      ```julia
      julia> FD.derivative(t -> f(x + t * u), 0)
      0.8691056836969242

      julia> Zygote.gradient(f, x)[1]
      3-element Vector{Float64}:
       0.8691056836969242
       0.9973491983376236
       0.5768822265195823
      ```
    ]
  ]
]

#new-section[Leveraging sparsity]

#slide[
  = From maps to matrices
  To compute the Jacobian matrix $J$ of a composition $f: bb(R)^m arrow.long bb(R)^n$:
  - #strike[product of intermediate Jacobian matrices]
  - reconstruction from several JVPs or VJPs

  #align(center)[
    #table(
      columns: (auto, auto, auto),
      inset: 10pt,
      align: center,
      table.header(
        [],
        [*forward mode*],
        [*reverse mode*],
      ),

      [idea], [1 JVP gives 1 column], [1 VJP gives 1 row],
      [formula], [$ J_(dot, j) = partial f(x)[e_j] $], [$ J_(i, dot) = partial f(x)^*[e_i] $],
      [cost], [$n$ JVPs (input dimension)], [$m$ VJPs (output dimension)],
    )
  ]
]

#slide[
  = Using fewer products

  When the Jacobian is sparse, we can compute it faster#footnote(cite(<curtisEstimationSparseJacobian1974>, form: "prose")).

  If columns $j_1, dots, j_k$ of $J$ are structurally *orthogonal* (their nonzeros never overlap), we deduce them all from a single JVP:
  $ J_(j_1) + dots + J_(j_k) = partial f(x)[e_(j_1) + dots + e_(j_k)] $

  Once we have grouped columns, sparse AD has two steps:

  3. one JVP for each group $c = {j_1, dots, j_k}$
  4. decompression into individual columns $j_1, dots, j_k$
]

#slide[
  = Two preliminary steps

  When grouping columns, we want to

  - guarantee orthogonality (correctness) $arrow.r.double$ pattern detection
  - form the smallest number of groups (efficiency) $arrow.r.double$ coloring

  #align(center)[
    #table(
      columns: (auto, auto),
      inset: 10pt,
      align: left,
      table.header(
        [*preparation*],
        [*execution*],
      ),

      [
        1. pattern detection \
        2. coloring
      ],
      [
        3. matrix-vector products \
        4. decompression
      ],
    )
  ]

  The preparation phase can be *amortized* across several inputs.
]

#slide[
  = The gist in one slide

  #set align(center)
  #image("../assets/img/paper/fig1.png", width: 90%)
  #cite(<hillSparserBetterFaster2025>, form: "prose")
]

#slide[
  = Tracing dependencies in the computation graph

  #toolbox.side-by-side[
    #set align(center)
    #image("../assets/img/blog/compute_graph.png", width: 100%)
    #cite(<hillIllustratedGuideAutomatic2025>, form: "prose")
  ][
    Computation graph for $ y_1 &= x_1 x_2 + "sign"(x_3) \ y_2 &= "sign"(x_3) times (x_4 / 2) $

    Its Jacobian matrix will have 3 nonzero coefficients:

    $
      mat(
        1, 1, 0, 0;
        0, 0, 0, 1;
      )
    $
  ]
]

#slide[
  = Pocket pattern detection

  #only(1)[
    ```julia
    import Base: +, *, /, sign

    struct Tracer
    	indices::Set{Int}
    end

    Tracer() = Tracer(Set{Int}())

    +(a::Tracer, b::Tracer) = Tracer(a.indices ∪ b.indices)
    *(a::Tracer, b::Tracer) = Tracer(a.indices ∪ b.indices)
    /(a::Tracer, b::Real) = Tracer(a.indices)
    sign(a::Tracer) = Tracer()  # zero derivatives
    ```
  ]
  #only(2)[

    ```julia
    julia> f(x) = [x[1] * x[2] * sign(x[3]), sign(x[3]) * x[4] / 2];

    julia> x = Tracer.(Set.([1, 2, 3, 4]))
    4-element Vector{Tracer}:
     Tracer(Set([1]))
     Tracer(Set([2]))
     Tracer(Set([3]))
     Tracer(Set([4]))

    julia> f(x)
    2-element Vector{Tracer}:
     Tracer(Set([2, 1]))
     Tracer(Set([4]))
    ```
  ]
]

#slide[
  = Coloring for Jacobians

  #toolbox.side-by-side[
    *Matrix problem*

    Orthogonal partition of the columns of $A$.

    If $A_(i j_1) != 0$ and $A_(i, j_2) != 0$, then columns $j_1$ and $j_2$ are in different groups $c(j_1) != c(j_2)$
  ][
    *Graph problem*

    Partial distance-2 coloring of a bipartite graph $cal(G) = (cal(I) union cal(J), cal(E))$

    If $(i, j_1) in cal(E)$ and $(i, j_2) in cal(E)$, then vertices $j_1$ and $j_2$ have different colors $c(j_1) != c(j_2)$.
  ]

  These are equivalent#footnote(cite(<gebremedhinWhatColorYour2005>, form: "prose")) if we define the graph representation $ cal(E) = {(i, j) in cal(I) times cal(J): A_(i, j) != 0} $
]

#slide[
  = Coloring for Jacobians, illustrated
  Coloring of intersection graph / distance-2 coloring of bipartite graph

  #set align(center)
  #image("../assets/img/survey/bipartite_column_full.png", width: 90%)
  #cite(<gebremedhinWhatColorYour2005>, form: "prose")
]

#slide[
  = Coloring for Hessians

  What if our matrix has structure, like $A_(i, j) = A_(j, i)$?

  We can compute a slightly different coloring#footnote(cite(<colemanEstimationSparseHessian1984>, form: "prose")) with fewer colors.

  #set align(center)
  #image("../assets/img/coloring_paper/star_coloring.png", width: 50%)
]

#slide[
  = Coloring for bidirectional Jacobians

  What if the columns are not orthogonal enough?

  We can use both rows and columns#footnote(cite(<colemanEfficientComputationSparse1998>, form: "prose")) inside our coloring.

  #set align(center)
  #image("../assets/img/coloring_paper/star_bicoloring.png", width: 50%)
]

#slide[
  = Benefits of bidirectional coloring

  Compute Jacobians with a dense row *and* a dense column, using forward-mode AD + reverse-mode AD.

  #toolbox.side-by-side[
    #set align(center)
    #image("../assets/img/coloring_paper/row.png", width: 50%)
    #image("../assets/img/coloring_paper/col.png", width: 50%)
    Unidirectional
  ][
    #set align(center)
    #toolbox.side-by-side[
      #image("../assets/img/coloring_paper/birow.png", width: 80%)
    ][
      #image("../assets/img/coloring_paper/bicol.png", width: 80%)
    ]
    #image("../assets/img/coloring_paper/bi.png", width: 60%)
    Bidirectional
  ]
]

#slide[
  = From bidirectional to symmetric [new]

  To color the rows and columns of $J$, color the columns of $H = mat(
    0, J;
    J, 0;
  )$

  It sounds simple, but:

  - Some colors may be redundant
  - Detecting these is tightly linked to the two-colored structures
  - Efficient decompression requires lots of preprocessing

  Explanations and benchmarks in #cite(<montoisonRevisitingSparseMatrix2025>, form: "prose")
]

#slide[
  = The sharp bits

  #toolbox.side-by-side[
    *Pattern detection*

    - Local versus global sparsity
    - Control flow
    - Linear and nonlinear interactions
  ][
    *Coloring*

    - Only heuristic algorithms
    - Vertex ordering matters a lot
  ]
]

#new-section[Implementation]

#slide[
  = AD in Python & Julia
  #set align(center)
  #image("../assets/img/juliacon/python_julia_user.png", width: 90%)
]

#slide[
  = Interfaces for experimenting [new]

  Once we have a common syntax, we can do more!

  #toolbox.side-by-side[
    #image("../assets/img/misc/keras.jpg")
    In Python, #link("https://keras.io/")[`Keras`] supports Tensorflow, PyTorch and JAX.
  ][
    #set align(center)
    #image("../assets/img/logo.svg", width: 55%)
    In Julia, 14 AD backends inside #link("https://github.com/JuliaDiff/DifferentiationInterface.jl")[`Differentiationterface.jl`]
  ]

]

#slide[
  = One API to rule them all

  Decouple the scientific libraries from the AD underneath.

  #set align(center)
  #image("../assets/img/di.png", width: 95%)
  #cite(<dalleCommonInterfaceAutomatic2025>, form: "prose")
]

#slide[
  = Previous implementations of sparse AD

  - In low-level programming languages (C, Fortran)
  - In closed-source languages (Matlab)
  - In domain-specific languages (AMPL, CasADi)

  Basically nothing in Python (either in JAX or PyTorch).

  First drafts in Julia for scientific machine learning, but severely limited: single-backend, slow.
]

#slide[
  = A modern sparse AD ecosystem [new]

  Independent packages working together:
  - Step 1: #link("https://github.com/adrhill/SparseConnectivityTracer.jl")[`SparseConnectivityTracer.jl`] #cite(<hillSparserBetterFaster2025>)
  - Steps 2 & 4: #link("https://github.com/gdalle/SparseMatrixColorings.jl")[`SparseMatrixColorings.jl`] #cite(<montoisonRevisitingSparseMatrix2025>)
  - Step 3: #link("https://github.com/JuliaDiff/DifferentiationInterface.jl")[`Differentiationterface.jl`] #cite(<dalleCommonInterfaceAutomatic2025>)

  #align(center)[
    #table(
      columns: (auto, auto, auto, auto),
      inset: 10pt,
      align: left,
      table.header(
        [],
        [*`SCT.jl`*],
        [*`SMC.jl`*],
        [*`DI.jl`*],
      ),

      [lines of code], [5202], [5184], [19980],
      [indirect dependents], [461], [487], [896],
      [downloads / month], [7.8k], [9.7k], [33k],
    )
  ]

  Compatible with generic code!
]

#slide[
  = Impact

  #toolbox.side-by-side[
    Users already include...

    - Scientific computing: #link("https://sciml.ai/")[`SciML`] (Julia's `scipy`)
      - Differential equations
      - Nonlinear solvers
      - Optimization
    - Probabilistic programming: #link("https://turinglang.org/")[`Turing.jl`]
    - Symbolic regression: #link("https://github.com/MilesCranmer/PySR")[`PySR`]

  ][
    Python bindings in construction:
    - #link("https://github.com/gdalle/pysparsematrixcolorings")[`pysparsematrixcolorings`]
    - #link("https://github.com/gdalle/sparsediffax")[`sparsediffax`]
  ]
]

#slide[
  = Live demo

  This is the part where things go sideways.
]

#slide[
  = Perspectives

  - GPU-compatible pattern detection and coloring
  - Pattern detection in JAX with program transformations
  - New, unsuspected applications "just because we can"
]

#slide[
  = Going further

  On general AD:

  - #cite(<baydinAutomaticDifferentiationMachine2018>, form: "prose")
  - #cite(<margossianReviewAutomaticDifferentiation2019>, form: "prose")
  - #cite(<blondelElementsDifferentiableProgramming2024>, form: "prose")

  On sparse AD:

  - #cite(<gebremedhinWhatColorYour2005>, form: "prose")
  - #cite(<griewankEvaluatingDerivativesPrinciples2008>, form: "prose")
  - #cite(<hillIllustratedGuideAutomatic2025>, form: "prose")

]

#slide[
  = Bibliography

  #text(size: 12pt)[
    #bibliography("AD.bib", title: none, style: "apa")
  ]
]

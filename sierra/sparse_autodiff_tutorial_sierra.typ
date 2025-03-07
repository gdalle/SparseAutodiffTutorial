#import "@preview/polylux:0.4.0": *
#import "@preview/metropolis-polylux:0.1.0" as metropolis
#import metropolis: new-section, focus

#show: metropolis.setup.with(
  text-font: "Fira Sans",
  math-font: "New Computer Modern Math",
  code-font: "Fira Code",
  text-size: 23pt,
  // footer: [Sparse automatic differentiation -- G. Dalle], // defaults to none
)
#slide[
  #set page(header: none, footer: none, margin: 3em, fill: white)

  #text(size: 1.3em)[
    *Sparse automatic differentiation*
  ]

  From theory to practice

  #metropolis.divider

  #set text(size: .8em, weight: "light")
  *Guillaume Dalle*#footnote(numbering: "*")[#text(size: 14pt)[joint work with Adrian Hill & Alexis Montoison]] -- LVMT, École des Ponts (#link("https://gdalle.github.io")[`gdalle.github.io`])

  Inria Paris, 07.03.2025
]

#slide[
  = Agenda

  #metropolis.outline
]

#new-section[Introduction]

#slide[
  = Newton's method

  #toolbox.side-by-side[
    *Root-finding*

    Solve $F(x) = 0$ by iterating

    $ x_(t+1) = x_t - underbrace([partial F(x_t)], "jacobian")^(-1) F(x_t) $
  ][
    *Optimization*

    Solve $min f(x)$ by iterating

    $ x_(t+1) = x_t - underbrace([nabla^2 f(x_t)], "hessian")^(-1) nabla f(x_t) $
  ]

  Linear system involving a derivative matrix $A$.
]

#slide[
  = Implicit differentiation

  Differentiate $x arrow y(x)$ knowing *optimality conditions* $c(x, y(x)) = 0$.

  Applications: fixed-point iterations, optimization problems.

  Implicit function theorem#footnote[#cite(<blondelEfficientModularImplicit2022>, form: "prose")]

  $ partial_1 c(x, y(x)) + partial_2 c(x, y(x)) dot partial y(x) = 0 $
  $ partial y(x) = -underbrace([partial_2 c(x, y(x))], "jacobian")^(-1) partial_1 c(x, y(x)) $

  Linear system involving a derivative matrix $A$.
]

#slide[
  = Conventional wisdom

  - Jacobian and Hessian matrices are too large to compute or store

  - We can only access lazy maps $u arrow.long.bar A u$ (JVPs, VJPs, HVPs#footnote(cite(<dagreouHowComputeHessianvector2024>, form: "prose")))

  - Linear systems $A^(-1) v$ must be solved with iterative methods

  - Downsides: each iteration is expensive, convergence is tricky
]

#slide[
  = The benefits of sparsity

  - Jacobian and Hessian matrices have mostly zero coefficients

  - We can compute and store $A$ explicitly

  - Linear systems $A^(-1) v$ can be solved with iterative or direct methods

  - Upsides: faster iterations, or even exact solves
]

#new-section[Automatic differentiation]


#slide[
  = Pocket AD

  #only(1)[
    ```julia
    import Base: +, *  # overload standard operators

    struct Dual
        val::Float64
        der::Float64
    end

    +(x::Dual, y::Dual) = Dual(x.val + y.val, x.der + y.der)
    *(x::Dual, y::Dual) = Dual(x.val * y.val, x.der*y.val + x.val*y.der)
    +(x, y::Dual) = Dual(x, 0) + y
    *(x, y::Dual) = Dual(x, 0) * y
    ```
  ]

  #only(2)[

    Does it work?

    ```julia
    julia> f(x) = 1 + 2 * x + 3 * x * x;

    julia> f(4)
    57

    julia> f(Dual(4, 1))  # exact derivative
    Dual(57.0, 26.0)

    julia> (f(4 + 1e-5) - f(4)) / 1e-5  # approximate derivative
    26.000029998840542
    ```
  ]
]

#slide[
  = What is AD?

  #align(center)[
    #table(
      columns: (auto, auto),
      inset: 10pt,
      align: center,
      table.header(
        [*input*],
        [*output*],
      ),

      [program to compute the function $ x arrow.r.bar.long f(x) $],
      [program to compute the differential $ x arrow.r.bar.long partial f(x) $ which is a linear map $u arrow.r.bar partial f(x) [u]$],
    )
  ]

  Two ingredients only:

  1. hardcode basic derivatives ($+$, $times$, $exp$, $log$, ...)
  2. handle compositions $f = g compose h$
]

#slide[
  = Composition

  For a function $f = g compose h$, the chain rule gives

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
  = Forward chain rule, illustrated

  #figure(
    image("../assets/img/book/forward.png", width: 100%),
    caption: cite(<blondelElementsDifferentiableProgramming2024>, form: "prose"),
  )
]

#slide[
  = Reverse chain rule, illustrated

  #figure(
    image("../assets/img/book/reverse.png", width: 100%),
    caption: cite(<blondelElementsDifferentiableProgramming2024>, form: "prose"),
  )
]

#slide[
  = Two modes

  Forward-mode AD computes Jacobian-Vector Products:

  $ u arrow.r.bar.long partial f(x)[u] = J u $

  Reverse-mode AD computes Vector-Jacobian Products:

  $
    w arrow.r.bar.long quad & partial f(x)^* [w] = w^* J
  $

  No need to materialize intermediate Jacobian matrices!
]

#slide[
  #show: focus
  Theorem: cost of 1 JVP or VJP \ $prop$ cost of 1 function evaluation
]

#slide[
  = Interpretations

  #toolbox.side-by-side[
    - Forward mode: "pushforward" of an input perturbation

    - Reverse mode: "pullback" of an output sensitivity

    Reverse mode gives gradients for roughly the same cost as the function itself:

    $ nabla f(x) = partial f(x)^* [1] $
  ][
    #figure(
      image("../assets/img/book/reverse_memory.png"),
      caption: [The devil is in the details #cite(<blondelElementsDifferentiableProgramming2024>)],
    )
  ]

]


#slide[
  = Pocket AD, chain rule version

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

    julia> x, Δx = rand(3), [1, 0, 0];
    ```
    #set text(size: 16pt)

#v(10%)

    #toolbox.side-by-side[
      ```julia
      julia> ∂(f)(x)(Δx)  # partial derivative
      0.8691056836969242

      julia> ∂ᵀ(f)(x)(1)  # gradient
      3-element Vector{Float64}:
       0.8691056836969242
       0.9973491983376236
       0.5768822265195823
      ```
    ][
      ```julia
      julia> FD.derivative(t -> f(x + t * Δx), 0)
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

#new-section[Exploiting sparsity]

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
      [cost], [$n$ JVPs (input dimension)], [$m$ JVPs (output dimension)],
    )
  ]
]

#slide[
  = Using fewer products

  When the Jacobian is sparse, we can compute it faster#footnote(cite(<curtisEstimationSparseJacobian1974>, form:"prose")).

  If columns $j_1, dots, j_k$ of $J$ are structurally orthogonal (their nonzeros never overlap), we deduce them all from a single JVP:
  $ J_(j_1) + dots + J_(j_k) = partial f(x)[e_(j_1) + dots + e_(j_k)] $

  Once we have grouped columns, sparse AD has two steps:

  3. one JVP for each group $c = {j_1, dots, j_k}$
  4. decompression into individual columns $j_1, dots, j_k$
]

#slide[
  = The gist in one slide

  #figure(
    image("../assets/img/paper/fig1.png", width: 100%),
    caption: cite(<hillSparserBetterFaster2025>, form: "prose"),
  )
]

#slide[
  = Two preliminary steps

  When grouping columns, we want to

  - guarantee structural orthogonality (correctness)
  - form the smallest number of groups (efficiency)

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

  The preparation phase can be amortized across several inputs.
]

#new-section[Pattern detection and coloring]

#slide[
  = Tracing dependencies in the computation graph

  #columns(2)[
    #image("../assets/img/blog/compute_graph.png", width: 100%)

    #colbreak()

    Computation graph for $ y_1 &= x_1 x_2 + "sign"(x_3) \  y_2 &=  "sign"(x_3) times (x_4 / 2) $

    Its Jacobian will have 3 nonzero coefficients.
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
    /(a::Tracer, b) = Tracer(a.indices)
    sign(a::Tracer) = Tracer()  # zero derivatives
    ```
  ]
  #only(2)[

    Does it work?

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
  = Partitions of a matrix

  / Orthogonal: for all $(i, j)$ s.t. $A_(i j) != 0$,
    - column $j$ is alone in group $c(j)$ with a nonzero in row $i$
  / Symmetrically orthogonal: for all $(i, j)$ s.t. $A_(i j) != 0$,
    - either column $j$ is alone in group $c(j)$ with a nonzero in row $i$
    - or column $i$ is alone in group $c(i)$ with a nonzero in row $j$

  Each partition can be reformulated as a specific coloring problem#footnote(cite(<gebremedhinWhatColorYour2005>, form: "prose")).
]

#slide[
  = Graph representations of a matrix

  / Column intersection: $(j_1, j_2) in cal(E) arrow.l.r.double.long exists i, A_(i j_1) != 0 "and" A_(i j_2) != 0$
  / Bipartite: $(i, j) in cal(E) arrow.l.r.double.long A_(i j) != 0$ (2 vertex sets $cal(I)$ and $cal(J)$)
  / Adjacency (sym.): $(i, j) in cal(E) arrow.l.r.double.long i != j$ & $A_(i j) != 0$
  #figure(
    image("../assets/img/survey/bipartite_column_2.png", width: 80%),
    caption: cite(<gebremedhinWhatColorYour2005>, form: "prose"),
  )
]

#slide[
  = Jacobian coloring

  Coloring of intersection graph / distance-2 coloring of bipartite graph
  #figure(
    image("../assets/img/survey/bipartite_column_full.png", width: 100%),
    caption: cite(<gebremedhinWhatColorYour2005>, form: "prose"),
  )
]

#slide[
  = Hessian coloring

  #only(1)[

    Star coloring of adjacency graph
    #figure(
      image("../assets/img/survey/star_coloring3.png", width: 90%),
      caption: cite(<gebremedhinEfficientComputationSparse2009>, form: "prose"),
    )
  ]

  #only(2)[

    #columns(2)[
      Why a "star" coloring#footnote(cite(<colemanEstimationSparseHessian1984>, form: "prose"))? Consider

      $
        A = mat(
        A_(k k), A_(k i), dot, dot;
        A_(i k), A_(i i), A_(i j), dot;
        dot, A_(j i), A_(j j), A_(j l);
        dot, dot, A_(l j), A_(l l);
      )
      $

      #colbreak()

      If coloring $c$ yields a symmetrically orthogonal partition:

      - $c(i) != c(j)$
      - $c(i) != c(k)$
      - $c(j) != c(l)$
    ]

    Any path on 4 vertices $(i, j, k, l)$ must use at least 3 colors
    $arrow.l.r.double.long$ any 2-colored subgraph is a collection of disjoint stars (it contains no path longer than 3).
  ]
]

#slide[
  = Jacobian bicoloring

  Bidirectional coloring of bipartite graph, with neutral color
  #figure(
    image("../assets/img/survey/bicoloring.png", width: 80%),
    caption: cite(<gebremedhinWhatColorYour2005>, form: "prose"),
  )
]

#slide[
  = Bicoloring from symmetric coloring [new]

  To color the rows and columns of $J$, color the columns of $H = mat(
    0, J;
    J, 0;
  )$

  It sounds simple, but:

  - Some colors may be redundant
  - Detecting these is tightly linked to the two-colored structures
  - Efficient decompression requires lots of preprocessing
]

#slide[
  = The sharp bits

  #toolbox.side-by-side[
    *Pattern detection*

    - Linear versus nonlinear interactions
    - Local versus global sparsity
  ][
    *Coloring*

    - Only heuristic algorithms
    - Vertex ordering matters a lot
  ]
]

#new-section[Implementation]

#slide[
  = AD in Python & Julia

  #only(1)[
    #figure(image("../assets/img/juliacon/python_julia_user.png", width: 100%))
  ]
  #only(2)[
    #figure(image("../assets/img/juliacon/python_julia_dev.png", width: 95%))
  ]
]

#slide[
  = Interfaces for experimenting [new]

  #toolbox.side-by-side[
    #figure(
      image("../assets/img/misc/keras.jpg"),
      caption: [In Python, #link("https://keras.io/")[`Keras`] supports Tensorflow, PyTorch and JAX.],
    )
  ][
    #figure(
      image("../assets/img/logo.svg", width: 55%),
      caption: [In Julia, 14 AD backends inside #link("https://github.com/JuliaDiff/DifferentiationInterface.jl")[`Differentiationterface.jl`]],
    )
  ]

  Once we have a common syntax, we can do more!
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
  - Step 1: #link("https://github.com/adrhill/SparseConnectivityTracer.jl")[`SparseConnectivityTracer.jl`]
  - Steps 2 & 4: #link("https://github.com/gdalle/SparseMatrixColorings.jl")[`SparseMatrixColorings.jl`]
  - Step 3: #link("https://github.com/JuliaDiff/DifferentiationInterface.jl")[`Differentiationterface.jl`]

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

      [lines of code], [4861], [4242], [16971],
      [indirect dependents], [420], [437], [426],
      [downloads / month], [4.2k], [16k], [20k],
    )
  ]

  Compatible with generic code!
]

#slide[
  = Impact

  Users already include...

  - Scientific computing: #link("https://sciml.ai/")[`SciML`] (Julia's `scipy`)
    - Differential equations
    - Nonlinear solvers
    - Optimization
  - Probabilistic programming: #link("https://turinglang.org/")[`Turing.jl`]
  - Symbolic regression: #link("https://github.com/MilesCranmer/PySR")[`PySR`]

]

#slide[
  = Live demo

  This is the part where things go sideways.
]

#new-section[Conclusion]

#slide[
  = Perspectives

  - GPU-compatible pattern detection and coloring
  - Adaptation in JAX with program transformations
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
  - *#cite(<hillSparserBetterFaster2025>, form: "prose")*
]

#slide[
  = Bibliography

  #text(size: 12pt)[
    #bibliography("AD.bib", title: none, style: "apa")
  ]
]

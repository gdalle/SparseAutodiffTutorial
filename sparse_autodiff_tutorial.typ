#import "@preview/polylux:0.3.1": *
#import themes.university: *

#set text(size: 30pt)

#let juliablue = rgb("#4063d8")
#show link: underline

#show: university-theme.with(
  short-author: "G. Dalle",
  short-title: "Sparse Automatic Differentiation",
  short-date: "28.11.2024",
  color-a: juliablue,
  color-b: juliablue,
  color-c: rgb("#FFFFFF"),
)

#title-slide(
  authors: ("Guillaume Dalle", "Alexis Montoison", "Adrian Hill"),
  title: "Sparse Automatic Differentiation",
  subtitle: "The fastest Jacobians in the West",
  date: "28.11.2024",
  institution-name: "EPFL, INDY lab seminar",
)

#slide(title: [Motivation])[
  - Many algorithms require derivative matrices:
    - Differential equations $arrow$ Jacobian
    - Convex optimization $arrow$ Hessian
  - Large matrices are expensive to compute and store...
  - ... except when they are sparse!
  - Sparsity can be leveraged automatically.
]

#slide(title: [What is AD?], new-section: [Forward & reverse AD])[

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
      [program to compute the derivative \ (linear operator) $ x arrow.r.bar.long partial f(x) $],
    )
  ]

  Two ingredients only:

  1. hardcode basic derivatives ($+$, $times$, $exp$, $log$, ...)
  2. handle compositions $f = g compose h$

]


#slide(title: [Linear operators])[
  For a function $f = g compose h$, the chain rule gives

  $
    "standard" & wide & partial f(x) & = partial g (h(x)) compose partial h (x) \
    "adjoint" & wide & partial f(x)^* & = partial h (x)^* compose partial g (h(x))^*
  $

  These linear operators apply as follows:

  $
    "forward" & wide & partial f(x): & u stretch(arrow.r.bar)^(partial h(x)) v stretch(arrow.r.bar)^(partial g(
      h(x)
    )) w \
    "reverse" & wide & partial f(x)^*: & u stretch(arrow.l.bar)_(partial h(x)^*) v stretch(arrow.l.bar)_(partial g(
      h(x)
    )^*) w \
  $
]

#slide(title: [Two modes])[
  Forward-mode AD computes Jacobian-Vector Products:

  $ u arrow.r.bar.long partial f(x)[u] $

  Reverse-mode AD computes Vector-Jacobian Products:

  $
    w arrow.r.bar.long quad & partial f(x)^* [w]
  $

  No need to materialize intermediate Jacobian matrices!
]

#focus-slide(background-color: juliablue)[
  Cost of 1 JVP or VJP for $f$ $approx$ \ cost of 1 evaluation of $f$.
]

#slide(title: [From operators to matrices], new-section: [Leveraging sparsity])[
  To compute the Jacobian matrix $J$ of a composition $f: bb(R)^m arrow.long bb(R)^n$:
  - #strike[product of intermediate Jacobian matrices]
  - reconstruction from JVPs or VJPs

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

#slide(title: [Using fewer products])[
  When the Jacobian is sparse, we can compute it faster.

  If columns $j_1, dots, j_k$ of $J$ are structurally orthogonal (their nonzeros never overlap), we deduce them all from a single JVP:
  $ J_(j_1) + dots + J_(j_k) = partial f(x)[e_(j_1) + dots + e_(j_k)] $

  Once we have grouped columns, sparse AD has two steps:

  1. compressed differentiation of each group $c = {j_1, dots, j_k}$
  2. decompression into individual columns $j_1, dots, j_k$
]

#slide(title: [Compression])[
  #figure(
    image("assets/img/survey/compression.png", height: 90%),
    caption: cite(<gebremedhinWhatColorYour2005>, form: "prose"),
  )
]

#slide(title: [Two preliminary steps])[
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
        1. structure detection \
        2. coloring
      ],
      [
        3. compressed differentiation \
        4. decompression
      ],
    )
  ]

  The preparation phase can be amortized across several inputs.
]

#slide(title: [Operator overloading], new-section: [Structure detection])[

]

#slide(title: [Partitions for matrix $A$], new-section: [Coloring])[
  / Orthogonal: for all $(i, j)$ s.t. $A_(i j) != 0$,
    - column $j$ is the only one of its group with a nonzero in row $i$
  / Symmetrically orthogonal: for all $(i, j)$ s.t. $A_(i j) != 0$,
    - column $j$ is the only one of its group with a nonzero in row $i$ OR
    - OR column $i$ is the only one of its group with a nonzero in row $j$

  Each partition can be reformulated as a specific coloring problem.
]

#slide(title: [Graph types for matrix $A$])[
  / Column intersection: $(j_1, j_2) in cal(E) arrow.l.r.double.long exists i, A_(i j_1) != 0 "and" A_(i j_2) != 0$
  / Bipartite: $(i, j) in cal(E) arrow.l.r.double.long A_(i j) != 0$ (2 vertex sets $cal(I)$ and $cal(J)$)
  / Adjacency (sym.): $(i, j) in cal(E) arrow.l.r.double.long i != j$ & $A_(i j) != 0$
  #figure(
    image("assets/img/survey/bipartite_column_2.png", width: 80%),
    caption: cite(<gebremedhinWhatColorYour2005>, form: "prose"),
  )
]

#slide(title: [Jacobian coloring])[
  Coloring of intersection graph / distance-2 coloring of bipartite graph
  #figure(
    image("assets/img/survey/bipartite_column_full.png", width: 100%),
    caption: cite(<gebremedhinWhatColorYour2005>, form: "prose"),
  )
]

#slide(title: [Hessian coloring])[
  Star coloring of adjacency graph
  #figure(
    image("assets/img/survey/star_coloring.png", width: 70%),
    caption: cite(<gebremedhinWhatColorYour2005>, form: "prose"),
  )
]

#slide(title: [Hessian coloring (2)])[
  Why a star coloring?
]

#slide(title: [Jacobian bicoloring])[
  Bidirectional coloring of bipartite graph, with neutral color
  #figure(
    image("assets/img/survey/bicoloring.png", width: 80%),
    caption: cite(<gebremedhinWhatColorYour2005>, form: "prose"),
  )
]

#slide(title: [New insights on Jacobian bicoloring])[
  1. A bicoloring of $J$ is given by a star coloring of $H = mat(0, J; J^T, 0)$
  2. Diagonal of $H$ is zero: relax star coloring into _no zig-zag coloring_ $arrow.r.double.long$ no path on 4 vertices can have colors $(c_1, c_2, c_1, c_2)$
]

#slide(title: [A high-level sparse AD ecosystem], new-section: [Implementations])[
  - Step 1 (structure detection): #link("https://github.com/adrhill/SparseConnectivityTracer.jl")[`SparseConnectivityTracer.jl`]
  - Steps 2 & 4 (coloring, decompression): #link("https://github.com/gdalle/SparseMatrixColorings.jl")[`SparseMatrixColorings.jl`]
  - Step 3 (compressed diffeentiation): #link("https://github.com/JuliaDiff/DifferentiationInterface.jl")[`DifferentiationInterface.jl`]

  #align(center)[
    #image("assets/img/logo.svg", height: 60%)
  ]
]

#slide(title: [Impact])[
  `DifferentiationInterface.jl` has
  
  - 11 months of age
  - 14430 lines of Julia code
  - 200 stars on GitHub
  - 392 indirect dependents
  - around 19k downloads per month (excluding bots)
]

#slide(title: [Perspectives])[
  - GPU-compatible structure detection and coloring
  - Pure Julia autodiff engine based on SSA-IR
]

#slide(title: [Going further], new-section: [Appendix])[
  On general AD:

  - #cite(<blondelElementsDifferentiableProgramming2024>, form: "prose")
  - #cite(<margossianReviewAutomaticDifferentiation2019>, form: "prose")
  - #cite(<baydinAutomaticDifferentiationMachine2018>, form: "prose")

  On sparse AD:

  - #cite(<griewankEvaluatingDerivativesPrinciples2008>, form: "prose")
  - #cite(<gebremedhinWhatColorYour2005>, form: "prose")
  - #cite(<waltherComputingSparseHessians2008>, form: "prose")
]

#slide(title: [Bibliography])[
  #bibliography("assets/bib/sparsead.bib", title: none, style: "apa")
]

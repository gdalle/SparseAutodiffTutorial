#import "@preview/polylux:0.3.1": *
#import themes.university: *

#set text(size: 30pt)

#let juliablue = rgb("#4063d8")

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

  1. compressed differentiation
  2. decompression
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
        1. pattern detection \
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

#slide(title: [Bipartite graph representation], new-section: [Coloring problems])[

]



#slide(title: [Sources], new-section: [Appendix])[
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

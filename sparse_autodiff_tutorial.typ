#import "@preview/polylux:0.3.1": *
#import themes.university: *

#set text(size: 30pt, font: "DejaVu Sans")

#show: university-theme.with(
  short-author: "G. Dalle",
  short-title: "Sparse automatic differentiation",
  short-date: "28.11.2024",
  color-a: rgb("#4063d8"),
  color-b: rgb("#4063d8"),
  color-c: rgb("#FFFFFF"),
)

#title-slide(
  authors: ("Guillaume Dalle", "Alexis Montoison", "Adrian Hill"),
  title: "Sparse automatic differentiation",
  subtitle: "The fastest Jacobians in the West",
  date: "28.11.2024",
  institution-name: "EPFL, INDY lab seminar",
)

#slide(title: [Motivation])[
  - Many algorithms require Jacobians or Hessians.
  - In large dimension, matrices are expensive...
  - ... except when they are sparse.
]

#slide(title: [Chain rule], new-section: [Forward & reverse AD])[
  For a function $f = g compose h$, the chain rule is

  $ D f(x) = D h (g(x)) compose D g (x) $
]

#slide(title: [From operators to matrices], new-section: [Dense Jacobians & Hessians])[

]

#slide(title: [More efficient products], new-section: [Sparse Jacobians & Hessians])[

]

#slide(title: [Sparse AD prelude], new-section: [Detection and coloring])[

]

#slide(title: [Bibliography], new-section: [Appendix])[
  #bibliography("assets/bib/sparsead.bib", title: none)
]

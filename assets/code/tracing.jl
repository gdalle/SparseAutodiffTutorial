import Base: +, *, /, sign

struct Tracer
	indices::Set{Int}
end

Tracer() = Tracer(Set{Int}())

+(a::Tracer, b::Tracer) = Tracer(a.indices ∪ b.indices)
*(a::Tracer, b::Tracer) = Tracer(a.indices ∪ b.indices)
/(a::Tracer, b) = Tracer(a.indices)
sign(a::Tracer) = Tracer()

f(x) = [x[1] * x[2] * sign(x[3]), sign(x[3]) * x[4] / 2];
x = Tracer.(Set.([1, 2, 3, 4]))
f(x)

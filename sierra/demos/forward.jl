import Base: +, *

struct Dual
    val::Float64
    der::Float64
end

+(x::Dual, y::Dual) = Dual(x.val + y.val, x.der + y.der)
*(x::Dual, y::Dual) = Dual(
    x.val * y.val, x.der * y.val + x.val * y.der
)
+(x, y::Dual) = Dual(x, 0) + y
*(x, y::Dual) = Dual(x, 0) * y

f(x) = 1 + 2 * x + 3 * x * x;

f(Dual(4, 1))

(f(4 + 1e-5) - f(4)) / 1e-5

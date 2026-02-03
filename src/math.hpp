#pragma once

#include <cassert>
#include <cmath>
#include <type_traits>

namespace dvx
{

// Compute a*b - c*d using Kahan's algorithm
template<typename Float>
Float kahan(Float a, Float b, Float c, Float d)
{
    Float cd    = c * d;
    Float error = fma(c, d, -cd);
    return fma(a, b, -cd) - error;
}

#define VECTOR_MAKE_CONVENIENCE_ACCESSOR(name, index, N) \
    template<typename = std::enable_if_t<(index < N)>>   \
    inline T& name()                                     \
    {                                                    \
        return (*this)[index];                           \
    }                                                    \
                                                         \
    template<typename = std::enable_if_t<(index < N)>>   \
    inline T const& name() const                         \
    {                                                    \
        return (*this)[index];                           \
    }

#define DEFINE_BINARY_OPERATOR(type1, type2, rtype, op)                                                                                    \
    template<typename T, typename U, unsigned int N>                                                                                       \
    rtype<decltype(std::declval<T>() op std::declval<U>()), N> operator op (type1<T, N> const& v1, type2<U, N> const& v2)        \
    {                                                                                                                                      \
        rtype<decltype(std::declval<T>() op std::declval<U>()), N> out;                                                                  \
        for (unsigned int i = 0; i < N; i++)                                                                                               \
        {                                                                                                                                  \
            out[i] = v1[i] op v2[i];                                                                                                       \
        }                                                                                                                                  \
        return out;                                                                                                                        \
    }                                                                                                                                      \
    template<typename T, typename U, unsigned int N,                                                                                       \
             std::enable_if_t<std::is_same_v<type1<T, N>, rtype<decltype(std::declval<T>() op std::declval<U>()), N>>>* = nullptr> \
    type1<T, N>& operator op##= (type1<T, N>& v1, type2<U, N> const& v2)                                                           \
    {                                                                                                                                      \
        for (unsigned int i = 0; i < N; i++)                                                                                               \
        {                                                                                                                                  \
            v1[i] op##= v2[i];                                                                                                            \
        }                                                                                                                                  \
        return v1;                                                                                                                         \
    }

// Same as `DEFINE_BINARY_OPERATOR` but N is a parameter
#define DEFINE_BINARY_OPERATOR_N(type1, type2, rtype, N, op)                                                                         \
    template<typename T, typename U>                                                                                                 \
    rtype<decltype(std::declval<T>() op std::declval<U>())> operator##op (type1<T> const& v1, type2<U> const& v2)           \
    {                                                                                                                                \
        rtype<decltype(std::declval<T>() op std::declval<U>())> out;                                                               \
        for (unsigned int i = 0; i < N; i++)                                                                                         \
        {                                                                                                                            \
            out[i] = v1[i] op v2[i];                                                                                                 \
        }                                                                                                                            \
        return out;                                                                                                                  \
    }                                                                                                                                \
    template<typename T, typename U, std::enable_if_t<!std::is_same_v<type1<T>, type2<U>>>* = nullptr>                       \
    rtype<decltype(std::declval<U>() op std::declval<T>())> operator##op (type2<U> const& v2, type1##<T> const& v1)           \
    {                                                                                                                                \
        rtype<decltype(std::declval<U>() op std::declval<T>())> out;                                                               \
        for (unsigned int i = 0; i < N; i++)                                                                                         \
        {                                                                                                                            \
            out[i] = v1[i] op v2[i];                                                                                                 \
        }                                                                                                                            \
        return out;                                                                                                                  \
    }                                                                                                                                \
    template<typename T, typename U,                                                                                                 \
             std::enable_if_t<std::is_same_v<type1<T>, rtype<decltype(std::declval<T>() op std::declval<U>())>>>* = nullptr> \
    type1<T>& operator##op##=(type1<T>& v1, type2<U> const& v2)                                                              \
    {                                                                                                                                \
        for (unsigned int i = 0; i < N; i++)                                                                                         \
        {                                                                                                                            \
            v1[i] op##= v2[i];                                                                                                      \
        }                                                                                                                            \
        return v1;                                                                                                                   \
    }

#define DEFINE_BINARY_OPERATOR_SCALAR(type, op)                                  \
    template<typename T, typename Scalar, unsigned int N,                        \
             typename std::enable_if_t<std::is_arithmetic_v<Scalar>>* = nullptr> \
    type<T, N> operator op (type<T, N> const& v, Scalar scalar)          \
    {                                                                            \
        type<T, N> result;                                                     \
        for (unsigned int i = 0; i < N; ++i)                                     \
        {                                                                        \
            result[i] = v[i] op scalar;                                          \
        }                                                                        \
        return result;                                                           \
    }                                                                            \
                                                                                 \
    template<typename T, typename Scalar, unsigned int N,                        \
             typename std::enable_if_t<std::is_arithmetic_v<Scalar>>* = nullptr> \
    type<T, N> operator op (Scalar scalar, type<T, N> const& v)          \
    {                                                                            \
        type<T, N> result;                                                     \
        for (unsigned int i = 0; i < N; ++i)                                     \
        {                                                                        \
            result[i] = scalar op v[i];                                          \
        }                                                                        \
        return result;                                                           \
    }                                                                            \
    template<typename T, typename Scalar, unsigned int N,                        \
             typename std::enable_if_t<std::is_arithmetic_v<Scalar>>* = nullptr> \
    type<T, N>& operator op##= (type<T, N> const& v, Scalar scalar)        \
    {                                                                            \
        for (unsigned int i = 0; i < N; i++)                                     \
        {                                                                        \
            v[i] op##= scalar;                                                  \
        }                                                                        \
        return v;                                                                \
    }

// Same as `DEFINE_BINARY_OPERATOR_SCALAR` but N is a parameter
#define DEFINE_BINARY_OPERATOR_SCALAR_N(type, N, op)                             \
    template<typename T, typename Scalar,                                        \
             typename std::enable_if_t<std::is_arithmetic_v<Scalar>>* = nullptr> \
    type<T> operator op (type<T> const& v, Scalar scalar)                \
    {                                                                            \
        type<T> result;                                                        \
        for (unsigned int i = 0; i < N; ++i)                                     \
        {                                                                        \
            result[i] = v[i] op scalar;                                          \
        }                                                                        \
        return result;                                                           \
    }                                                                            \
                                                                                 \
    template<typename T, typename Scalar,                                        \
             typename std::enable_if_t<std::is_arithmetic_v<Scalar>>* = nullptr> \
    type<T> operator op (Scalar scalar, type<T> const& v)                \
    {                                                                            \
        type<T> result;                                                        \
        for (unsigned int i = 0; i < N; ++i)                                     \
        {                                                                        \
            result[i] = scalar op v[i];                                          \
        }                                                                        \
        return result;                                                           \
    }                                                                            \
    template<typename T, typename Scalar,                                        \
             typename std::enable_if_t<std::is_arithmetic_v<Scalar>>* = nullptr> \
    type<T>& operator op##= (type<T> const& v, Scalar scalar)              \
    {                                                                            \
        for (unsigned int i = 0; i < N; i++)                                     \
        {                                                                        \
            v[i] op##= scalar;                                                  \
        }                                                                        \
        return v;                                                                \
    }

#define DEFINE_MASKING_AND_REDUCTION_FUNCTIONS_N(type, N)                                             \
    template<typename T, typename Predicate>                                                          \
    type<T> where(Predicate&& predicate, type<T> const& true_value, type<T> const& false_value) \
    {                                                                                                 \
        type<T> result;                                                                             \
        for (unsigned int i = 0; i < N; ++i)                                                          \
            result[i] = predicate(i) ? true_value[i] : false_value[i];                                \
        return result;                                                                                \
    }                                                                                                 \
                                                                                                      \
    template<typename Predicate>                                                                      \
    bool all(Predicate&& predicate)                                                                   \
    {                                                                                                 \
        bool result = true;                                                                           \
        for (unsigned int i = 0; i < N; ++i)                                                          \
            result &= predicate(i);                                                                   \
        return result;                                                                                \
    }                                                                                                 \
                                                                                                      \
    template<typename Predicate>                                                                      \
    bool any(Predicate&& predicate)                                                                   \
    {                                                                                                 \
        bool result = false;                                                                          \
        for (unsigned int i = 0; i < N; ++i)                                                          \
            result |= predicate(i);                                                                   \
        return result;                                                                                \
    }

namespace detail
{

template<typename Predicate1, typename Predicate2>
struct PredicateAnd
{
    Predicate1 const& pred1;
    Predicate2 const& pred2;

    bool operator()(unsigned int i) const
    {
        return pred1(i) && pred2(i);
    }
};

} // namespace detail

#define DEFINE_PREDICATE(name, suffix, op)                                                           \
    namespace detail                                                                                 \
    {                                                                                                \
    template<typename Vector>                                                                        \
    struct Predicate##suffix                                                                         \
    {                                                                                                \
        typename Vector::ValueType value;                                                            \
        Vector const&              vector;                                                           \
                                                                                                     \
        bool operator()(unsigned int i) const                                                        \
        {                                                                                            \
            return vector[i] op value;                                                               \
        }                                                                                            \
                                                                                                     \
        template<typename Other>                                                                     \
        PredicateAnd<Predicate##suffix<Vector>, Other> operator&&(Other const& other) const        \
        {                                                                                            \
            return PredicateAnd<Predicate##suffix<Vector>, Other>{.pred1 = *this, .pred2 = other}; \
        }                                                                                            \
    };                                                                                               \
    }                                                                                                \
                                                                                                     \
    template<typename Vector, typename Scalar,                                                       \
             typename std::enable_if_t<std::is_arithmetic_v<Scalar>>* = nullptr>                     \
    detail::Predicate##suffix<Vector> name (Vector const& vector, Scalar value)                   \
    {                                                                                                \
        using T = typename Vector::ValueType;                                                        \
        return detail::Predicate##suffix<Vector>{.value = T(value), .vector = vector};             \
    }

DEFINE_PREDICATE(is_not_equal, Neq, !=);
DEFINE_PREDICATE(is_equal, Eq, ==);
DEFINE_PREDICATE(is_greater_equal, GreaterEq, >=);
DEFINE_PREDICATE(is_greater, Greater, >);

template<typename Vector>
detail::PredicateNeq<Vector> is_nonzero(Vector const& vector)
{
    return is_not_equal(vector, 0);
}

template<typename Vector>
detail::PredicateEq<Vector> is_zero(Vector const& vector)
{
    return is_equal(vector, 0);
}

template<typename T, unsigned int N>
class VectorBase
{
public:
    using ValueType = T;

    static constexpr unsigned int Size = N;

    inline VectorBase(T initial_value = T(0))
    {
        for (unsigned int i = 0; i < N; ++i)
            m_data[i] = initial_value;
    }

    template<typename U>
    inline VectorBase(VectorBase<U, N> const& other)
    {
        (*this) = other;
    }

    inline VectorBase(T const* data)
    {
        // Load vector from an array
        for (unsigned int i = 0; i < N; ++i)
            m_data[i] = data[i];
    }

    inline T& operator[](unsigned int i)
    {
        assert(i < N);
        return m_data[i];
    }

    inline T const& operator[](unsigned int i) const
    {
        assert(i < N);
        return m_data[i];
    }

    template<typename U>
    inline VectorBase<T, N>& operator=(VectorBase<U, N> const& other)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            m_data[i] = static_cast<T>(other[i]);
        }

        return *this;
    }

protected:
    T m_data[N];
};

template<typename T, unsigned int N>
class Vector : public VectorBase<T, N>
{
public:
    using VectorBase<T, N>::VectorBase;

    template<typename = std::enable_if_t<(N == 2)>>
    inline Vector(T x, T y)
    {
        VectorBase<T, N>::m_data[0] = x;
        VectorBase<T, N>::m_data[1] = y;
    }

    template<typename = std::enable_if_t<(N == 3)>>
    inline Vector(T x, T y, T z)
    {
        VectorBase<T, N>::m_data[0] = x;
        VectorBase<T, N>::m_data[1] = y;
        VectorBase<T, N>::m_data[2] = z;
    }

    template<typename = std::enable_if_t<(N == 4)>>
    inline Vector(T x, T y, T z, T w)
    {
        VectorBase<T, N>::m_data[0] = x;
        VectorBase<T, N>::m_data[1] = y;
        VectorBase<T, N>::m_data[2] = z;
        VectorBase<T, N>::m_data[3] = w;
    }

    VECTOR_MAKE_CONVENIENCE_ACCESSOR(x, 0, N);
    VECTOR_MAKE_CONVENIENCE_ACCESSOR(y, 1, N);
    VECTOR_MAKE_CONVENIENCE_ACCESSOR(z, 2, N);
    VECTOR_MAKE_CONVENIENCE_ACCESSOR(w, 3, N);
};

DEFINE_BINARY_OPERATOR_SCALAR(Vector, *)
DEFINE_BINARY_OPERATOR_SCALAR(Vector, /)
DEFINE_BINARY_OPERATOR_SCALAR(Vector, +)
DEFINE_BINARY_OPERATOR_SCALAR(Vector, -)

DEFINE_BINARY_OPERATOR(Vector, Vector, Vector, *)
DEFINE_BINARY_OPERATOR(Vector, Vector, Vector, /)
DEFINE_BINARY_OPERATOR(Vector, Vector, Vector, +)
DEFINE_BINARY_OPERATOR(Vector, Vector, Vector, -)

template<typename T, unsigned int N>
Vector<T, N> operator-(Vector<T, N> const& v)
{
    Vector<T, N> result;
    for (unsigned int i = 0; i < N; i++)
    {
        result[i] = -v[i];
    }
    return result;
}

template<typename T, unsigned int N>
T dot(Vector<T, N> const& left, Vector<T, N> const& right)
{
    T result(0);
    for (unsigned int i = 0; i < N; i++)
    {
        result += static_cast<T>(left[i] * right[i]);
    }
    return result;
}

template<typename T, unsigned int N>
inline T norm(Vector<T, N> const& v)
{
    return std::sqrt(dot(v, v));
}

template<typename T>
Vector<T, 3> cross(Vector<T, 3> const& left, Vector<T, 3> const& right)
{
    Vector<T, 3> result;
    result[0] = kahan(left[1], right[2], right[1], left[2]);
    result[1] = kahan(right[0], left[2], left[0], right[2]);
    result[2] = kahan(left[0], right[1], right[0], left[1]);
    return result;
}

/**
 * \brief Construct a unit-norm frame (i.e., an orthonormal basis) from a given (unit-length) vector z
 *
 * The returned vectors x and y fulfill the properties:
 * 1) ||x|| = ||y|| = 1
 * 2) <z,x> = 0, <z,y>=0, <x,y>=0
 * 3) det([x, y, z]) = 1
 *
 * \param[in]  z The scene data structure
 * \param[out] x The intersection instance
 * \param[out] y The outgoing direction in world space, pointing away from the intersection point
 */
template<typename T>
void construct_frame(Vector<T, 3> const& z, Vector<T, 3>* x, Vector<T, 3>* y)
{
    *x = normalize(cross(z + Vector<T, 3>(0.1f, 0.2f, 0.3f), z));
    *y = normalize(cross(z, *x));
}

template<typename T, unsigned int N>
inline Vector<T, N> normalize(Vector<T, N> const& v)
{
    return v / std::sqrt(dot(v, v));
}

template<typename T, unsigned int N>
inline Vector<T, N> reflect(Vector<T, N> const& v, Vector<T, N> const& normal)
{
    return normal * T(2) * dot(v, normal) - v;
}

template<typename T, unsigned int N>
inline T volume(Vector<T, N> const& v)
{
    T result(1);
    for (unsigned int i = 0; i < N; ++i)
        result *= v[i];
    return result;
}

#define DECLARE_FLOAT_DEFINES(type, N)   \
    template<typename T>                 \
    using type##N    = type<T, N>;  \
    using type##N##f = type##N<float>; \
    using type##N##d = type##N<double>;

#define DECLARE_INTEGER_DEFINES(type, N) \
    using type##N##u = type##N<unsigned int>;

DECLARE_FLOAT_DEFINES(Vector, 2)
DECLARE_FLOAT_DEFINES(Vector, 3)
DECLARE_FLOAT_DEFINES(Vector, 4)
DECLARE_INTEGER_DEFINES(Vector, 2)
DECLARE_INTEGER_DEFINES(Vector, 3)
DECLARE_INTEGER_DEFINES(Vector, 4)

template<typename T, unsigned int N>
class Point : public Vector<T, N>
{
public:
    using Vector<T, N>::Vector;
};

DEFINE_BINARY_OPERATOR_SCALAR(Point, +)
DEFINE_BINARY_OPERATOR_SCALAR(Point, -)
DEFINE_BINARY_OPERATOR_SCALAR(Point, /)
DEFINE_BINARY_OPERATOR_SCALAR(Point, *)

DEFINE_BINARY_OPERATOR(Point, Vector, Point, +)
DEFINE_BINARY_OPERATOR(Vector, Point, Point, +)
DEFINE_BINARY_OPERATOR(Point, Vector, Point, -)
DEFINE_BINARY_OPERATOR(Vector, Point, Point, -)
DEFINE_BINARY_OPERATOR(Point, Vector, Point, *)
DEFINE_BINARY_OPERATOR(Vector, Point, Point, *)
DEFINE_BINARY_OPERATOR(Point, Vector, Point, /)
DEFINE_BINARY_OPERATOR(Point, Point, Vector, -)
DEFINE_BINARY_OPERATOR(Point, Point, Point, +)

DECLARE_FLOAT_DEFINES(Point, 2)
DECLARE_FLOAT_DEFINES(Point, 3)
DECLARE_FLOAT_DEFINES(Point, 4)

} // namespace dvx
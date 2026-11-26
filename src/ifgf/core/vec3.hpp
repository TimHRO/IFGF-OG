#pragma once

// Plain 3-vector, usable in host code and in SYCL kernels.

#include <ifgf/core/math.hpp>

#include <cmath>

namespace ifgf {

template <class Real>
struct Vec3 {
  Real x{}, y{}, z{};

  constexpr Real& operator[](int d) { return d == 0 ? x : (d == 1 ? y : z); }
  constexpr const Real& operator[](int d) const { return d == 0 ? x : (d == 1 ? y : z); }

  constexpr Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
  constexpr Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
  constexpr Vec3& operator*=(Real a) { x *= a; y *= a; z *= a; return *this; }

  friend constexpr bool operator==(const Vec3&, const Vec3&) = default;
};

template <class Real> constexpr Vec3<Real> operator+(Vec3<Real> a, const Vec3<Real>& b) { return a += b; }
template <class Real> constexpr Vec3<Real> operator-(Vec3<Real> a, const Vec3<Real>& b) { return a -= b; }
template <class Real> constexpr Vec3<Real> operator*(Vec3<Real> a, Real s) { return a *= s; }
template <class Real> constexpr Vec3<Real> operator*(Real s, Vec3<Real> a) { return a *= s; }

template <class Real> constexpr Real dot(const Vec3<Real>& a, const Vec3<Real>& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <class Real> constexpr Real norm2(const Vec3<Real>& a) { return dot(a, a); }
template <class Real> Real norm(const Vec3<Real>& a) { return math::sqrt(norm2(a)); }

template <class To, class From>
constexpr Vec3<To> vec3Cast(const Vec3<From>& v)
{
  return {static_cast<To>(v.x), static_cast<To>(v.y), static_cast<To>(v.z)};
}

using Vec3d = Vec3<double>;
using Vec3f = Vec3<float>;

}  // namespace ifgf

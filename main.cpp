#include <iostream>
#include <complex>
#include <iomanip>
#include <eigen3/Eigen/Dense>

using namespace std;

typedef double Real;
typedef complex<Real> Complex;
typedef int32_t Integer;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> Matrix;

//static const Real H_BARED = 1.054571726e-34;
static const Real H_BARED = 1;
static const Integer INTEGRATION_STEPS = 1e3;
static const Integer N_MAX = 100;
static const Integer N_OUTPUT = 3;
static const Real mass = 1;
static const Real length = 1000;
static const Real omega = 0.001;
static const Real A = 1.0e-3;
static const Real B = 1.0e-4;

Complex integrate(std::function<Complex(Real)> f, double from, double to, int steps) {
    Complex s = 0;
    double h = (to - from) / steps;
    for (int i = 0; i < steps; ++i) {
        double pos = from + h * i;
        double mid = pos + h * .5;
        double end = pos + h;
        s += (f(pos) + f(mid) * Complex(4.0, 0) + f(end));
    }
    s /= 6.0;
    s *= h;
    return s;
}


Real V(Real x) {
    return omega * omega * mass * x * x / 2.0;
    //return 0.5*mass*x*x*omega*omega*(1.0+A*x+B*x*x);
}

Real k(Integer n) {
    return 2 * M_PI * n / length;
}

Real K(Integer i, Integer j) {
    if (i != j) return 0;
    Real hk = H_BARED * k(j);
    return .5 * hk * hk / mass;
}

Complex U(Integer i, Integer j) {
    auto f = [&](Real x) {
        return exp(Complex(0, (k(i) - k(j)) * x)) * V(x);
    };
    return integrate(f, -length / 2, length / 2, INTEGRATION_STEPS) / length;
}

Complex H(Integer i, Integer j) {
    return K(i, j) + U(i, j);
}

Matrix makeMatrix() {
    Matrix A(N_MAX * 2 + 1, N_MAX * 2 + 1);
    for (int i = -N_MAX; i <= N_MAX; i++)
        for (int j = -N_MAX; j <= N_MAX; j++)
            A(i + N_MAX, j + N_MAX) = H(i, j);
    return A;
}

int main() {
    cout << "Program started" << endl;
    cout << "Matrix generating..." << endl;
    Matrix A = makeMatrix();
    cout << "Matrix generated" << endl;
    cout << "Matrix eigen computing..." << endl;
    Eigen::ComplexEigenSolver<Matrix> solver;
    solver.compute(A, true);
    cout << "Matrix eigen computed" << endl;
    auto vectors = solver.eigenvectors();

    for (int i = 0; i < solver.eigenvalues().rows() && i < N_OUTPUT; ++i) {
        Complex psi(0, 0);
        for (int j = 0; j < N_MAX; ++j) {
            psi += vectors(j, i);
        }
        psi /= sqrt(length);
        cout << setiosflags(ios::fixed) << setprecision(5) << "E_" << i << " : " << solver.eigenvalues()[i].real();
        cout << "\t\t PSI_" << i << " : " << psi << "*exp(i " << 2 * i / length << " PI x)" << endl;
    }

}
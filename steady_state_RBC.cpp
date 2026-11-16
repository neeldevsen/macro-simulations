#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <random>
#include <string>
#include <cstdlib>


using namespace Eigen;

MatrixXd max_value(const std::vector<double>& K, const std::vector<double>& a, const int M, const int N, const double alpha, const double delta, const double beta, const MatrixXd& Q, MatrixXi& G, MatrixXd& C, MatrixXd& I)
{

    MatrixXd V (M, N);
    for (int i{0}; i < M; ++i)
    {
        for (int j{0}; j < N; ++j)
        {
            double R_ij {1.0 / 3.0 * std::exp(a[j]) * std::pow(K[i], alpha) + (1.0 - delta) * K[i]};
            double max_val {-INFINITY};
            for (int l{0}; l < M; ++l)
            {
                double C_ijl {R_ij - K[l]};
                if (C_ijl <= 0) {continue;}
                double current_val {std::log(C_ijl) + beta * Q(l, j)};
                if (current_val > max_val)
                {
                    max_val = current_val;
                    V(i,j) = max_val;
                    G(i,j) = l;
                    C(i,j) = C_ijl;
                }
            }
            I(i,j) = K[G(i,j)] + (delta - 1.0) * K[i];
        }
    }
    return V;
}

void macro_simulator(double alpha, double beta, double delta, double rho, double sigma, const int M, const int N, const int T, const double tolerance, const std::string& file_name)
{
    constexpr double sqrt2 {std::sqrt(2.0)};

    double K_star {std::pow(3.0 * ( 1.0 / beta - 1.0 + delta) / alpha, (1.0 / (alpha - 1.0)))};

    std::vector<double> K (M);
    for (int i {0}; i < M; ++i)
    {
        K[i] = 0.8 * K_star  + 0.4 * (static_cast<double>(i)) * K_star / (static_cast<double>(M-1));

    }

    std::vector<double> a (N);
    for (int j {0}; j < N; ++j)
    {
        a[j] = (3.0*sigma*(static_cast<double>(2*j+1-N))) / (static_cast<double>(N-1) * std::sqrt(1.0 - rho * rho));
    }

    const double da {(a[N-1] - a[0]) / (N -1)};

    MatrixXd P (N,N);
    for (int i{0}; i < N; ++i)
    {
        for (int j{1}; j < N-1; ++j)
        {
            P(i, j) = 0.5 * (std::erf((a[j] + da / 2.0 - rho * a[i]) / (sqrt2 * sigma))- std::erf((a[j] - da / 2.0 - rho * a[i]) / (sqrt2 * sigma)));
        }
        P(i, 0) = 0.5 * (std::erf((a[0] + da / 2.0 - rho * a[i]) / (sqrt2 * sigma)) + 1);
        P(i, N-1) = 0.5 - 0.5 * (std::erf((a[N-1] - da / 2.0 - rho * a[i]) / (sqrt2 * sigma)));
    }

    MatrixXd V_old (M,N);
    V_old.setZero();

    MatrixXd V_new (M,N);
    V_new.setZero();

    MatrixXd C (M,N);
    C.setZero();

    MatrixXi G (M,N);
    G.setZero();

    MatrixXd I (M,N);
    I.setZero();

    do 
    {
        V_old = V_new;
        MatrixXd Q (M, N);
        Q = V_old * P.transpose();
        V_new = max_value(K,a,M,N,alpha,delta,beta,Q,G,C,I);
    }
    while ((V_new - V_old).cwiseAbs().maxCoeff() > tolerance);
// time to simulate the economy

std::ofstream f(file_name);
f << "t,Capital,Real GDP,Consumption,Investment,Productivity \n";

    int i {std::floor(M / 2)};
    int j {std::floor(N / 2)};

    std::mt19937 mt{std::random_device{}()};
    std::uniform_real_distribution U(0.0, 1.0);
    double u {0.0};
    double x {0.0};
    

    for (int k{0}; k < T; ++k)
    {
        f << k << ',' << K[i] << ',' << std::exp(a[j]) * std::pow(K[i],alpha) / 3.0 << ',' << C(i,j) << ',' << I(i,j) << ',' << std::exp(a[j]) << '\n';
        i = G (i, j);
        u = U(mt);
        int m {0};
        x = 0;
        while (x < u)
        {
            x += P(j, m);
            ++m;
        }
        j = m - 1;
        
    }

    f.close();
}

int main()
{
    double alpha {0.33};
    double beta {0.996655};
    double delta {0.008404};
    double rho {0.983048};
    double sigma {0.005872};
    int M {1500};
    int N {21};
    int T {300};
    double eps {1e-6};

    macro_simulator(alpha, beta, delta, rho, sigma, M, N, T, eps, "macro_simulation.csv");
    std::system("notify-send 'Simulation complete' 'RBC model has finished running.'");
    return 0;
}